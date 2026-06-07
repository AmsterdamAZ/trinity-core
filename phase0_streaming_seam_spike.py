#!/usr/bin/env python3
"""
Trinity — Phase-0 Streaming-Seam Spike  (Roadmap Story 0.2, the GO/NO-GO gate)

PURPOSE
  Prove we can tap an *incremental* token/sentence stream out of Hermes's
  synchronous AIAgent loop and segment it into sentences in real time — the one
  load-bearing assumption behind building Trinity's voice layer on a Hermes fork
  (see ADR-001 §3.3, §7). If this works, the per-sentence TTS pipeline is
  feasible. If it does NOT, fall back (see VERDICT section at the bottom).

WHAT IT DOES
  1. Builds a lean AIAgent pointed at Anthropic / Claude Opus (no tools, no MCP).
  2. Installs a `stream_delta_callback` that timestamps every token as it arrives.
  3. Feeds those tokens into a minimal real-time sentence segmenter.
  4. Runs one conversation with a multi-sentence "briefing" style prompt.
  5. Reports: time-to-first-token, time-to-first-*sentence*, delta count, and
     whether text arrived incrementally vs. as one buffered dump.
  6. Prints a GO / NO-GO verdict.

  This is throwaway de-risking code, not production. It is deliberately defensive
  about the exact AIAgent wiring (a 15k-line class) and will tell you what to
  adjust if the constructor or callback name differs in your fork.

HOW TO RUN
  # from inside your forked hermes-agent repo, with its venv active:
  export ANTHROPIC_API_KEY=sk-ant-...
  python phase0_streaming_seam_spike.py \
      --provider anthropic --model claude-opus-4-8

  Adjust --model to whatever Opus id your registry uses. If import or
  construction fails, read the "ADAPT ME" notes inline — they mark the only
  fork-specific bits.

REFERENCES
  Agent Loop Internals — callback surfaces table lists `stream_delta_callback`
  ("Each streaming token (when enabled)"). run_conversation() signature and the
  anthropic_messages API mode are documented there too.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Real-time sentence segmenter (mirrors Voice I/O spec §4.2, minus hold-one-ahead)
# ----------------------------------------------------------------------------
SENTENCE_ENDERS = ".?!"


class SentenceSegmenter:
    """Buffers streamed text; flushes a segment on a sentence boundary or a
    max-char safety break. Emits (segment_text, monotonic_timestamp)."""

    def __init__(self, max_chars: int = 220):
        self.buf: str = ""
        self.max_chars = max_chars
        self.segments: list[tuple[str, float]] = []

    def feed(self, text: str) -> list[str]:
        """Add streamed text, return any sentences that completed on this feed."""
        emitted: list[str] = []
        self.buf += text
        while True:
            cut = self._find_boundary()
            if cut is None:
                break
            seg = self.buf[:cut].strip()
            self.buf = self.buf[cut:].lstrip()
            if seg:
                self.segments.append((seg, time.perf_counter()))
                emitted.append(seg)
        return emitted

    def flush(self) -> str | None:
        """Emit whatever remains at end-of-stream (the would-be final segment)."""
        seg = self.buf.strip()
        self.buf = ""
        if seg:
            self.segments.append((seg, time.perf_counter()))
            return seg
        return None

    def _find_boundary(self) -> int | None:
        # First sentence-ender followed by whitespace/end → boundary just after it.
        for i, ch in enumerate(self.buf):
            if ch in SENTENCE_ENDERS:
                nxt = self.buf[i + 1] if i + 1 < len(self.buf) else ""
                if nxt == "" or nxt.isspace():
                    return i + 1
        # Safety break: sentence running too long with no ender yet.
        if len(self.buf) >= self.max_chars:
            space = self.buf.rfind(" ", 0, self.max_chars)
            return space + 1 if space > 0 else self.max_chars
        return None


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
@dataclass
class StreamMetrics:
    t_start: float = 0.0
    delta_times: list[float] = field(default_factory=list)
    delta_chars: list[int] = field(default_factory=list)

    def on_delta(self, text: str) -> None:
        self.delta_times.append(time.perf_counter())
        self.delta_chars.append(len(text))

    @property
    def num_deltas(self) -> int:
        return len(self.delta_times)

    @property
    def total_chars(self) -> int:
        return sum(self.delta_chars)

    def ttf_delta(self) -> float | None:
        return (self.delta_times[0] - self.t_start) if self.delta_times else None

    def ttl_delta(self) -> float | None:
        return (self.delta_times[-1] - self.t_start) if self.delta_times else None

    def max_gap(self) -> float | None:
        if len(self.delta_times) < 2:
            return None
        return max(b - a for a, b in zip(self.delta_times, self.delta_times[1:]))


# ----------------------------------------------------------------------------
# Agent construction  ——  ADAPT ME if your fork's wiring differs
# ----------------------------------------------------------------------------
def build_agent(provider: str, model: str):
    """Import and construct a lean AIAgent on Anthropic/Opus.

    The AIAgent constructor takes many args and varies across versions. We try a
    minimal kwarg set and degrade gracefully. If this raises, open run_agent.py,
    find the AIAgent.__init__ signature, and set the kwargs to match — that is
    the only fork-specific edit this spike needs.
    """
    try:
        from run_agent import AIAgent  # type: ignore
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            "Could not import run_agent.AIAgent.\n"
            f"  -> {exc}\n"
            "  Run this from inside the forked hermes-agent repo with its venv "
            "active (the same env where `hermes` works)."
        )

    # Provider/model selection. Per the docs, provider 'anthropic' resolves to
    # the anthropic_messages API mode automatically; passing model is enough.
    candidate_kwargs = {
        "provider": provider,
        "model": model,
        # keep it lean: no tools, no MCP — we are testing raw text streaming only
        "tools_enabled": False,
        "enable_memory": False,
    }
    sig = _safe_signature(AIAgent)
    kwargs = _filter_kwargs(candidate_kwargs, sig)
    _report_kwargs(candidate_kwargs, kwargs, sig)

    try:
        return AIAgent(**kwargs)
    except TypeError as exc:
        sys.exit(
            f"AIAgent(**{kwargs}) failed: {exc}\n"
            "  -> ADAPT ME: match build_agent()'s kwargs to your AIAgent.__init__ "
            "signature in run_agent.py."
        )


def install_stream_callback(agent, on_delta) -> bool:
    """Wire our delta handler onto the agent and enable streaming.

    Returns True if a streaming-delta hook was found. The doc names the hook
    `stream_delta_callback` ("each streaming token, when enabled"); we also try a
    couple of fallback names and streaming-enable flags defensively.
    """
    # Defensive wrapper: the callback may be invoked as (text), (text, meta),
    # or with a keyword — coerce whatever we get to a string of new tokens.
    def cb(*args, **kw):
        text = ""
        if args:
            text = args[0] if isinstance(args[0], str) else str(args[0])
        elif "delta" in kw:
            text = str(kw["delta"])
        elif "text" in kw:
            text = str(kw["text"])
        on_delta(text)

    hooked = False
    for name in ("stream_delta_callback", "on_stream_delta", "delta_callback"):
        if hasattr(agent, name):
            setattr(agent, name, cb)
            print(f"[setup] installed delta hook on attribute: {name}")
            hooked = True
            break

    # Try to flip streaming on (name varies; set any that exist).
    for flag in ("stream", "streaming", "enable_streaming", "stream_responses"):
        if hasattr(agent, flag):
            try:
                setattr(agent, flag, True)
                print(f"[setup] enabled streaming via: {flag}=True")
            except Exception:  # noqa: BLE001
                pass

    if not hooked:
        print(
            "[setup] WARNING: no stream-delta attribute found on AIAgent.\n"
            "        -> ADAPT ME: check the Callback Surfaces table / run_agent.py "
            "for the real hook name and add it to install_stream_callback()."
        )
    return hooked


def run_turn(agent, prompt: str):
    """Call run_conversation() (preferred) or chat() as a fallback."""
    if hasattr(agent, "run_conversation"):
        return agent.run_conversation(user_message=prompt)
    if hasattr(agent, "chat"):
        return agent.chat(prompt)
    sys.exit("AIAgent exposes neither run_conversation() nor chat(). ADAPT ME.")


# ----------------------------------------------------------------------------
# small reflection helpers
# ----------------------------------------------------------------------------
def _safe_signature(fn):
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _filter_kwargs(kwargs: dict, sig) -> dict:
    if sig is None:
        return dict(kwargs)  # can't introspect; try them all
    accepted = set(sig.parameters)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kw:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in accepted}


def _report_kwargs(wanted: dict, used: dict, sig) -> None:
    dropped = {k for k in wanted if k not in used}
    if dropped:
        print(f"[setup] note: AIAgent did not accept {sorted(dropped)} — skipped.")
    if sig is not None:
        print(f"[setup] AIAgent accepts: {sorted(sig.parameters)}")


# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------
def verdict(m: StreamMetrics, seg: SentenceSegmenter, hooked: bool) -> None:
    line = "=" * 68
    print("\n" + line)
    print("STREAMING-SEAM SPIKE — RESULTS")
    print(line)

    ttf = m.ttf_delta()
    ttl = m.ttl_delta()
    total_wall = (m.delta_times[-1] - m.t_start) if m.delta_times else 0.0
    first_sentence_t = (
        seg.segments[0][1] - m.t_start if seg.segments else None
    )

    def fmt(x):
        return f"{x*1000:.0f} ms" if isinstance(x, float) else "n/a"

    print(f"  delta hook installed     : {hooked}")
    print(f"  deltas received          : {m.num_deltas}")
    print(f"  total chars streamed     : {m.total_chars}")
    print(f"  time to first token      : {fmt(ttf)}")
    print(f"  time to first SENTENCE   : {fmt(first_sentence_t)}")
    print(f"  time to last token       : {fmt(ttl)}")
    print(f"  largest inter-token gap  : {fmt(m.max_gap())}")
    print(f"  sentences segmented      : {len(seg.segments)}")

    # Incrementality heuristic: many deltas, first arriving well before the last,
    # and at least one full sentence emitted before the stream ended.
    incremental = (
        m.num_deltas >= 5
        and ttf is not None
        and total_wall > 0
        and ttf < 0.8 * total_wall
    )
    sentence_before_end = (
        len(seg.segments) >= 2  # at least one mid-stream + the final flush
    )

    print(line)
    if hooked and incremental and sentence_before_end:
        print("  VERDICT: ✅ GO")
        print("  Incremental tokens confirmed; sentences segmented mid-stream.")
        print("  The in-tree streaming seam is viable — proceed with the voice/")
        print("  entry point (Story 1.3 feeds this segmenter from the real hook).")
    elif hooked and m.num_deltas <= 2:
        print("  VERDICT: ❌ NO-GO (buffered, not incremental)")
        print("  Tokens arrived as one/two dumps — the callback is post-hoc, not")
        print("  a live stream. Before reverting, try the documented streaming")
        print("  paths that DO emit deltas: TUI-gateway `message.delta` events or")
        print("  the API server's SSE /v1/chat/completions (ADR-001 §6 fallback,")
        print("  Roadmap Story 0.4). Only if those also fail, revert to the")
        print("  Claude Agent SDK plan (ADR-001 §7).")
    elif not hooked:
        print("  VERDICT: ⚠️  INCONCLUSIVE — no delta hook was wired.")
        print("  Find the real callback name (Agent Loop 'Callback Surfaces') and")
        print("  re-run. Until then this neither passes nor fails the gate.")
    else:
        print("  VERDICT: ⚠️  WEAK — stream looks partly incremental.")
        print("  Inspect the per-segment log above; tune thresholds or test with a")
        print("  longer multi-sentence prompt before recording GO/NO-GO.")
    print(line)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
DEFAULT_PROMPT = (
    "Give me a four-sentence spoken status briefing on a fictional SaaS app: "
    "active users, revenue trend, one risk, and one recommended action. "
    "Plain sentences, no lists or markdown."
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Trinity Phase-0 streaming-seam spike")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="Opus model id as your registry names it")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-chars", type=int, default=220,
                    help="segmenter safety-break length")
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in the environment first.")

    metrics = StreamMetrics()
    segmenter = SentenceSegmenter(max_chars=args.max_chars)

    def on_delta(text: str) -> None:
        metrics.on_delta(text)
        for sentence in segmenter.feed(text):
            dt = time.perf_counter() - metrics.t_start
            # In the real pipeline this is where the segment is dispatched to TTS.
            print(f"  [{dt*1000:7.0f} ms] → SEGMENT #{len(segmenter.segments)-1}: "
                  f"{sentence!r}")

    print("Building lean AIAgent (Anthropic / Opus, no tools, no MCP)...")
    agent = build_agent(args.provider, args.model)
    hooked = install_stream_callback(agent, on_delta)

    print(f"\nPrompt: {args.prompt}\n")
    print("Streaming segments as they complete:")
    metrics.t_start = time.perf_counter()
    try:
        run_turn(agent, args.prompt)
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        tail = segmenter.flush()
        if tail is not None:
            dt = time.perf_counter() - metrics.t_start
            print(f"  [{dt*1000:7.0f} ms] → SEGMENT #{len(segmenter.segments)-1} "
                  f"(final): {tail!r}")

    verdict(metrics, segmenter, hooked)


if __name__ == "__main__":
    main()
