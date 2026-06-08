"""
Streaming sentence segmenter + hold-one-ahead (Story 1.3; Voice I/O spec §4).

Consumes the orchestrator's token stream (the Story 0.2 `stream_delta_callback`
tap) and emits one segment envelope per sentence over an injected sink (the WS in
production). This is what lets TTS start on sentence one while the model is still
writing the rest — the core of the ~1.0-1.5s first-audible-word budget.

Envelope (spec §4.1):
    {"session_id", "segment_id" (0-based monotonic), "text", "is_final", "audio_format"}

Hold-one-ahead (spec §4.3, reconciled with voice-agent-latency-prompt):
  A completed sentence is HELD until we have proof it isn't the last — proof being
  the first content of the *next* sentence. On that proof it's emitted is_final=
  false; the next sentence takes the held slot. When the stream ends, the single
  held segment is emitted is_final=true. Releasing on next-sentence-START (not
  completion) keeps FIRST-segment latency unchanged; only the FINAL segment eats
  "one sentence" of delay — exactly the property the latency prompt calls for.

Wiring (production):
    seg = StreamingSentenceSegmenter(sink=ws_sink, session_id=turn_id)
    agent.stream_delta_callback = seg.feed       # 0.2 tap
    agent.run_conversation(user_message=...)
    seg.finalize()                               # flush held as is_final=true

Target repo path: trinity/voice/segmenter.py
"""
from __future__ import annotations

from typing import Callable

SegmentSink = Callable[[dict], None]

# Words that take a trailing period but do NOT end a sentence. Decimals/currency
# ($2.3M) need no entry here — a boundary requires whitespace after the period,
# which "2.3" never has.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "inc", "corp", "ltd",
    "co", "vs", "etc", "eg", "ie", "approx", "no", "fig", "dept", "est", "gen",
    "sen", "rep", "u.s", "u.k", "a.m", "p.m",
}
_TERMINALS = ".?!"


class StreamingSentenceSegmenter:
    def __init__(self, sink: SegmentSink, session_id: str,
                 max_chars: int = 240, audio_format: str = "pcm_s16le_24k"):
        self._sink = sink
        self.session_id = session_id
        self.max_chars = max_chars
        self.audio_format = audio_format
        self._buf = ""            # in-progress (not-yet-completed) text
        self._held: str | None = None   # the one completed-but-unemitted segment
        self._next_id = 0
        self._finalized = False

    # --- public API -----------------------------------------------------------
    def feed(self, delta: str) -> None:
        """Consume a chunk of streamed model output."""
        if not delta or self._finalized:
            return
        self._buf += delta
        while True:
            seg = self._extract_next()
            if seg is None:
                break
            self._stage(seg)
        # Release the held segment the moment the NEXT sentence has begun
        # (any non-whitespace remainder) — keeps first-segment latency unchanged.
        if self._held is not None and self._buf.strip():
            self._emit(self._held, is_final=False)
            self._held = None

    def finalize(self) -> None:
        """End the stream: flush the trailing partial, emit the held as is_final."""
        if self._finalized:
            return
        self._finalized = True
        leftover = self._buf.strip()
        self._buf = ""
        if leftover:
            self._stage(leftover)
        if self._held is not None:
            self._emit(self._held, is_final=True)
            self._held = None
        else:
            # Empty turn: still emit a terminal envelope so the client can close.
            self._emit("", is_final=True)

    # --- internals --------------------------------------------------------------
    def _stage(self, seg: str) -> None:
        # A newer sentence completed => the currently-held one is provably not last.
        if self._held is not None:
            self._emit(self._held, is_final=False)
        self._held = seg

    def _emit(self, text: str, *, is_final: bool) -> None:
        self._sink({
            "session_id": self.session_id,
            "segment_id": self._next_id,
            "text": text,
            "is_final": is_final,
            "audio_format": self.audio_format,
        })
        self._next_id += 1

    def _extract_next(self) -> str | None:
        """Pull the leftmost complete sentence from the buffer, or force a safe
        break if the buffer runs past max_chars with no boundary. None if neither."""
        idx = self._find_boundary(self._buf)
        if idx is not None:
            seg = self._buf[:idx + 1].strip()
            self._buf = self._buf[idx + 1:].lstrip()
            return seg or None
        if len(self._buf) >= self.max_chars:
            cut = self._safe_break(self._buf, self.max_chars)
            seg = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            return seg or None
        return None

    def _find_boundary(self, buf: str) -> int | None:
        """Index of a sentence-ending punctuation that is followed by whitespace
        and is not an abbreviation. Requiring trailing whitespace defers the call
        until the next char arrives, which also rules out decimals ($2.3M)."""
        for i, ch in enumerate(buf):
            if ch not in _TERMINALS:
                continue
            if i + 1 >= len(buf):
                break                      # last char: can't confirm yet -> wait
            if not buf[i + 1].isspace():
                continue                   # e.g. the '.' in 2.3 -> not a boundary
            if ch == "." and self._is_abbrev(buf, i):
                continue
            return i
        return None

    @staticmethod
    def _is_abbrev(buf: str, i: int) -> bool:
        j, chars = i - 1, []
        while j >= 0 and (buf[j].isalpha() or buf[j] == "."):
            chars.append(buf[j])
            j -= 1
        word = "".join(reversed(chars)).strip(".").lower()
        return word in _ABBREV

    @staticmethod
    def _safe_break(buf: str, limit: int) -> int:
        window = buf[:limit]
        cut = max(window.rfind(" "), window.rfind("\n"),
                  window.rfind(","), window.rfind(";"))
        return cut + 1 if cut > 0 else limit


def collecting_sink() -> tuple[list[dict], SegmentSink]:
    """Test/inspection helper: returns (list, sink) where sink appends envelopes."""
    out: list[dict] = []
    return out, out.append


if __name__ == "__main__":
    import json

    text = ("Your apps are healthy. Verbella has 42,000 active users, up 8 percent. "
            "MRR crossed $2.3M but growth is slowing. Dr. Lee flagged a competitor "
            "undercutting on price.")
    print("Streaming the briefing token-by-token:\n")
    seg = StreamingSentenceSegmenter(sink=lambda e: print("  ->", json.dumps(e)),
                                     session_id="demo")
    for word in text.split(" "):
        seg.feed(word + " ")
    seg.finalize()
