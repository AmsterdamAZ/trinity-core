"""
Trinity LLM-as-judge grader (Story 0.3, Half B — final piece).

For open-ended tasks where substring/regex checks are too crude (e.g. "is this
briefing actually accurate and concise?"), an LLM scores the response against a
rubric. Runs on the registry's `judge` role (Opus) at temperature 0. HARD grader
by default — quality regressions are exactly what the eval-gated swap workflow
must catch.

Construction reuses the proven lean-AIAgent path, specialized for grading:
no persona (load_soul_identity=False), no tools, no memory, an ephemeral grading
system prompt, response captured via stream_delta_callback. Verdict is strict
JSON {"score", "pass", "reasoning"}, parsed defensively.

Target repo path: trinity/evals/judge.py
"""
from __future__ import annotations

import inspect
import json
import re
import sys

from trinity.model_registry import registry

from .harness import GraderResult

JUDGE_SYSTEM = (
    "You are a strict, fair evaluator of AI assistant responses. You are given a TASK, "
    "the assistant's RESPONSE, and a RUBRIC. Score how well the response satisfies the "
    "rubric. Be specific and unforgiving about real failures, but do not penalize style "
    "choices the rubric does not mention. Respond with ONLY a JSON object — no preamble, "
    "no markdown fences:\n"
    '{"score": <number 0.0-1.0>, "pass": <true|false>, "reasoning": "<one or two sentences>"}'
)


def _filter_kwargs(kwargs: dict, fn) -> dict:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _parse_verdict(text: str) -> dict:
    """Extract the JSON verdict, tolerant of stray prose or code fences."""
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.I).strip()
    m = re.search(r"\{.*\}", cleaned, re.S)   # first {...} block
    if not m:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    return json.loads(m.group(0))


def judge_response(task_prompt: str, response: str, rubric: str,
                   threshold: float = 0.7, *, soft: bool = False) -> GraderResult:
    """Score `response` against `rubric` with the judge model. Returns a GraderResult."""
    try:
        from run_agent import AIAgent  # lazy: needs the trinity-core venv
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Could not import run_agent.AIAgent ({exc}). Run inside the trinity-core venv.")

    jm = registry().model_for("judge")
    chunks: list[str] = []

    def on_delta(*a, **k):
        chunks.append(a[0] if a and isinstance(a[0], str) else str(k.get("delta", k.get("text", ""))))

    want = {
        "provider": jm.provider,
        "model": jm.model,
        "load_soul_identity": False,                  # grade neutrally — no Trinity persona
        "skip_memory": True,
        "enabled_toolsets": [],
        "quiet_mode": True,
        "ephemeral_system_prompt": JUDGE_SYSTEM,
        "request_overrides": {"temperature": 0},      # best-effort deterministic grading
        "stream_delta_callback": on_delta,
    }
    agent = AIAgent(**_filter_kwargs(want, AIAgent.__init__))
    if hasattr(agent, "stream_delta_callback") and getattr(agent, "stream_delta_callback") is None:
        agent.stream_delta_callback = on_delta

    grading_prompt = (
        f"TASK:\n{task_prompt}\n\n"
        f"RESPONSE:\n{response}\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        "Return your JSON verdict now."
    )
    agent.run_conversation(user_message=grading_prompt)

    raw = "".join(chunks).strip()
    try:
        v = _parse_verdict(raw)
        score = float(v.get("score", 0.0))
        reason = str(v.get("reasoning", ""))[:300]
    except Exception as e:  # noqa: BLE001
        # Unparseable verdict = fail loud, but don't crash the whole suite.
        return GraderResult("judge", False, f"unparseable judge output: {e}", soft=soft)

    passed = score >= threshold
    return GraderResult("judge", passed, f"score={score:.2f} (>= {threshold}); {reason}", soft=soft)
