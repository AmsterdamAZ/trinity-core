"""
Trinity eval harness — task schema + deterministic graders (Story 0.3, Half B).

Deterministic graders only here: accuracy, turn count, tokens, latency. The
LLM-as-judge grader is the next addition. Tasks are declared in tasks.py and
executed by runner.py against the model resolved for each task's role.

Target repo path: trinity/evals/harness.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---- Task schema ------------------------------------------------------------
@dataclass(frozen=True)
class AccuracySpec:
    """Declarative accuracy check over the response text. All set fields must hold."""
    must_contain: tuple[str, ...] = ()           # all present (case-insensitive)
    must_not_contain: tuple[str, ...] = ()       # none present (case-insensitive)
    regex: Optional[str] = None                  # must match (search, case-insensitive)
    not_regex: Optional[str] = None              # must NOT match
    min_sentences: Optional[int] = None          # rough sentence-count floor
    predicate: Optional[Callable[[str], bool]] = None  # arbitrary check; True = pass


@dataclass(frozen=True)
class EvalTask:
    id: str
    category: str                # "regression" | "failure_mode"
    role: str                    # model role to resolve (orchestrator/router/bulk/...)
    prompt: str
    accuracy: AccuracySpec = field(default_factory=AccuracySpec)
    max_turns: Optional[int] = None
    max_tokens: Optional[int] = None
    max_latency_ms: Optional[int] = None      # soft: total completion budget
    max_ttf_ms: Optional[int] = None          # soft: time-to-first-token budget (~ voice target)
    judge_rubric: Optional[str] = None        # if set, the LLM-judge grader runs with this rubric
    judge_threshold: float = 0.7              # judge passes when score >= this


# ---- Run records ------------------------------------------------------------
@dataclass
class GraderResult:
    name: str
    passed: bool
    detail: str = ""
    soft: bool = False     # soft graders (e.g. latency) are recorded + reported but do
                           # NOT fail the task or trip the regression gate


@dataclass
class TaskResult:
    task_id: str
    category: str
    role: str
    model: str
    response: str
    latency_ms: float          # total turn completion (wall clock)
    ttf_ms: Optional[float]    # time to first streamed token — the voice-relevant metric
    turns: Optional[int]
    tokens: Optional[int]
    chars: int
    graders: list[GraderResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        hard = [g for g in self.graders if not g.soft]
        return bool(hard) and all(g.passed for g in hard)


# ---- Deterministic graders --------------------------------------------------
_SENTENCE = re.compile(r"[.!?](?:\s|$)")


def grade_accuracy(response: str, spec: AccuracySpec) -> GraderResult:
    low = response.lower()
    fails: list[str] = []
    for s in spec.must_contain:
        if s.lower() not in low:
            fails.append(f"missing {s!r}")
    for s in spec.must_not_contain:
        if s.lower() in low:
            fails.append(f"present {s!r}")
    if spec.regex and not re.search(spec.regex, response, re.I):
        fails.append(f"no match /{spec.regex}/")
    if spec.not_regex and re.search(spec.not_regex, response, re.I):
        fails.append(f"matched forbidden /{spec.not_regex}/")
    if spec.min_sentences is not None:
        n = len(_SENTENCE.findall(response))
        if n < spec.min_sentences:
            fails.append(f"{n} sentences < {spec.min_sentences}")
    if spec.predicate is not None:
        try:
            if not spec.predicate(response):
                fails.append("predicate failed")
        except Exception as e:  # noqa: BLE001
            fails.append(f"predicate error: {e}")
    return GraderResult("accuracy", not fails, "; ".join(fails) or "ok")


def grade_turns(turns: Optional[int], max_turns: Optional[int]) -> Optional[GraderResult]:
    if max_turns is None:
        return None
    if turns is None:
        return GraderResult("turns", True, "turns not captured (skipped)")
    return GraderResult("turns", turns <= max_turns, f"{turns} <= {max_turns}")


def grade_tokens(tokens: Optional[int], max_tokens: Optional[int]) -> Optional[GraderResult]:
    if max_tokens is None:
        return None
    if tokens is None:
        return GraderResult("tokens", True, "tokens not captured (skipped — wire usage)")
    return GraderResult("tokens", tokens <= max_tokens, f"{tokens} <= {max_tokens}")


def grade_latency(latency_ms: float, max_latency_ms: Optional[int]) -> Optional[GraderResult]:
    # SOFT grader: recorded and reported (per the Story 0.3 grader list) but excluded
    # from pass/fail — cold starts and network jitter make latency noisy, and a single
    # spike should not fail a task whose output is correct. The meaningful latency
    # target is the voice-loop TTFT, measured separately.
    if max_latency_ms is None:
        return None
    return GraderResult("latency", latency_ms <= max_latency_ms,
                        f"{latency_ms:.0f}ms <= {max_latency_ms}ms", soft=True)


def grade_ttf(ttf_ms: Optional[float], max_ttf_ms: Optional[int]) -> Optional[GraderResult]:
    # SOFT grader: time to first streamed token — the voice-relevant latency metric
    # (maps to the ~1.0-1.5s first-audible-word target). Soft for the same reason as
    # latency: timing is noisy and shouldn't fail a correct response.
    if max_ttf_ms is None:
        return None
    if ttf_ms is None:
        return GraderResult("ttf", True, "ttf not captured (skipped)", soft=True)
    return GraderResult("ttf", ttf_ms <= max_ttf_ms,
                        f"{ttf_ms:.0f}ms <= {max_ttf_ms}ms", soft=True)


def grade(task: EvalTask, result: TaskResult) -> None:
    """Apply all applicable graders, appending to result.graders in place."""
    result.graders.append(grade_accuracy(result.response, task.accuracy))
    for g in (
        grade_turns(result.turns, task.max_turns),
        grade_tokens(result.tokens, task.max_tokens),
        grade_latency(result.latency_ms, task.max_latency_ms),
        grade_ttf(result.ttf_ms, task.max_ttf_ms),
    ):
        if g is not None:
            result.graders.append(g)
