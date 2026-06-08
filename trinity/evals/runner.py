"""
Trinity eval runner — executes tasks against role-resolved models, grades them,
records a baseline, and gates regressions (Story 0.3, Half B).

The Hermes coupling lives here. The `AIAgent` import is lazy, so harness.py and
tasks.py import fine without the Hermes env. Design choices, all defensive about
v0.16.0 internals (mirrors the Phase-0 spike):
  - Response text   : captured via the PROVEN `stream_delta_callback`.
  - Latency         : wall-clock around run_conversation (always reliable).
  - Turn count      : best-effort via `step_callback`.
  - Token usage     : best-effort (TODO: wire real usage from runtime_provider).
  - Unknown kwargs  : filtered against the AIAgent signature, never passed blindly.

Target repo path: trinity/evals/runner.py

Run (from the repo root, venv active):
    python -m trinity.evals.runner --list            # structural check, no API calls, no key needed
    $env:ANTHROPIC_API_KEY="..."; python -m trinity.evals.runner --record-baseline   # record baseline (cheap calls)
    python -m trinity.evals.runner                   # regression check vs baseline (the CI gate)
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from trinity.model_registry import registry

from .harness import EvalTask, TaskResult, grade
from .tasks import BASELINE_TASKS

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_FILE = RESULTS_DIR / "baseline.json"
LATEST_FILE = RESULTS_DIR / "latest.json"


def _filter_kwargs(kwargs: dict, fn) -> dict:
    """Drop kwargs the target doesn't accept (v0.16.0-safe; e.g. tools_enabled is gone)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _best_effort_tokens(agent) -> int | None:
    """TODO: wire real token usage from Hermes/runtime_provider. Until then, try a
    few common attribute names; return None if not found (token grader then skips)."""
    for attr in ("last_usage", "usage", "total_tokens", "token_usage"):
        v = getattr(agent, attr, None)
        if isinstance(v, int):
            return v
        if isinstance(v, dict):
            for k in ("total_tokens", "output_tokens", "completion_tokens"):
                if isinstance(v.get(k), int):
                    return v[k]
    return None


def run_task(task: EvalTask) -> TaskResult:
    """Build an AIAgent for the task's role, run one turn, collect metrics, grade."""
    try:
        from run_agent import AIAgent  # lazy: needs the trinity-core venv
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Could not import run_agent.AIAgent ({exc}). Run inside the trinity-core venv.")

    rm = registry().model_for(task.role)

    chunks: list[str] = []
    turns = {"n": 0}
    first_delta = {"t": None}

    def on_delta(*a, **k):
        if first_delta["t"] is None:
            first_delta["t"] = time.perf_counter()
        chunks.append(a[0] if a and isinstance(a[0], str) else str(k.get("delta", k.get("text", ""))))

    def on_step(*a, **k):
        turns["n"] += 1

    # Lean, deterministic construction. Unknown kwargs are dropped.
    want = {
        "provider": rm.provider,
        "model": rm.model,
        "skip_memory": True,
        "enabled_toolsets": [],          # ADAPT: confirm [] = tool-free; else use disabled_toolsets
        "quiet_mode": True,
        "stream_delta_callback": on_delta,
        "step_callback": on_step,
    }
    agent = AIAgent(**_filter_kwargs(want, AIAgent.__init__))

    # Hooks may attach as attributes rather than ctor kwargs — set defensively.
    for name, cb in (("stream_delta_callback", on_delta), ("step_callback", on_step)):
        if hasattr(agent, name) and getattr(agent, name) is None:
            setattr(agent, name, cb)

    t0 = time.perf_counter()
    agent.run_conversation(user_message=task.prompt)
    latency_ms = (time.perf_counter() - t0) * 1000
    ttf_ms = (first_delta["t"] - t0) * 1000 if first_delta["t"] is not None else None

    response = "".join(chunks).strip()
    result = TaskResult(
        task_id=task.id, category=task.category, role=task.role, model=rm.model,
        response=response, latency_ms=latency_ms, ttf_ms=ttf_ms,
        turns=turns["n"] or None, tokens=_best_effort_tokens(agent), chars=len(response),
    )
    grade(task, result)
    if task.judge_rubric:
        from .judge import judge_response  # lazy: only import when a task needs it
        result.graders.append(
            judge_response(task.prompt, result.response, task.judge_rubric, task.judge_threshold)
        )
    return result


def run_suite(tasks=BASELINE_TASKS) -> list[TaskResult]:
    results: list[TaskResult] = []
    for t in tasks:
        print(f"… {t.id} ({t.category}, role={t.role})")
        r = run_task(t)
        ttf = f"{r.ttf_ms:.0f}ms" if r.ttf_ms is not None else "n/a"
        print(f"   {'PASS' if r.passed else 'FAIL'}  "
              f"ttf={ttf}  total={r.latency_ms:.0f}ms  turns={r.turns}  tokens={r.tokens}")
        for g in r.graders:
            if not g.passed:
                mark = "~ (soft)" if g.soft else "x"
                print(f"      {mark} {g.name}: {g.detail}")
            elif g.name == "judge":
                print(f"      . judge: {g.detail}")
        results.append(r)
    return results


def _serialize(results):
    return [{**asdict(r), "passed": r.passed} for r in results]


def write_results(results, path: Path):
    RESULTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(_serialize(results), indent=2), encoding="utf-8")


def compare_to_baseline(results) -> int:
    """CI gate: fail if any task that PASSED in baseline now FAILS. Returns exit code."""
    if not BASELINE_FILE.exists():
        print("No baseline recorded yet — run with --record-baseline first.")
        return 0
    baseline = {r["task_id"]: r["passed"] for r in json.loads(BASELINE_FILE.read_text())}
    regressions = [r.task_id for r in results if baseline.get(r.task_id) and not r.passed]
    if regressions:
        print(f"\nREGRESSION: {regressions} passed in baseline but fail now.")
        return 1
    print("\nNo regressions against baseline.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Trinity eval harness")
    ap.add_argument("--list", action="store_true", help="list tasks and exit (no API calls)")
    ap.add_argument("--record-baseline", action="store_true", help="run and save results as the baseline")
    args = ap.parse_args()

    if args.list or not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{len(BASELINE_TASKS)} tasks:")
        for t in BASELINE_TASKS:
            print(f"  [{t.category:12s}] {t.id:34s} role={t.role}")
        if not args.list:
            print("\n(ANTHROPIC_API_KEY not set — structural check only. Set it to run for real.)")
        return

    results = run_suite()
    write_results(results, LATEST_FILE)
    print(f"\nResults -> {LATEST_FILE}")
    if args.record_baseline:
        write_results(results, BASELINE_FILE)
        print(f"Baseline recorded -> {BASELINE_FILE}")
        return
    sys.exit(compare_to_baseline(results))


if __name__ == "__main__":
    main()
