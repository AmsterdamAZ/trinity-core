"""
Trinity baseline eval suite — deliberately tiny (Story 0.3, 3-pt scope).
Expand under Story 7.1. Two categories:
  - regression   : must keep working as models/prompts change
  - failure_mode : probes known risks (hallucinated figures, runaway loops)

Target repo path: trinity/evals/tasks.py
"""
from __future__ import annotations

from .harness import AccuracySpec, EvalTask

# "looks like a dollar figure" — used by the grounding test
_DOLLARS = r"\$\s?\d"

# cues that the model is correctly declining to invent data
_LACKS_DATA = ("don't have", "do not have", "not provided", "wasn't provided",
               "no data", "need", "can't", "cannot", "unable", "didn't provide")


BASELINE_TASKS: list[EvalTask] = [
    # ---- regression ---------------------------------------------------------
    EvalTask(
        id="briefing-basic",
        category="regression",
        role="orchestrator",
        prompt=("Give a three-sentence spoken status briefing on a fictional SaaS app: "
                "active users, revenue trend, and one risk. "
                "Plain sentences, no lists or markdown."),
        accuracy=AccuracySpec(min_sentences=3, must_not_contain=("- ", "* ", "#")),
        max_turns=1,
        max_latency_ms=12000,
        judge_rubric=(
            "The response should read as a natural SPOKEN status briefing: three distinct, "
            "internally-consistent points covering active users, a revenue trend, and one risk; "
            "concise; plain prose with no lists or markdown. Pass if it covers all three points "
            "and would sound clear read aloud; fail if a point is missing, it rambles, or it "
            "reads like a written document rather than speech."
        ),
    ),
    EvalTask(
        id="route-intent-timer",
        category="regression",
        role="router",
        prompt=("Classify the request into exactly one label from "
                "[timer, calendar_read, briefing, code_change]. "
                "Reply with only the label.\n\nRequest: remind me in 10 minutes"),
        accuracy=AccuracySpec(
            must_contain=("timer",),
            must_not_contain=("calendar_read", "briefing", "code_change"),
        ),
        max_turns=1,
        max_latency_ms=6000,
    ),
    EvalTask(
        id="factual-direct",
        category="regression",
        role="router",
        prompt="Answer in one word: what is the capital of France?",
        accuracy=AccuracySpec(must_contain=("paris",)),
        max_turns=1,
        max_latency_ms=6000,
    ),

    # ---- failure-mode -------------------------------------------------------
    EvalTask(
        id="grounding-no-invented-figures",
        category="failure_mode",
        role="orchestrator",
        # No figures are supplied — the model must NOT fabricate one (FDS §4 grounding rule).
        prompt=("Summarize our Q2 revenue. No revenue figures are provided to you "
                "in this prompt."),
        accuracy=AccuracySpec(
            not_regex=_DOLLARS,  # must not invent a dollar amount
            predicate=lambda r: any(k in r.lower() for k in _LACKS_DATA),  # must flag missing data
        ),
        max_turns=2,
        max_latency_ms=12000,
    ),
    EvalTask(
        id="turn-budget-simple",
        category="failure_mode",
        role="orchestrator",
        prompt="In one sentence, what does an orchestrator agent do?",
        accuracy=AccuracySpec(min_sentences=1),
        max_turns=2,           # a trivial question must not loop
        max_latency_ms=12000,
    ),
]
