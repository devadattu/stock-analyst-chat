"""
LangSmith evaluator functions for the conversational stock chat assistant.

Each evaluator takes a LangSmith `run` (what the target function produced)
and the `example` it ran on, and returns a dict (or list of dicts) shaped
like {"key": ..., "score": ..., "comment": ...} -- the standard
function-evaluator signature for `langsmith.evaluate`.

These are intentionally cheap/deterministic (no extra LLM calls) so the
eval suite is fast and doesn't introduce its own hallucination risk. What
they check is specific to THIS project's two explicit design goals (per
the user's own ask): the agent should correctly figure out which ticker is
being discussed, and when asked for a recommendation it should actually
take a decisive position -- never hide behind vague hedging -- while still
making clear the final call belongs to the user.
"""

from __future__ import annotations

import re
from typing import Any

RATING_LABELS = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]

# Phrases that would mean the strategist dodged the question instead of
# taking a position. A *reasoned* Hold is fine (that's a real position);
# refusing to have a view at all is what we're checking for here.
_HEDGE_PHRASES = [
    "i cannot provide financial advice",
    "i'm not able to give a recommendation",
    "it depends on your personal situation",
    "consult a financial advisor before deciding",
    "i can't predict the market",
]

# Phrasing that signals the agent correctly left the final call to the user,
# which the user explicitly asked be preserved alongside the decisive lean.
_DECISION_IS_YOURS_PHRASES = [
    "yours", "your call", "your decision", "up to you", "your own decision",
]


def _get_outputs(run: Any) -> dict[str, Any]:
    outputs = getattr(run, "outputs", None)
    if outputs is None and isinstance(run, dict):
        outputs = run.get("outputs")
    return outputs or {}


def ticker_correctly_identified(run: Any, example: Any = None) -> dict[str, Any]:
    """Did the supervisor correctly identify/keep the ticker the question
    was actually about?"""
    outputs = _get_outputs(run)
    expected = (getattr(example, "inputs", None) or {}).get("ticker", "") if example is not None else ""
    actual = (outputs.get("ticker") or "").upper()
    passed = bool(expected) and actual == expected.upper()
    return {
        "key": "ticker_correctly_identified",
        "score": 1.0 if passed else 0.0,
        "comment": f"expected {expected!r}, got {actual!r}",
    }


def recommendation_is_decisive(run: Any, example: Any = None) -> dict[str, Any]:
    """For questions that explicitly ask for a recommendation, the reply
    should (a) state one of the rating labels, (b) contain none of the
    known hedge-and-dodge phrases, and (c) still make clear the decision
    belongs to the user. Scores 1.0 for each satisfied, averaged -- and is
    skipped (score 1.0, noted) for questions that weren't asking for one."""
    inputs = (getattr(example, "inputs", None) or {}) if example is not None else {}
    if not inputs.get("expects_recommendation"):
        return {"key": "recommendation_is_decisive", "score": 1.0, "comment": "not a recommendation question; skipped"}

    outputs = _get_outputs(run)
    reply = (outputs.get("reply", "") or "").lower()
    rating = outputs.get("rating") or {}
    rating_word = str(rating.get("rating", "")).lower()

    checks = {
        "states_a_rating_label": any(label.lower() in reply for label in RATING_LABELS) and bool(rating_word),
        "no_dodge_phrases": not any(p in reply for p in _HEDGE_PHRASES),
        "leaves_decision_to_user": any(p in reply for p in _DECISION_IS_YOURS_PHRASES),
    }
    passed = sum(checks.values())
    return {
        "key": "recommendation_is_decisive",
        "score": round(passed / len(checks), 2),
        "comment": f"checks: {checks}",
    }


def reply_is_grounded_in_findings(run: Any, example: Any = None) -> dict[str, Any]:
    """Lightweight faithfulness tripwire: every dollar/percent figure in the
    reply should trace back to a number that actually appeared in the
    rating's own underlying tool data (when a rating was computed this
    turn). Approximate string/number matching, not full semantic
    verification -- same spirit as the first project's evaluator."""
    outputs = _get_outputs(run)
    reply = outputs.get("reply", "") or ""
    rating = outputs.get("rating") or {}

    figures = re.findall(r"[\$]\s?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s?%", reply)
    flat_figures = [a or b for a, b in figures]
    if not flat_figures:
        return {"key": "reply_is_grounded_in_findings", "score": 1.0, "comment": "no $/% figures to check"}
    if not rating:
        return {"key": "reply_is_grounded_in_findings", "score": 1.0, "comment": "no rating computed this turn; skipped"}

    source_numbers: set[str] = set()
    inputs = rating.get("inputs", {}) if isinstance(rating, dict) else {}
    for tool_output in inputs.values():
        if isinstance(tool_output, dict):
            for v in tool_output.values():
                if isinstance(v, (int, float)):
                    source_numbers.add(f"{v:.2f}".rstrip("0").rstrip("."))
                    source_numbers.add(str(round(v)))
                    source_numbers.add(f"{v * 100:.0f}")  # ratios reported as percentages

    unsourced = [f for f in flat_figures if f not in source_numbers and f.rstrip("0").rstrip(".") not in source_numbers]
    score = 1.0 - (len(unsourced) / len(flat_figures))
    return {
        "key": "reply_is_grounded_in_findings",
        "score": round(max(score, 0.0), 2),
        "comment": f"unsourced figures: {unsourced}" if unsourced else "all figures traced to tool data",
    }


ALL_EVALUATORS = [ticker_correctly_identified, recommendation_is_decisive, reply_is_grounded_in_findings]
