#!/usr/bin/env python3
"""
Run the LangSmith eval suite against the conversational stock chat agent.

    python -m evals.run_eval

Requires ANTHROPIC_API_KEY + LANGSMITH_API_KEY in your environment/.env
(`evaluate()` logs the experiment to LangSmith regardless of whether
LANGSMITH_TRACING is "true" for this run). Results show up in LangSmith
under the dataset's "Experiments" tab, with a score per evaluator per
example, so you can compare experiments across prompt/model changes.
"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith.evaluation import evaluate

from chat_agent.graph import build_graph, run_turn
from evals.dataset import DATASET_NAME, ensure_dataset
from evals.evaluators import ALL_EVALUATORS


def target(inputs: dict) -> dict:
    """LangSmith calls this once per dataset example with `inputs`.

    Each example is a single, fresh turn -- a brand-new graph and an empty
    history containing only this one question -- so examples don't leak
    context into each other."""
    graph = build_graph()
    history = [HumanMessage(content=inputs["question"])]
    result = run_turn(graph, history, ticker=None)
    return {"reply": result["reply"], "ticker": result["ticker"], "rating": result["rating"]}


def main() -> int:
    load_dotenv()
    ensure_dataset()

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix="stock-chat",
        description="Ticker identification, recommendation decisiveness, and groundedness checks.",
    )
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
