#!/usr/bin/env python3
"""
Terminal chat loop for the conversational, multi-agent stock research
assistant.

    python chat.py

Ask free-form questions about a stock -- "what is AAPL's book value and
ROE", "should I buy TSLA", "what's the RSI on NVDA right now", "compare
that to what Alpaca shows" -- and keep the conversation going; the agent
remembers what's been discussed and which ticker you're currently talking
about, across turns, for the life of this run.

Behind the scenes each turn is routed through a small team of specialist
agents (fundamentals, technicals, news, a cross-check analyst, and a
strategist for recommendations) coordinated by a supervisor -- see
chat_agent/graph.py for the full architecture. Every turn is traced to
LangSmith if LANGSMITH_TRACING=true is set in .env.

This produces educational analysis, not financial advice. See README.md.
"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from chat_agent.graph import build_graph, run_turn

EXIT_COMMANDS = {"exit", "quit", "q", ":q", "bye"}

BANNER = """\
Stock Research Assistant (multi-agent, conversational)
--------------------------------------------------------
Ask about any stock: fundamentals, technicals, recent news, a cross-check
against Alpaca, or a straight-up Buy/Hold/Sell call. I'll stay decisive
when you ask for a recommendation, name the risks, and leave the final
decision to you.

Type 'exit' (or 'quit') to leave.
"""


def _print_trace_note() -> None:
    import os

    if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
        agent = os.environ.get("LANGSMITH_AGENT_ID", "(unset)")
        env = os.environ.get("LANGSMITH_AGENT_ENVIRONMENT", "(unset)")
        print(f"[tracing on -- LangSmith agent '{agent}', environment '{env}']\n")
    else:
        print("[tracing off -- set LANGSMITH_TRACING=true in .env to see traces in LangSmith]\n")


def main() -> int:
    load_dotenv()
    _print_trace_note()
    print(BANNER)

    graph = build_graph()
    history: list = []
    ticker = None

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("bye.")
            break

        history.append(HumanMessage(content=user_input))

        try:
            result = run_turn(graph, history, ticker)
        except Exception as exc:  # keep the REPL alive on a single bad turn
            print(f"\n[error] something went wrong this turn: {exc}\n")
            history.pop()  # don't leave an unanswered HumanMessage in history
            continue

        history = result["messages"]
        ticker = result["ticker"]

        print(f"\nassistant> {result['reply']}\n")

    try:
        from langsmith.utils import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
