"""
Multi-agent LangGraph workflow for the conversational stock analyst.

Shape (each node is its own span in the LangSmith trace):

    START -> supervisor -> (routes to one of) -> fundamentals_analyst
                                                -> technicals_analyst
                                                -> news_analyst
                                                -> crosscheck_analyst
                                                -> strategist
             ^__________________________________________|  (each specialist
                                                              returns to the
                                                              supervisor)
    supervisor -> respond -> END   (once supervisor decides it has enough)

The supervisor is an LLM making a routing decision each step (which
specialist to call next, or finish and respond), using LangChain's
structured-output feature so the decision is a typed dict, not text to
parse. Each specialist is its own LLM call, bound only to its own narrow
toolset, running a small internal tool-calling loop. Specialists write a
short "finding" string to a shared scratchpad (`findings`) rather than
polluting the user-facing conversation directly; `respond` reads that
scratchpad once the supervisor is satisfied and composes the single
assistant-facing reply.

`findings` and `next` are turn-scoped: chat.py resets `findings` to []
before every new user turn, so this scratchpad never leaks across turns —
only `messages` (the real conversation) and `ticker` (what's currently
being discussed) persist, and chat.py manages that persistence by simply
re-feeding the growing message history into `invoke()` each turn.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from chat_agent.alpaca_tools import CROSSCHECK_TOOLS
from chat_agent.signals import compute_rating
from chat_agent.tools import FUNDAMENTALS_TOOLS, NEWS_TOOLS, TECHNICALS_TOOLS

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

SPECIALIST_DESCRIPTIONS = {
    "fundamentals_analyst": "Valuation, profitability, growth, liquidity, leverage, book value per share, ROE, ROA, free cash flow.",
    "technicals_analyst": "Price action, trend, moving averages, RSI, MACD, Bollinger Bands.",
    "news_analyst": "Recent news headlines for the company.",
    "crosscheck_analyst": "Independently verifies a price against Alpaca Markets' data, to catch disagreement between data sources. Route here when the user asks to verify/double-check a number, or before a recommendation that should rest on confirmed data.",
    "strategist": "Synthesizes whatever has been gathered into a decisive Buy/Hold/Sell-style recommendation with reasoning and risks. Route here ONLY when the user is explicitly asking for a recommendation, opinion, or 'should I buy/sell' — not for a plain factual lookup.",
}


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    ticker: str | None
    findings: list[str]
    next: str
    rating: dict[str, Any] | None


class RouteDecision(TypedDict):
    next: str
    ticker: str | None
    reasoning: str


def _build_llm(model: str | None = None, temperature: float = 0.0) -> ChatAnthropic:
    return ChatAnthropic(model=model or DEFAULT_MODEL, temperature=temperature)


def _run_tool_calling_loop(llm_with_tools, tools_by_name: dict, messages: list, max_iterations: int = 3):
    """A small, self-contained tool-calling loop for one specialist's turn.
    Returns the final AIMessage once the model stops requesting tools (or
    the iteration cap is hit)."""
    messages = list(messages)
    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return response
        for tool_call in tool_calls:
            tool_fn = tools_by_name.get(tool_call["name"])
            if tool_fn is None:
                result = {"error": f"unknown tool {tool_call['name']}"}
            else:
                try:
                    result = tool_fn.invoke(tool_call["args"])
                except Exception as exc:  # keep the loop going even if one tool call fails
                    result = {"error": str(exc)}
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    last = messages[-1]
    return last if isinstance(last, AIMessage) else AIMessage(content="(reached tool-call limit before finishing)")


def _make_specialist_node(display_name: str, tools: list, focus: str):
    tools_by_name = {t.name: t for t in tools}
    system_prompt = (
        f"You are the {display_name} on a stock research team. Your focus: {focus}\n"
        "Use your tools to answer the specific question the team supervisor routed to you. "
        "Be concise and precise, and cite the actual numbers your tools return — never "
        "estimate or invent a figure. If a tool errors or a figure is unavailable, say so "
        "plainly rather than guessing."
    )

    def node(state: ChatState) -> dict:
        llm = _build_llm()
        llm_with_tools = llm.bind_tools(tools)
        ticker_note = f"The stock currently being discussed is {state['ticker']}." if state.get("ticker") else ""
        local_messages = [SystemMessage(content=f"{system_prompt}\n{ticker_note}"), *state["messages"][-6:]]
        final = _run_tool_calling_loop(llm_with_tools, tools_by_name, local_messages)
        finding = f"[{display_name}] {final.content}"
        return {"findings": state["findings"] + [finding]}

    return node


STRATEGIST_SYSTEM_PROMPT = """You are a seasoned equity strategist on a stock research team.

You've been given a deterministic, rule-based rating (Buy/Hold/Sell-style,
with a confidence level, an overall score, and a full per-category
breakdown with reasons). Your job is to turn that into a short, direct,
expert-toned take — the way a real analyst would, not a hedge-everything
chatbot:

- State your lean clearly (use the given rating's label) — do not soften
  it into vague "it depends" language.
- Give the 2-3 strongest reasons FOR that lean, grounded in the breakdown.
- Name the 1-2 biggest risks or counterarguments honestly, even though
  they cut against your stated lean.
- Close by making clear this is analysis to inform the user's own decision,
  not a directive — the call is theirs, not yours.

Report the given rating label, confidence, and score exactly as provided —
never substitute your own number."""


def _strategist_node(state: ChatState) -> dict:
    ticker = state.get("ticker")
    if not ticker:
        finding = "[strategist] No company/ticker has been identified yet in this conversation — ask the user which one they mean before giving a recommendation."
        return {"findings": state["findings"] + [finding]}

    rating = compute_rating(ticker)
    rating_text = (
        f"Deterministic rating for {rating['ticker']}: {rating['rating']} "
        f"(confidence {rating['confidence']}, overall score {rating['overall_score']})\n"
        f"Breakdown: {rating['breakdown']}\n"
        f"Disclaimer to include: {rating['disclaimer']}"
    )
    llm = _build_llm()
    response = llm.invoke([SystemMessage(content=STRATEGIST_SYSTEM_PROMPT), HumanMessage(content=rating_text)])
    finding = f"[strategist] {response.content}"
    return {"findings": state["findings"] + [finding], "rating": rating}


SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor of a stock research team with these specialists:

{specialist_list}

Current company/ticker being discussed: {ticker}
Findings gathered so far this turn:
{findings}

Look at the user's latest message and the findings so far. Decide the
single best next step:
- If another specialist's input would materially improve the answer and
  hasn't been gathered yet this turn, route to them.
- If the user's message is conversational/general and needs no live data
  (a greeting, a conceptual question you can answer from general
  knowledge, a follow-up you can already answer from findings above),
  route to FINISH.
- If enough has been gathered to answer well, route to FINISH.

Also decide the ticker this turn concerns: if the user named or clearly
implied a company/ticker, use that (uppercase ticker symbol). If they're
continuing to discuss the same company as before, keep the current one. If
no company has ever been identified and none is named now, leave it null.

Respond with your routing decision."""


def _supervisor_node(state: ChatState) -> dict:
    llm = _build_llm()
    structured_llm = llm.with_structured_output(RouteDecision, method="function_calling")

    specialist_list = "\n".join(f"- {name}: {desc}" for name, desc in SPECIALIST_DESCRIPTIONS.items())
    findings_text = "\n".join(state["findings"]) if state["findings"] else "(none yet)"
    system = SystemMessage(
        content=SUPERVISOR_SYSTEM_PROMPT.format(
            specialist_list=specialist_list,
            ticker=state.get("ticker") or "not yet identified",
            findings=findings_text,
        )
    )

    decision = structured_llm.invoke([system, *state["messages"][-6:]])
    next_step = decision.get("next") or "FINISH"
    new_ticker = decision.get("ticker") or state.get("ticker")

    # Safety valve: never loop through specialists forever in one turn.
    if len(state["findings"]) >= 4 and next_step != "FINISH":
        next_step = "FINISH"

    return {"next": next_step, "ticker": new_ticker}


RESPOND_SYSTEM_PROMPT = """You are a direct, knowledgeable stock research assistant talking with the user.

If specialist findings are given below, treat them as your source of truth
for any numbers or claims — weave them into one clear, well-organized
reply, citing specific figures. If the user asked for a recommendation or
opinion, state a clear lean (never a wishy-washy "it depends") using the
strategist's rating if one was gathered, note the key risks, and remind
them the final decision is theirs. If no findings were gathered (a general
or conversational message), just respond naturally and honestly — don't
fabricate data you don't have."""


def _respond_node(state: ChatState) -> dict:
    llm = _build_llm()
    findings_text = "\n\n".join(state["findings"]) if state["findings"] else "(no specialist tools were needed this turn)"
    system = SystemMessage(content=RESPOND_SYSTEM_PROMPT)
    note = HumanMessage(content=f"Specialist findings gathered this turn:\n{findings_text}")
    response = llm.invoke([system, *state["messages"], note])
    return {"messages": [response]}


def _route_from_supervisor(state: ChatState) -> str:
    next_step = state.get("next", "FINISH")
    return next_step if next_step in SPECIALIST_DESCRIPTIONS else "respond"


def build_graph(model: str | None = None):
    """Compile the multi-agent LangGraph workflow. Returns a runnable graph."""
    workflow = StateGraph(ChatState)

    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node(
        "fundamentals_analyst",
        _make_specialist_node("fundamentals analyst", FUNDAMENTALS_TOOLS, SPECIALIST_DESCRIPTIONS["fundamentals_analyst"]),
    )
    workflow.add_node(
        "technicals_analyst",
        _make_specialist_node("technicals analyst", TECHNICALS_TOOLS, SPECIALIST_DESCRIPTIONS["technicals_analyst"]),
    )
    workflow.add_node(
        "news_analyst",
        _make_specialist_node("news analyst", NEWS_TOOLS, SPECIALIST_DESCRIPTIONS["news_analyst"]),
    )
    workflow.add_node(
        "crosscheck_analyst",
        _make_specialist_node("cross-check analyst", CROSSCHECK_TOOLS, SPECIALIST_DESCRIPTIONS["crosscheck_analyst"]),
    )
    workflow.add_node("strategist", _strategist_node)
    workflow.add_node("respond", _respond_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "fundamentals_analyst": "fundamentals_analyst",
            "technicals_analyst": "technicals_analyst",
            "news_analyst": "news_analyst",
            "crosscheck_analyst": "crosscheck_analyst",
            "strategist": "strategist",
            "respond": "respond",
        },
    )
    for specialist in ("fundamentals_analyst", "technicals_analyst", "news_analyst", "crosscheck_analyst", "strategist"):
        workflow.add_edge(specialist, "supervisor")
    workflow.add_edge("respond", END)

    return workflow.compile()


def run_turn(graph, history: list[BaseMessage], ticker: str | None, model: str | None = None) -> dict:
    """Run one conversational turn. `history` should already include the
    new HumanMessage. `findings` is deliberately reset to [] here — it is
    scoped to this single turn only, never persisted across turns."""
    state: ChatState = {"messages": history, "ticker": ticker, "findings": [], "next": "", "rating": None}
    final_state = graph.invoke(state)
    return {
        "messages": final_state["messages"],
        "ticker": final_state.get("ticker"),
        "rating": final_state.get("rating"),
        "reply": final_state["messages"][-1].content,
    }
