# Stock Analyst Chat

A conversational, **multi-agent** companion to
[`stock-analyst-agent`](../stock-analyst-agent): instead of one agent
producing a single static report, this project runs a small team of
specialist agents behind a chat loop, so you can ask free-form,
follow-up-friendly questions about a stock -- "what's AAPL's book value and
ROE", "what's the trend on NVDA look like", "should I buy TSLA" -- and keep
the conversation going.

**This project produces educational analysis, not financial advice.**
Recommendations are a disclosed, rule-based heuristic over public data --
not a validated trading signal -- and every recommendation repeats that
disclaimer. Nothing here should be the basis for an actual investment
decision. See [Legal note](#legal-note) below.

## Why multi-agent, and why decisive

Two things this project is deliberately built around:

1. **Multiple specialist agents, not one generalist.** A `supervisor`
   agent reads the conversation and routes each turn to whichever
   specialist(s) can actually answer it -- `fundamentals_analyst`,
   `technicals_analyst`, `news_analyst`, `crosscheck_analyst` (verifies a
   price against [Alpaca Markets'](https://alpaca.markets) independent
   data), and `strategist` (turns a deterministic rating into a
   recommendation). Each specialist is bound to only its own narrow
   toolset and runs its own small tool-calling loop; a shared `findings`
   scratchpad collects what each one reports back, and a final `respond`
   step composes the one reply you actually see. See
   `chat_agent/graph.py`.

2. **Recommendations take an actual position.** A real analyst rarely
   says "it depends" -- they lean Buy or Sell with a stated conviction and
   let the investor decide. So `chat_agent/signals.py` computes a
   deterministic score (same rule-based, auditable approach as the first
   project) with a deliberately **narrow Hold band** -- only a genuine
   near-exact toss-up lands on Hold; everything else resolves to a graded
   **Strong Buy / Buy / Hold / Sell / Strong Sell** call. The `strategist`
   agent is instructed to report that rating in a direct, opinionated
   tone, name the real risks, and always close by making clear the
   decision is yours, not the tool's.

The rating itself is still computed by plain Python rules, not invented by
the LLM -- the LLM's job is to explain a given number, never to make one
up -- which keeps it reproducible, auditable in the LangSmith trace, and
testable offline (`tests/test_signals.py`).

## What's in here

```
chat_agent/
  tools.py        yfinance-backed tools: price history, fundamentals (incl. book
                   value/share, ROE, ROA, current/quick ratio, FCF), news,
                   technicals (incl. MACD, Bollinger Bands)
  alpaca_tools.py  cross_check_price tool: compares yfinance's last close against
                   Alpaca Markets' latest trade price, flags disagreement >3%
  signals.py       deterministic Strong Buy/Buy/Hold/Sell/Strong Sell rating,
                   derived from the tools' own output, with a narrow Hold band
  graph.py         the multi-agent LangGraph workflow: supervisor <-> specialists
                   <-> strategist, then a respond node
chat.py            terminal chat loop entrypoint (multi-turn, remembers the
                   ticker and conversation across turns)
evals/
  dataset.py       LangSmith dataset of (ticker, question) pairs -- factual +
                   recommendation-seeking questions
  evaluators.py    3 evaluators: ticker identification, recommendation
                   decisiveness, reply groundedness
  run_eval.py      runs the eval suite and logs an experiment to LangSmith
tests/
  test_signals.py  pure unit tests for the scoring logic (no network/API calls)
```

## Setup

1. **Python 3.11+** recommended.

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **API keys.** Copy `.env.example` to `.env` and fill in:

   | Variable | Where to get it | Required for |
   |---|---|---|
   | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) | running the agent at all |
   | `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com/) → Settings → API Keys | tracing + evals |
   | `LANGSMITH_TRACING=true` | — | turns tracing on |
   | `LANGSMITH_AGENT_ID` | any name (defaults to `stock-analyst-chat`) | groups traces by agent in the LangSmith UI |
   | `LANGSMITH_AGENT_ENVIRONMENT` | e.g. `local` | the environment dropdown in the LangSmith UI |
   | `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | free account at [alpaca.markets](https://alpaca.markets) | the cross-check specialist |

   Alpaca's free paper-trading signup gives you a key with market-data
   access -- no funding needed, since this project only reads prices.

   LangSmith's current workspace model addresses traces by an **agent id +
   environment** (not an older "project name"): in the UI, use the agent
   switcher and environment dropdown in the top bar, then **Tracing** or
   **Datasets & Experiments** in the left sidebar.

## Running it

```bash
python chat.py
```

```
Stock Research Assistant (multi-agent, conversational)
--------------------------------------------------------
Ask about any stock: fundamentals, technicals, recent news, a cross-check
against Alpaca, or a straight-up Buy/Hold/Sell call. I'll stay decisive
when you ask for a recommendation, name the risks, and leave the final
decision to you.

Type 'exit' (or 'quit') to leave.

you> what is AAPL's book value per share and ROE?

assistant> AAPL's book value per share is $X.XX and ROE is XX% ...

you> should I buy it?

assistant> Lean: Buy (confidence 0.XX, score 0.XX) ...
Biggest risks: ...
This is analysis to inform your call, not a directive -- the decision is yours.

you> exit
bye.
```

Each turn is routed by the `supervisor` to whichever specialist(s) are
relevant -- a plain factual question like the first one above only
triggers `fundamentals_analyst`; "should I buy it?" additionally triggers
`strategist`, which computes the deterministic rating and turns it into a
direct recommendation. The agent remembers the ticker across turns, so
"should I buy it?" correctly resolves to AAPL without repeating the name.

## Observability: reading the LangSmith trace

With `LANGSMITH_TRACING=true`, every turn shows up under your
`LANGSMITH_AGENT_ID` / `LANGSMITH_AGENT_ENVIRONMENT` in LangSmith. Open a
trace and you'll see the graph's nodes as nested spans:

- **supervisor** (one span per routing decision) -- the structured-output
  LLM call deciding which specialist to call next, or to finish
- **fundamentals_analyst / technicals_analyst / news_analyst /
  crosscheck_analyst** -- each specialist's own internal tool-calling
  loop, with the exact tool inputs/outputs it worked from
- **strategist** -- expand it to see the full deterministic rating
  breakdown (per-category scores, weights, and reasons) that was handed
  to the LLM, so you can confirm it reported the given numbers rather
  than inventing new ones
- **respond** -- the final LLM call that composes the one reply you see,
  with all of that turn's findings as its input

This is the same "trace any sentence back to the exact tool call or rule
that produced it" principle as the first project, extended across a team
of agents instead of one.

## Evals

The eval suite (`evals/dataset.py`) runs single-turn questions -- a mix of
plain factual lookups and explicit "should I buy/sell" asks -- through
`evals/evaluators.py`'s 3 deterministic, non-LLM evaluators:

- **ticker_correctly_identified** -- did the supervisor route to the
  ticker the question was actually about?
- **recommendation_is_decisive** -- for recommendation-seeking questions,
  does the reply state an actual rating label, avoid generic dodge
  phrases, and still make clear the decision belongs to the user?
- **reply_is_grounded_in_findings** -- does every $/% figure in the reply
  trace back to a number that actually came from the rating's underlying
  tool data? (a lightweight hallucination tripwire)

Run it with:

```bash
python -m evals.run_eval
```

This logs an experiment against the `stock-analyst-chat-questions`
dataset in LangSmith -- open the dataset's **Experiments** tab to see a
score per evaluator per question, and to compare experiments after you
change a prompt, the model, the routing logic, or the scoring rules.

There are also plain unit tests for the scoring rules themselves, with no
network or API calls:

```bash
pip install pytest
pytest tests/
```

## Possible next steps

- Persist conversation history across runs (currently in-memory for the
  life of one `chat.py` process)
- A peer-comparison specialist, to compare one ticker against others
- An LLM-as-judge evaluator for recommendation *quality*, not just
  structure/decisiveness
- A small web UI on top of the same graph

## Legal note

This project, and anything it outputs, is for learning multi-agent
architecture and observability. It is not a licensed financial advisor,
has not been validated against real trading outcomes, and its
recommendations are a toy heuristic over free/paper-trading data. Don't
make investment decisions based on it.
