"""
Creates (or updates) the LangSmith dataset the conversational eval suite
runs against.

Unlike the first project's one-ticker-per-example dataset, each example
here is a single-turn QUESTION about a ticker -- a mix of plain factual
lookups ("what is AAPL's book value and ROE") and explicit recommendation
asks ("should I buy TSLA") -- because what's actually new and worth
evaluating in this project is the conversational routing and the
decisiveness of the recommendation, not just report structure.

Run directly to (re)create the dataset:
    python -m evals.dataset
"""

from __future__ import annotations

from langsmith import Client

DATASET_NAME = "stock-analyst-chat-questions"

# (ticker, question, expects_recommendation) -- a deliberately varied mix:
# mega-cap tech, a different sector, a higher-volatility name; plain
# fundamentals/technicals/news questions alongside explicit "should I
# buy/sell" asks, so the eval suite exercises both specialist routing and
# the strategist's decisiveness.
EXAMPLES = [
    {"ticker": "AAPL", "question": "What is AAPL's book value per share and ROE?", "expects_recommendation": False},
    {"ticker": "AAPL", "question": "Should I buy AAPL right now?", "expects_recommendation": True},
    {"ticker": "MSFT", "question": "What's the current trend and RSI for MSFT?", "expects_recommendation": False},
    {"ticker": "JNJ", "question": "Is JNJ a buy, hold, or sell?", "expects_recommendation": True},
    {"ticker": "TSLA", "question": "What's the latest news on TSLA and has the price moved much?", "expects_recommendation": False},
    {"ticker": "TSLA", "question": "Give me your honest recommendation on TSLA.", "expects_recommendation": True},
    {"ticker": "NVDA", "question": "Can you cross-check NVDA's price against Alpaca's data?", "expects_recommendation": False},
    {"ticker": "NVDA", "question": "What's NVDA's P/E ratio and free cash flow?", "expects_recommendation": False},
]


def ensure_dataset(client: Client | None = None) -> str:
    """Create the dataset + examples if they don't already exist. Idempotent-ish:
    safe to re-run, but will not duplicate a dataset with the same name."""
    client = client or Client()

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        dataset_id = existing[0].id
        print(f"Dataset '{DATASET_NAME}' already exists ({dataset_id}); leaving as-is.")
        return dataset_id

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Single-turn questions (factual + recommendation-seeking) used to evaluate "
        "the conversational multi-agent stock assistant's routing, groundedness, and decisiveness.",
    )
    client.create_examples(
        inputs=[{"ticker": e["ticker"], "question": e["question"], "expects_recommendation": e["expects_recommendation"]} for e in EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"Created dataset '{DATASET_NAME}' ({dataset.id}) with {len(EXAMPLES)} examples.")
    return dataset.id


if __name__ == "__main__":
    ensure_dataset()
