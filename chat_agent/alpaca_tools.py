"""
Alpaca Markets cross-check tool.

Independently fetches a ticker's latest quote/trade from Alpaca's Market
Data API and compares it against yfinance's last close, so the agent can
flag disagreement between two independent providers rather than quietly
trusting a single source. Requires a free Alpaca account (paper trading
account is enough — ALPACA_API_KEY / ALPACA_SECRET_KEY, no funding needed).
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool

from chat_agent.tools import _get_price_history_impl


def _get_alpaca_client():
    from alpaca.data.historical import StockHistoricalDataClient

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set in the environment")
    return StockHistoricalDataClient(api_key, secret_key)


def _cross_check_price_impl(ticker: str) -> dict[str, Any]:
    ticker = ticker.strip().upper()
    yfinance_data = _get_price_history_impl(ticker, period="5d")
    yfinance_price = yfinance_data.get("last_close")

    try:
        from alpaca.data.requests import StockLatestTradeRequest

        client = _get_alpaca_client()
        request = StockLatestTradeRequest(symbol_or_symbols=ticker)
        trades = client.get_stock_latest_trade(request)
        alpaca_trade = trades.get(ticker)
        alpaca_price = round(float(alpaca_trade.price), 2) if alpaca_trade else None
        alpaca_timestamp = str(alpaca_trade.timestamp) if alpaca_trade else None
    except Exception as exc:
        return {
            "ticker": ticker,
            "yfinance_last_close": yfinance_price,
            "alpaca_last_trade_price": None,
            "error": f"Alpaca lookup failed: {exc}",
            "agreement": "unknown",
        }

    agreement = "unknown"
    pct_diff = None
    if yfinance_price is not None and alpaca_price is not None:
        pct_diff = round(abs(yfinance_price - alpaca_price) / alpaca_price * 100, 3)
        # yfinance's "last close" and Alpaca's "latest trade" aren't the same
        # instant (prior session close vs. most recent print, possibly
        # after-hours), so a modest gap is expected — only flag a real
        # disagreement past a few percent.
        agreement = "consistent" if pct_diff <= 3.0 else "DISAGREEMENT"

    return {
        "ticker": ticker,
        "yfinance_last_close": yfinance_price,
        "alpaca_last_trade_price": alpaca_price,
        "alpaca_trade_timestamp": alpaca_timestamp,
        "pct_difference": pct_diff,
        "agreement": agreement,
        "note": "yfinance reports a recent close; Alpaca reports the latest trade print — "
        "these are not always the exact same instant, so small gaps are normal.",
    }


@tool
def cross_check_price(ticker: str) -> dict[str, Any]:
    """Independently verify a stock's current price by comparing yfinance's
    data against Alpaca Markets' latest trade data. Use this when the user
    asks to double-check or verify a price/data point, or before relying on
    a number for an important conclusion like a buy/sell recommendation.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
    """
    return _cross_check_price_impl(ticker)


CROSSCHECK_TOOLS = [cross_check_price]
