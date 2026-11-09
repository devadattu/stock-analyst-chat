"""
yfinance-backed data tools for the conversational stock analyst.

Each `_..._impl` function is the deterministic, directly-callable version
(used by signals.py and evals without going through the LLM); each `@tool`
wrapper is what gets bound to a specific specialist agent in graph.py.
Grouping specialists by toolset (FUNDAMENTALS_TOOLS, TECHNICALS_TOOLS,
NEWS_TOOLS) keeps each specialist's attention narrow and its LangSmith
trace easy to read.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _get_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol.strip().upper())


def _safe(info: dict, key: str) -> Any:
    v = info.get(key)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def _get_fundamentals_impl(ticker: str) -> dict[str, Any]:
    t = _get_ticker(ticker)
    info = t.info or {}
    g = lambda k: _safe(info, k)

    return {
        "ticker": ticker.upper(),
        "short_name": g("shortName"),
        "sector": g("sector"),
        "industry": g("industry"),
        "market_cap": g("marketCap"),
        # Valuation
        "trailing_pe": g("trailingPE"),
        "forward_pe": g("forwardPE"),
        "price_to_book": g("priceToBook"),
        "peg_ratio": g("trailingPegRatio") or g("pegRatio"),
        "book_value_per_share": g("bookValue"),
        # Profitability / returns
        "profit_margin": g("profitMargins"),
        "operating_margin": g("operatingMargins"),
        "return_on_equity": g("returnOnEquity"),
        "return_on_assets": g("returnOnAssets"),
        # Growth
        "revenue_growth_yoy": g("revenueGrowth"),
        "earnings_growth_yoy": g("earningsGrowth"),
        # Liquidity / leverage
        "current_ratio": g("currentRatio"),
        "quick_ratio": g("quickRatio"),
        "debt_to_equity": g("debtToEquity"),
        "free_cash_flow": g("freeCashflow"),
        # Other
        "dividend_yield": g("dividendYield"),
        "beta": g("beta"),
        "52wk_high": g("fiftyTwoWeekHigh"),
        "52wk_low": g("fiftyTwoWeekLow"),
    }


@tool
def get_fundamentals(ticker: str) -> dict[str, Any]:
    """Get fundamental metrics for a stock: valuation (P/E, P/B, PEG, book
    value per share), profitability and returns (margins, ROE, ROA), growth,
    liquidity (current/quick ratio), leverage, and free cash flow.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
    """
    return _get_fundamentals_impl(ticker)


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

def _get_price_history_impl(ticker: str, period: str = "6mo") -> dict[str, Any]:
    t = _get_ticker(ticker)
    hist = t.history(period=period, auto_adjust=True)
    if hist.empty:
        return {"ticker": ticker.upper(), "error": "no price data returned"}

    first_close = float(hist["Close"].iloc[0])
    last_close = float(hist["Close"].iloc[-1])
    pct_change = ((last_close - first_close) / first_close) * 100 if first_close else None

    return {
        "ticker": ticker.upper(),
        "period": period,
        "last_close": round(last_close, 2),
        "period_pct_change": round(pct_change, 2) if pct_change is not None else None,
        "period_high": round(float(hist["High"].max()), 2),
        "period_low": round(float(hist["Low"].min()), 2),
        "avg_daily_volume": int(hist["Volume"].mean()),
        "as_of": hist.index[-1].strftime("%Y-%m-%d"),
    }


@tool
def get_price_history(ticker: str, period: str = "6mo") -> dict[str, Any]:
    """Get price action for a stock ticker: latest close, % change over the
    period, period high/low, and average daily volume.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        period: One of "1mo", "3mo", "6mo", "1y", "2y", "5y". Defaults to "6mo".
    """
    return _get_price_history_impl(ticker, period)


# ---------------------------------------------------------------------------
# Technicals
# ---------------------------------------------------------------------------

def _rsi(closes: pd.Series, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _macd(closes: pd.Series) -> dict[str, float] | None:
    if len(closes) < 35:
        return None
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": round(float(macd_line.iloc[-1]), 3),
        "signal": round(float(signal_line.iloc[-1]), 3),
        "histogram": round(float(histogram.iloc[-1]), 3),
    }


def _bollinger(closes: pd.Series, window: int = 20, num_std: float = 2.0) -> dict[str, float] | None:
    if len(closes) < window:
        return None
    rolling_mean = closes.rolling(window=window).mean()
    rolling_std = closes.rolling(window=window).std()
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    last_price = float(closes.iloc[-1])
    band_width = float(upper.iloc[-1] - lower.iloc[-1])
    pct_b = (last_price - float(lower.iloc[-1])) / band_width if band_width else None
    return {
        "upper_band": round(float(upper.iloc[-1]), 2),
        "middle_band": round(float(rolling_mean.iloc[-1]), 2),
        "lower_band": round(float(lower.iloc[-1]), 2),
        "percent_b": round(pct_b, 2) if pct_b is not None else None,
    }


def _compute_technical_indicators_impl(ticker: str) -> dict[str, Any]:
    t = _get_ticker(ticker)
    hist = t.history(period="6mo", auto_adjust=True)
    if hist.empty or len(hist) < 20:
        return {"ticker": ticker.upper(), "error": "insufficient price history"}

    closes = hist["Close"]
    last_price = float(closes.iloc[-1])
    ma20 = float(closes.rolling(window=20).mean().iloc[-1])
    ma50 = float(closes.rolling(window=50).mean().iloc[-1]) if len(closes) >= 50 else None
    rsi14 = _rsi(closes, period=14)

    trend = "unknown"
    if ma50 is not None:
        if last_price > ma20 > ma50:
            trend = "uptrend"
        elif last_price < ma20 < ma50:
            trend = "downtrend"
        else:
            trend = "mixed"

    return {
        "ticker": ticker.upper(),
        "last_price": round(last_price, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2) if ma50 is not None else None,
        "rsi14": round(rsi14, 1) if rsi14 is not None else None,
        "trend": trend,
        "macd": _macd(closes),
        "bollinger_bands": _bollinger(closes),
    }


@tool
def compute_technical_indicators(ticker: str) -> dict[str, Any]:
    """Compute technical indicators for a stock: 20/50-day moving averages
    and trend, RSI(14), MACD (12/26/9), and 20-day Bollinger Bands.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
    """
    return _compute_technical_indicators_impl(ticker)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def _get_recent_news_impl(ticker: str, max_items: int = 8) -> dict[str, Any]:
    t = _get_ticker(ticker)
    try:
        raw_news = t.news or []
    except Exception as exc:
        logger.warning("get_recent_news failed for %s: %s", ticker, exc)
        return {"ticker": ticker.upper(), "headlines": [], "error": str(exc)}

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    headlines: list[dict[str, Any]] = []
    for item in raw_news:
        content = item.get("content", item)
        title = content.get("title")
        if not title:
            continue
        pub_date = content.get("pubDate") or content.get("providerPublishTime")
        published_iso = None
        if isinstance(pub_date, str):
            published_iso = pub_date
        elif isinstance(pub_date, (int, float)):
            published_iso = datetime.fromtimestamp(pub_date, tz=timezone.utc).isoformat()

        headlines.append(
            {
                "title": title,
                "publisher": (content.get("provider") or {}).get("displayName")
                if isinstance(content.get("provider"), dict)
                else content.get("publisher"),
                "published": published_iso,
                "link": content.get("canonicalUrl", {}).get("url")
                if isinstance(content.get("canonicalUrl"), dict)
                else content.get("link"),
            }
        )
        if len(headlines) >= max_items:
            break

    return {"ticker": ticker.upper(), "headlines": headlines, "cutoff_days": 30}


@tool
def get_recent_news(ticker: str, max_items: int = 8) -> dict[str, Any]:
    """Get recent news headlines for a stock ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        max_items: Maximum number of headlines to return (default 8).
    """
    return _get_recent_news_impl(ticker, max_items)


# ---------------------------------------------------------------------------
# Tool groupings — bound to specific specialist agents in graph.py
# ---------------------------------------------------------------------------

FUNDAMENTALS_TOOLS = [get_fundamentals]
TECHNICALS_TOOLS = [get_price_history, compute_technical_indicators]
NEWS_TOOLS = [get_recent_news]
ALL_TOOLS = [get_price_history, get_fundamentals, get_recent_news, compute_technical_indicators]
