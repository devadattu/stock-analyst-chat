"""
Deterministic, decisive recommendation scoring.

Like the earlier stock-analyst-agent project, the rating is computed by
plain, auditable rules over real numbers — not left for the LLM to invent —
so it's reproducible, traceable in LangSmith, and testable offline.

The key difference from that earlier project: this one is built to actually
take a position. A real equity analyst rarely issues a flat "Hold" — they
lean Buy or Sell with a stated conviction level, and let the investor make
the final call. So the "Hold" band here is intentionally narrow (only a
genuine near-exact toss-up lands there); everything else resolves to a
graded Buy/Sell call. The `strategist` agent (see graph.py) is instructed
to report this rating's reasoning in the same direct, opinionated tone —
while always naming the risks and making clear the decision belongs to the
user, not the tool.

IMPORTANT: this is a transparent heuristic for a learning project, not a
validated trading signal or licensed investment advice.
"""

from __future__ import annotations

from typing import Any, Literal

from chat_agent.tools import (
    _compute_technical_indicators_impl,
    _get_fundamentals_impl,
    _get_price_history_impl,
    _get_recent_news_impl,
)

Rating = Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]

WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.15,
    "valuation": 0.35,
    "news": 0.20,
}

_POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
    "upgrade", "upgraded", "record", "growth", "outperform", "strong",
    "gain", "gains", "jump", "jumps", "profit", "profits", "bullish",
    "raises", "raised", "buyback", "expansion",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "plunge", "plunges", "slump", "slumps", "downgrade",
    "downgraded", "weak", "decline", "declines", "loss", "losses", "lawsuit",
    "bearish", "cuts", "cut", "layoffs", "recall", "investigation", "fraud",
    "warning", "slowdown",
}


def _score_trend(technicals: dict[str, Any]) -> tuple[float, str]:
    trend = technicals.get("trend")
    if trend == "uptrend":
        return 1.0, "price above both MA20 and MA50 (uptrend)"
    if trend == "downtrend":
        return -1.0, "price below both MA20 and MA50 (downtrend)"
    if trend == "mixed":
        last_price = technicals.get("last_price")
        ma20 = technicals.get("ma20")
        if last_price is not None and ma20 is not None:
            return (0.3 if last_price >= ma20 else -0.3), "mixed trend, price near MA20"
        return 0.1, "mixed trend"
    return 0.0, "trend unknown (insufficient history)"


def _score_momentum(technicals: dict[str, Any]) -> tuple[float, str]:
    rsi = technicals.get("rsi14")
    macd = technicals.get("macd") or {}
    reasons = []
    score = 0.0

    if rsi is not None:
        if rsi >= 70:
            score += 0.3
            reasons.append(f"RSI {rsi} (overbought, momentum strong but stretched)")
        elif rsi <= 30:
            score -= 0.3
            reasons.append(f"RSI {rsi} (oversold, momentum weak)")
        else:
            score += ((rsi - 50) / 20) * 0.3
            reasons.append(f"RSI {rsi} (neutral range)")
    else:
        reasons.append("RSI unavailable")

    histogram = macd.get("histogram")
    if histogram is not None:
        score += 0.3 if histogram > 0 else -0.3
        reasons.append(f"MACD histogram {histogram} ({'bullish' if histogram > 0 else 'bearish'} crossover)")

    score = max(-1.0, min(1.0, score))
    return round(score, 2), "; ".join(reasons)


def _score_valuation(fundamentals: dict[str, Any]) -> tuple[float, str]:
    pe = fundamentals.get("trailing_pe")
    growth = fundamentals.get("earnings_growth_yoy")
    roe = fundamentals.get("return_on_equity")
    pb = fundamentals.get("price_to_book")
    reasons = []
    score = 0.0

    if pe is not None:
        if pe <= 0:
            score -= 0.3
            reasons.append(f"negative/undefined P/E ({pe})")
        elif pe < 15:
            score += 0.3
            reasons.append(f"low trailing P/E ({pe})")
        elif pe > 40:
            score -= 0.3
            reasons.append(f"high trailing P/E ({pe})")
        else:
            reasons.append(f"moderate trailing P/E ({pe})")
    else:
        reasons.append("P/E unavailable")

    if growth is not None:
        if growth > 0.15:
            score += 0.25
            reasons.append(f"strong earnings growth ({growth:.0%})")
        elif growth < 0:
            score -= 0.25
            reasons.append(f"declining earnings ({growth:.0%})")

    if roe is not None:
        if roe > 0.20:
            score += 0.25
            reasons.append(f"high return on equity ({roe:.0%}) — efficient use of shareholder capital")
        elif roe < 0.05:
            score -= 0.2
            reasons.append(f"weak return on equity ({roe:.0%})")

    if pb is not None:
        if pb < 1:
            score += 0.1
            reasons.append(f"trading below book value (P/B {pb})")
        elif pb > 15:
            score -= 0.1
            reasons.append(f"rich price-to-book ({pb})")

    score = max(-1.0, min(1.0, score))
    return round(score, 2), "; ".join(reasons)


def _score_news(news: dict[str, Any]) -> tuple[float, str]:
    headlines = news.get("headlines", [])
    if not headlines:
        return 0.0, "no recent headlines available"

    pos = neg = 0
    for h in headlines:
        title = (h.get("title") or "").lower()
        words = set(title.replace("-", " ").split())
        if words & _POSITIVE_WORDS:
            pos += 1
        if words & _NEGATIVE_WORDS:
            neg += 1

    net = pos - neg
    score = max(-1.0, min(1.0, net / max(len(headlines), 1) * 2))
    return round(score, 2), f"{pos} positive-leaning / {neg} negative-leaning of {len(headlines)} headlines (naive keyword heuristic)"


def _rating_from_score(score: float) -> Rating:
    if score >= 0.4:
        return "Strong Buy"
    if score >= 0.05:
        return "Buy"
    if score <= -0.4:
        return "Strong Sell"
    if score <= -0.05:
        return "Sell"
    return "Hold"


def compute_rating(ticker: str) -> dict[str, Any]:
    """Fetch fresh data for `ticker` and deterministically derive a decisive
    Buy/Hold/Sell-style rating with a full, auditable reasoning breakdown."""
    price = _get_price_history_impl(ticker)
    fundamentals = _get_fundamentals_impl(ticker)
    technicals = _compute_technical_indicators_impl(ticker)
    news = _get_recent_news_impl(ticker)

    trend_score, trend_reason = _score_trend(technicals)
    momentum_score, momentum_reason = _score_momentum(technicals)
    valuation_score, valuation_reason = _score_valuation(fundamentals)
    news_score, news_reason = _score_news(news)

    breakdown = {
        "trend": {"score": trend_score, "weight": WEIGHTS["trend"], "reason": trend_reason},
        "momentum": {"score": momentum_score, "weight": WEIGHTS["momentum"], "reason": momentum_reason},
        "valuation": {"score": valuation_score, "weight": WEIGHTS["valuation"], "reason": valuation_reason},
        "news": {"score": news_score, "weight": WEIGHTS["news"], "reason": news_reason},
    }

    overall_score = round(sum(breakdown[k]["score"] * WEIGHTS[k] for k in WEIGHTS), 3)
    rating = _rating_from_score(overall_score)

    agreement = sum(1 for k in WEIGHTS if _rating_from_score(breakdown[k]["score"]) == rating)
    magnitude = min(abs(overall_score) / 0.6, 1.0)
    confidence = round((agreement / len(WEIGHTS)) * 0.6 + magnitude * 0.4, 2)

    return {
        "ticker": ticker.upper(),
        "rating": rating,
        "overall_score": overall_score,
        "confidence": confidence,
        "breakdown": breakdown,
        "disclaimer": (
            "This rating is a transparent, rule-based heuristic over public price, "
            "fundamentals, technicals, and headline data — it is educational analysis, "
            "not licensed financial advice. The reasoning is shown so you can judge it "
            "yourself; the decision to act is yours, not the tool's."
        ),
        "inputs": {
            "price_history": price,
            "fundamentals": fundamentals,
            "technicals": technicals,
            "news": news,
        },
    }
