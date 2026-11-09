"""
Pure unit tests for the deterministic rating logic in chat_agent/signals.py.

These never hit the network or call an LLM -- they feed hand-built dicts
(shaped like tools.py's real return values) into the scoring functions, so
they run fast and offline, e.g. in CI.

The thresholds under test are intentionally narrow around "Hold" -- this
project's whole point (per the user's explicit "don't be neutral" ask) is
that the rating leans Buy/Sell except in a genuine near-exact toss-up. See
the module docstring in chat_agent/signals.py for the rationale.

Run with: pytest tests/
"""

from chat_agent.signals import (
    WEIGHTS,
    _rating_from_score,
    _score_momentum,
    _score_news,
    _score_trend,
    _score_valuation,
)


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_score_trend_uptrend_is_positive():
    score, reason = _score_trend({"trend": "uptrend", "last_price": 110, "ma20": 105, "ma50": 100})
    assert score == 1.0
    assert "uptrend" in reason


def test_score_trend_downtrend_is_negative():
    score, _ = _score_trend({"trend": "downtrend", "last_price": 90, "ma20": 95, "ma50": 100})
    assert score == -1.0


def test_score_trend_mixed_above_ma20_is_mildly_positive():
    score, reason = _score_trend({"trend": "mixed", "last_price": 110, "ma20": 105})
    assert score == 0.3
    assert "mixed" in reason


def test_score_trend_mixed_below_ma20_is_mildly_negative():
    score, _ = _score_trend({"trend": "mixed", "last_price": 100, "ma20": 105})
    assert score == -0.3


def test_score_trend_unknown_is_neutral():
    score, _ = _score_trend({"trend": "unknown"})
    assert score == 0.0


def test_score_momentum_overbought_rsi_is_positive():
    score, reason = _score_momentum({"rsi14": 85, "macd": {}})
    assert score > 0
    assert "overbought" in reason


def test_score_momentum_oversold_rsi_is_negative():
    score, reason = _score_momentum({"rsi14": 15, "macd": {}})
    assert score < 0
    assert "oversold" in reason


def test_score_momentum_bullish_macd_histogram_adds_positive():
    score, reason = _score_momentum({"rsi14": None, "macd": {"histogram": 1.5}})
    assert score > 0
    assert "bullish" in reason


def test_score_momentum_bearish_macd_histogram_adds_negative():
    score, reason = _score_momentum({"rsi14": None, "macd": {"histogram": -1.5}})
    assert score < 0
    assert "bearish" in reason


def test_score_momentum_missing_everything_is_neutral():
    score, _ = _score_momentum({"rsi14": None, "macd": {}})
    assert score == 0.0


def test_score_valuation_low_pe_high_growth_high_roe_is_strongly_bullish():
    score, reason = _score_valuation(
        {"trailing_pe": 10, "earnings_growth_yoy": 0.25, "return_on_equity": 0.30, "price_to_book": 2}
    )
    assert score > 0
    assert "low trailing P/E" in reason
    assert "strong earnings growth" in reason
    assert "high return on equity" in reason


def test_score_valuation_negative_pe_declining_earnings_weak_roe_is_bearish():
    score, _ = _score_valuation(
        {"trailing_pe": -5, "earnings_growth_yoy": -0.1, "return_on_equity": 0.02, "price_to_book": 20}
    )
    assert score < 0


def test_score_valuation_below_book_value_adds_a_small_bonus():
    score, reason = _score_valuation(
        {"trailing_pe": None, "earnings_growth_yoy": None, "return_on_equity": None, "price_to_book": 0.8}
    )
    assert score > 0
    assert "below book value" in reason


def test_score_valuation_missing_data_is_neutral():
    score, reason = _score_valuation(
        {"trailing_pe": None, "earnings_growth_yoy": None, "return_on_equity": None, "price_to_book": None}
    )
    assert score == 0.0
    assert "unavailable" in reason


def test_score_news_more_positive_than_negative():
    news = {
        "headlines": [
            {"title": "Company beats earnings expectations, shares surge"},
            {"title": "Analyst upgrade sends stock higher"},
            {"title": "Quiet trading day, no major news"},
        ]
    }
    score, reason = _score_news(news)
    assert score > 0
    assert "positive-leaning" in reason


def test_score_news_more_negative_than_positive():
    news = {
        "headlines": [
            {"title": "Company misses earnings, shares plunge"},
            {"title": "Downgrade follows weak guidance"},
        ]
    }
    score, _ = _score_news(news)
    assert score < 0


def test_score_news_no_headlines_is_neutral():
    score, reason = _score_news({"headlines": []})
    assert score == 0.0
    assert "no recent headlines" in reason


def test_rating_from_score_thresholds_are_narrow_around_hold():
    # The whole point of this project's scoring: "Hold" is a thin band,
    # not the default outcome for anything middling.
    assert _rating_from_score(0.5) == "Strong Buy"
    assert _rating_from_score(0.4) == "Strong Buy"
    assert _rating_from_score(0.2) == "Buy"
    assert _rating_from_score(0.05) == "Buy"
    assert _rating_from_score(0.0) == "Hold"
    assert _rating_from_score(-0.04) == "Hold"
    assert _rating_from_score(-0.05) == "Sell"
    assert _rating_from_score(-0.2) == "Sell"
    assert _rating_from_score(-0.4) == "Strong Sell"
    assert _rating_from_score(-0.9) == "Strong Sell"
