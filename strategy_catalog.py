from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    key: str
    label: str
    description: str
    recommended: bool = False


STRATEGIES: dict[str, StrategyDefinition] = {
    "forecast_trend": StrategyDefinition(
        key="forecast_trend",
        label="Forecast Trend (Recommended)",
        description="Follows Kronos direction only when the current price structure is still constructive.",
        recommended=True,
    ),
    "mean_reversion": StrategyDefinition(
        key="mean_reversion",
        label="Mean Reversion",
        description="Buys forecasted rebounds after short-term weakness and exits after recovery fades.",
    ),
    "breakout_confirmation": StrategyDefinition(
        key="breakout_confirmation",
        label="Breakout Confirmation",
        description="Only enters when Kronos is bullish and price is already pressing a recent breakout area.",
    ),
}


def strategy_options() -> list[dict[str, str | bool]]:
    return [
        {
            "key": item.key,
            "label": item.label,
            "description": item.description,
            "recommended": item.recommended,
        }
        for item in STRATEGIES.values()
    ]


def get_strategy_definition(strategy_key: str) -> StrategyDefinition:
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_key}")
    return STRATEGIES[strategy_key]


def _trend_inputs(history_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    close = history_df["close"].astype(float)
    last_close = float(close.iloc[-1])
    recent_5d_return = float(last_close / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
    recent_20d_high = float(close.tail(20).max())
    sma20 = float(close.tail(min(20, len(close))).mean())
    sma60 = float(close.tail(min(60, len(close))).mean())
    predicted_close = pred_df["close"].astype(float)
    predicted_return = float(predicted_close.iloc[-1] / last_close - 1.0)
    forecast_slope = float(predicted_close.iloc[-1] - predicted_close.iloc[0])
    forecast_up_ratio = float((predicted_close.diff() > 0).mean())
    return {
        "last_close": last_close,
        "recent_5d_return": recent_5d_return,
        "recent_20d_high": recent_20d_high,
        "sma20": sma20,
        "sma60": sma60,
        "predicted_return": predicted_return,
        "forecast_slope": forecast_slope,
        "forecast_up_ratio": forecast_up_ratio,
        "predicted_max_close": float(predicted_close.max()),
        "predicted_end_close": float(predicted_close.iloc[-1]),
    }


def evaluate_strategy(
    strategy_key: str,
    history_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    signal_threshold: float,
) -> dict:
    inputs = _trend_inputs(history_df, pred_df)
    last_close = inputs["last_close"]
    buy = False
    sell = False
    rationale: list[str] = []

    if strategy_key == "forecast_trend":
        buy = (
            inputs["predicted_return"] > signal_threshold
            and inputs["forecast_slope"] > 0
            and last_close >= inputs["sma20"]
        )
        sell = inputs["predicted_return"] < -signal_threshold or last_close < inputs["sma20"] * 0.985
        if inputs["predicted_return"] > signal_threshold:
            rationale.append("Kronos expects positive forward return.")
        if inputs["forecast_slope"] > 0:
            rationale.append("Forecast path slopes upward.")
        if last_close >= inputs["sma20"]:
            rationale.append("Price is above the 20-day average.")
    elif strategy_key == "mean_reversion":
        buy = inputs["predicted_return"] > signal_threshold and inputs["recent_5d_return"] < 0
        sell = inputs["predicted_return"] < 0 or inputs["recent_5d_return"] > 0.03
        if inputs["recent_5d_return"] < 0:
            rationale.append("Recent weakness creates a rebound setup.")
        if inputs["predicted_return"] > signal_threshold:
            rationale.append("Kronos still predicts recovery.")
    elif strategy_key == "breakout_confirmation":
        buy = (
            inputs["predicted_return"] > signal_threshold
            and last_close >= inputs["recent_20d_high"] * 0.995
            and inputs["predicted_max_close"] > inputs["recent_20d_high"]
        )
        sell = inputs["predicted_return"] < 0 or last_close < inputs["sma20"]
        if last_close >= inputs["recent_20d_high"] * 0.995:
            rationale.append("Price is close to a 20-day breakout level.")
        if inputs["predicted_max_close"] > inputs["recent_20d_high"]:
            rationale.append("Kronos expects the breakout to extend.")
    else:
        raise ValueError(f"Unknown strategy: {strategy_key}")

    if buy:
        action = "BUY"
        target_position = 1.0
    elif sell:
        action = "SELL"
        target_position = 0.0
    else:
        action = "HOLD"
        target_position = 1.0 if inputs["predicted_return"] > signal_threshold else 0.0
        rationale.append("No strong setup; keep current exposure unless already aligned.")

    return {
        "strategy": strategy_key,
        "action": action,
        "target_position": target_position,
        "signal_threshold": signal_threshold,
        "predicted_return_pct": round(inputs["predicted_return"] * 100, 2),
        "recent_5d_return_pct": round(inputs["recent_5d_return"] * 100, 2),
        "forecast_up_ratio_pct": round(inputs["forecast_up_ratio"] * 100, 2),
        "last_close": round(last_close, 4),
        "sma20": round(inputs["sma20"], 4),
        "sma60": round(inputs["sma60"], 4),
        "recent_20d_high": round(inputs["recent_20d_high"], 4),
        "predicted_end_close": round(inputs["predicted_end_close"], 4),
        "rationale": rationale,
    }
