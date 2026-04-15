from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from config import ToolkitConfig
from data_sources import load_history, normalize_symbol
from kronos_engine import load_predictor, predict_future
from strategy_catalog import evaluate_strategy, get_strategy_definition


def summarize_prediction(
    symbol: str,
    label: str,
    history_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    source_used: str,
) -> dict:
    pred_close = pred_df["close"].astype(float)
    last_close = float(history_df["close"].iloc[-1])
    horizon = len(pred_df)
    return {
        "symbol": symbol,
        "label": label or symbol,
        "source": source_used,
        "last_trade_date": history_df["timestamps"].iloc[-1].strftime("%Y-%m-%d"),
        "last_close": round(last_close, 4),
        "pred_5d_close": round(float(pred_close.iloc[min(4, horizon - 1)]), 4),
        "pred_10d_close": round(float(pred_close.iloc[min(9, horizon - 1)]), 4),
        f"pred_{horizon}d_close": round(float(pred_close.iloc[-1]), 4),
        "pred_5d_return_pct": round((float(pred_close.iloc[min(4, horizon - 1)]) / last_close - 1) * 100, 2),
        "pred_10d_return_pct": round((float(pred_close.iloc[min(9, horizon - 1)]) / last_close - 1) * 100, 2),
        f"pred_{horizon}d_return_pct": round((float(pred_close.iloc[-1]) / last_close - 1) * 100, 2),
        "pred_max_close": round(float(pred_close.max()), 4),
        "pred_min_close": round(float(pred_close.min()), 4),
    }


def save_prediction_plot(
    symbol: str,
    label: str,
    history_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_history = history_df.tail(60).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_history["timestamps"], plot_history["close"], label="Historical Close", linewidth=1.8)
    ax.plot(pred_df["timestamps"], pred_df["close"], label="Kronos Forecast", linewidth=1.8)
    ax.axvline(plot_history["timestamps"].iloc[-1], color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{label or symbol} ({symbol}) Forecast")
    ax.set_ylabel("Close")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def build_signals(
    predictor,
    price_df: pd.DataFrame,
    lookback: int,
    pred_len: int,
    rebalance_step: int,
    signal_threshold: float,
    strategy_key: str,
) -> tuple[pd.DataFrame, pd.Series]:
    features = ["open", "high", "low", "close", "volume", "amount"]
    signal_rows: list[dict] = []
    positions = pd.Series(0.0, index=pd.DatetimeIndex(price_df["timestamps"]))

    for context_end in range(lookback, len(price_df) - pred_len, rebalance_step):
        context_df = price_df.iloc[context_end - lookback : context_end].copy()
        future_df = price_df.iloc[context_end : context_end + pred_len].copy()

        with torch.no_grad():
            pred_df = predictor.predict(
                df=context_df[features].reset_index(drop=True),
                x_timestamp=context_df["timestamps"].reset_index(drop=True),
                y_timestamp=future_df["timestamps"].reset_index(drop=True),
                pred_len=pred_len,
                T=1.0,
                top_k=1,
                top_p=1.0,
                sample_count=1,
                verbose=False,
            )

        last_close = float(context_df["close"].iloc[-1])
        predicted_close = float(pred_df["close"].iloc[-1])
        actual_close = float(future_df["close"].iloc[-1])
        signal = predicted_close / last_close - 1.0
        recommendation = evaluate_strategy(
            strategy_key=strategy_key,
            history_df=context_df,
            pred_df=pred_df.reset_index().rename(columns={"index": "timestamps"}),
            signal_threshold=signal_threshold,
        )
        position = recommendation["target_position"]

        segment_end = min(context_end + rebalance_step, len(price_df))
        segment_dates = pd.DatetimeIndex(price_df["timestamps"].iloc[context_end:segment_end])
        positions.loc[segment_dates] = position

        signal_rows.append(
            {
                "signal_date": future_df["timestamps"].iloc[0],
                "last_close": last_close,
                "predicted_horizon_close": predicted_close,
                "actual_horizon_close": actual_close,
                "predicted_return": signal,
                "actual_return": actual_close / last_close - 1.0,
                "position": position,
                "action": recommendation["action"],
            }
        )

    return pd.DataFrame(signal_rows), positions


def summarize_backtest(daily_df: pd.DataFrame, trades_df: pd.DataFrame, source_used: str) -> dict:
    total_return = float(daily_df["strategy_equity"].iloc[-1] - 1.0)
    benchmark_return = float(daily_df["benchmark_equity"].iloc[-1] - 1.0)
    max_drawdown = float((daily_df["strategy_equity"] / daily_df["strategy_equity"].cummax() - 1.0).min())
    hit_rate = float((trades_df["actual_return"] > 0).mean()) if not trades_df.empty else 0.0
    active_rate = float(trades_df["position"].mean()) if not trades_df.empty else 0.0
    return {
        "source": source_used,
        "trade_count": int(len(trades_df)),
        "strategy_return_pct": round(total_return * 100, 2),
        "benchmark_return_pct": round(benchmark_return * 100, 2),
        "excess_return_pct": round((total_return - benchmark_return) * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "hit_rate_pct": round(hit_rate * 100, 2),
        "time_in_market_pct": round(active_rate * 100, 2),
    }


def run_prediction_analysis(
    config: ToolkitConfig,
    symbol: str,
    label: str,
    start: str,
    end: str,
    lookback: int,
    pred_len: int,
    source: str,
    device: str,
    strategy_key: str,
    signal_threshold: float,
) -> dict:
    normalized_symbol = normalize_symbol(symbol)
    history_df, source_used = load_history(config, normalized_symbol, start, end, source=source)
    if len(history_df) < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {len(history_df)}.")

    context_df = history_df.tail(lookback).copy()
    pred_df = predict_future(config, context_df, pred_len=pred_len, device=device)

    prefix = config.output_dir / normalized_symbol
    prediction_csv = prefix.with_name(f"{normalized_symbol}_prediction.csv")
    history_csv = prefix.with_name(f"{normalized_symbol}_history.csv")
    summary_json = prefix.with_name(f"{normalized_symbol}_summary.json")
    forecast_png = prefix.with_name(f"{normalized_symbol}_forecast.png")

    pred_df.to_csv(prediction_csv, index=False)
    history_df.to_csv(history_csv, index=False)

    summary = summarize_prediction(normalized_symbol, label, context_df, pred_df, source_used)
    recommendation = evaluate_strategy(
        strategy_key=strategy_key,
        history_df=context_df,
        pred_df=pred_df,
        signal_threshold=signal_threshold,
    )
    summary["strategy"] = get_strategy_definition(strategy_key).label
    summary["recommended_action"] = recommendation["action"]
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    save_prediction_plot(normalized_symbol, label, context_df, pred_df, forecast_png)
    return {
        "summary": summary,
        "recommendation": recommendation,
        "prediction_path": prediction_csv,
        "history_path": history_csv,
        "summary_path": summary_json,
        "forecast_path": forecast_png,
        "prediction_df": pred_df,
        "history_df": history_df,
        "source": source_used,
        "symbol": normalized_symbol,
    }


def run_backtest_analysis(
    config: ToolkitConfig,
    symbol: str,
    start: str,
    end: str,
    lookback: int,
    pred_len: int,
    rebalance_step: int,
    signal_threshold: float,
    source: str,
    device: str,
    strategy_key: str,
) -> dict:
    normalized_symbol = normalize_symbol(symbol)
    price_df, source_used = load_history(config, normalized_symbol, start, end, source=source)
    if len(price_df) < lookback + pred_len + rebalance_step:
        raise ValueError("Not enough rows for the requested backtest window.")

    predictor = load_predictor(config, device=device)
    trades_df, positions = build_signals(
        predictor=predictor,
        price_df=price_df,
        lookback=lookback,
        pred_len=pred_len,
        rebalance_step=rebalance_step,
        signal_threshold=signal_threshold,
        strategy_key=strategy_key,
    )

    daily_df = price_df[["timestamps", "close"]].copy()
    daily_df["asset_return"] = daily_df["close"].pct_change().fillna(0.0)
    daily_df["position"] = positions.reindex(pd.DatetimeIndex(daily_df["timestamps"])).fillna(0.0).to_numpy()
    daily_df["position_for_return"] = daily_df["position"].shift(1).fillna(0.0)
    turnover = daily_df["position"].diff().abs().fillna(daily_df["position"].abs())
    daily_df["cost"] = turnover * config.transaction_cost_rate
    daily_df["strategy_return"] = daily_df["position_for_return"] * daily_df["asset_return"] - daily_df["cost"]
    daily_df["strategy_equity"] = (1.0 + daily_df["strategy_return"]).cumprod()
    daily_df["benchmark_equity"] = (1.0 + daily_df["asset_return"]).cumprod()

    base = config.output_dir / normalized_symbol
    trades_path = Path(str(base) + "_backtest_trades.csv")
    daily_path = Path(str(base) + "_backtest_daily.csv")
    summary_path = Path(str(base) + "_backtest_summary.json")

    trades_df.to_csv(trades_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    summary = summarize_backtest(daily_df, trades_df, source_used)
    summary["strategy"] = get_strategy_definition(strategy_key).label
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return {
        "summary": summary,
        "trades_path": trades_path,
        "daily_path": daily_path,
        "summary_path": summary_path,
        "trades_df": trades_df,
        "daily_df": daily_df,
        "source": source_used,
        "symbol": normalized_symbol,
    }


def run_full_analysis(
    config: ToolkitConfig,
    symbol: str,
    label: str,
    start: str,
    end: str,
    lookback: int,
    pred_len: int,
    rebalance_step: int,
    signal_threshold: float,
    source: str,
    device: str,
    strategy_key: str,
) -> dict:
    prediction = run_prediction_analysis(
        config=config,
        symbol=symbol,
        label=label,
        start=start,
        end=end,
        lookback=lookback,
        pred_len=pred_len,
        source=source,
        device=device,
        strategy_key=strategy_key,
        signal_threshold=signal_threshold,
    )
    backtest = run_backtest_analysis(
        config=config,
        symbol=symbol,
        start=start,
        end=end,
        lookback=min(lookback, max(60, lookback)),
        pred_len=min(pred_len, 10),
        rebalance_step=rebalance_step,
        signal_threshold=signal_threshold,
        source=source,
        device=device,
        strategy_key=strategy_key,
    )
    return {"prediction": prediction, "backtest": backtest}
