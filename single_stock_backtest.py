from __future__ import annotations

import argparse
import json

from analysis_service import run_backtest_analysis
from config import ToolkitConfig
from data_sources import normalize_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a rolling single-stock backtest using Kronos signals.")
    parser.add_argument("--symbol", required=True, help="Stock code, e.g. 601169 or SH601169")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--rebalance-step", type=int, default=5)
    parser.add_argument("--signal-threshold", type=float, default=0.0)
    parser.add_argument("--source", choices=["auto", "qlib", "akshare"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strategy", default="forecast_trend")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = ToolkitConfig()
    config.ensure_directories()
    symbol = normalize_symbol(args.symbol)

    result = run_backtest_analysis(
        config=config,
        symbol=symbol,
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        pred_len=args.pred_len,
        rebalance_step=args.rebalance_step,
        signal_threshold=args.signal_threshold,
        source=args.source,
        device=args.device,
        strategy_key=args.strategy,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
