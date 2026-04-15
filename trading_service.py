from __future__ import annotations

import json
from datetime import datetime

from config import ToolkitConfig
from trade_storage import (
    load_portfolio_state,
    record_live_order,
    record_live_sync,
    record_paper_trade,
    save_portfolio_state,
)


def _default_paper_state(config: ToolkitConfig) -> dict:
    return {
        "initial_cash": config.default_paper_cash,
        "cash": config.default_paper_cash,
        "positions": {},
        "trade_history": [],
        "realized_pnl": 0.0,
    }


def _default_live_state(config: ToolkitConfig) -> dict:
    return {
        "initial_cash": 0.0,
        "initial_equity": 0.0,
        "cash": config.default_paper_cash,
        "positions": {},
        "sync_history": [],
        "realized_pnl": 0.0,
    }


def _save_json_state(path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_paper_state(config: ToolkitConfig) -> dict:
    config.live_order_dir.mkdir(parents=True, exist_ok=True)
    state = load_portfolio_state(config, "paper")
    if "initial_equity" not in state:
        state["initial_equity"] = float(state.get("initial_cash", 0.0))
    if "trade_history" not in state:
        state["trade_history"] = []
    _save_json_state(config.paper_state_path, state)
    return state


def load_live_state(config: ToolkitConfig) -> dict:
    config.live_order_dir.mkdir(parents=True, exist_ok=True)
    state = load_portfolio_state(config, "live")
    if "sync_history" not in state:
        state["sync_history"] = []
    _save_json_state(config.live_state_path, state)
    return state


def _position_snapshot(position: dict, market_price: float) -> dict:
    shares = int(position.get("shares", 0))
    avg_price = float(position.get("avg_price", 0.0))
    market_value = round(shares * market_price, 2)
    cost_basis = round(shares * avg_price, 2)
    unrealized_pnl = round(market_value - cost_basis, 2)
    unrealized_return_pct = round((market_price / avg_price - 1.0) * 100, 2) if shares > 0 and avg_price > 0 else 0.0
    return {
        "shares": shares,
        "avg_price": round(avg_price, 4),
        "market_price": round(market_price, 4),
        "market_value": market_value,
        "cost_basis": cost_basis,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_return_pct": unrealized_return_pct,
    }


def build_portfolio_summary(state: dict, symbol: str, market_price: float) -> dict:
    position = state.get("positions", {}).get(symbol, {"shares": 0, "avg_price": 0.0})
    snapshot = _position_snapshot(position, market_price)
    total_equity = round(float(state.get("cash", 0.0)) + snapshot["market_value"], 2)
    baseline_equity = float(state.get("initial_equity", state.get("initial_cash", 0.0)))
    total_return_pct = round((total_equity / baseline_equity - 1.0) * 100, 2) if baseline_equity > 0 else 0.0
    return {
        "cash": round(float(state.get("cash", 0.0)), 2),
        "position": snapshot,
        "realized_pnl": round(float(state.get("realized_pnl", 0.0)), 2),
        "total_equity": total_equity,
        "total_return_pct": total_return_pct,
        "trade_count": len(state.get("trade_history", state.get("sync_history", []))),
    }


def get_paper_portfolio_summary(config: ToolkitConfig, symbol: str, market_price: float) -> dict:
    state = load_paper_state(config)
    return {
        "portfolio": build_portfolio_summary(state, symbol, market_price),
        "paper_state_file": str(config.paper_state_path),
        "database_file": str(config.trading_db_path),
    }


def sync_live_portfolio(
    config: ToolkitConfig,
    symbol: str,
    current_shares: int,
    avg_price: float,
    available_cash: float,
    market_price: float,
) -> dict:
    state = load_live_state(config)
    state["cash"] = round(float(available_cash), 2)
    state["positions"][symbol] = {
        "shares": int(current_shares),
        "avg_price": round(float(avg_price), 4),
    }
    current_equity = round(float(available_cash) + int(current_shares) * float(market_price), 2)
    if float(state.get("initial_equity", 0.0)) <= 0:
        state["initial_equity"] = current_equity
        state["initial_cash"] = round(float(available_cash), 2)
    sync_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "shares": int(current_shares),
        "avg_price": round(float(avg_price), 4),
        "cash": round(float(available_cash), 2),
        "market_price": round(float(market_price), 4),
        "total_equity": current_equity,
        "unrealized_pnl": round((float(market_price) - float(avg_price)) * int(current_shares), 2),
    }
    state["sync_history"].append(sync_record)
    save_portfolio_state(config, "live", state)
    record_live_sync(config, sync_record)
    state = load_live_state(config)
    return {
        "portfolio": build_portfolio_summary(state, symbol, market_price),
        "live_state_file": str(config.live_state_path),
        "database_file": str(config.trading_db_path),
    }


def execute_paper_trade(
    config: ToolkitConfig,
    symbol: str,
    recommendation: dict,
    execution_price: float,
) -> dict:
    state = load_paper_state(config)
    position = state["positions"].get(symbol, {"shares": 0, "avg_price": 0.0})
    action = recommendation["action"]
    timestamp = datetime.now().isoformat(timespec="seconds")
    shares_delta = 0
    status = "no_trade"

    if action == "BUY" and position["shares"] == 0:
        budget = state["cash"] * 0.95
        lots = int(budget // (execution_price * config.lot_size))
        shares_delta = lots * config.lot_size
        if shares_delta > 0:
            cost = round(shares_delta * execution_price, 2)
            state["cash"] = round(state["cash"] - cost, 2)
            position = {"shares": shares_delta, "avg_price": execution_price}
            state["positions"][symbol] = position
            status = "bought"
    elif action == "SELL" and position["shares"] > 0:
        shares_delta = -position["shares"]
        proceeds = round(position["shares"] * execution_price, 2)
        realized = round((execution_price - position["avg_price"]) * position["shares"], 2)
        avg_buy_price = round(float(position["avg_price"]), 4)
        state["cash"] = round(state["cash"] + proceeds, 2)
        state["realized_pnl"] = round(state.get("realized_pnl", 0.0) + realized, 2)
        position = {"shares": 0, "avg_price": 0.0}
        state["positions"][symbol] = position
        status = "sold"

    trade = {
        "timestamp": timestamp,
        "symbol": symbol,
        "action": action,
        "execution_price": round(float(execution_price), 4),
        "shares_delta": shares_delta,
        "cash_after": round(float(state["cash"]), 2),
        "position_after": position,
        "status": status,
        "strategy": recommendation["strategy"],
    }
    if status == "sold":
        trade["avg_buy_price"] = avg_buy_price
    state["trade_history"].append(trade)
    save_portfolio_state(config, "paper", state)
    record_paper_trade(config, trade)
    state = load_paper_state(config)
    return {
        "trade": trade,
        "portfolio": build_portfolio_summary(state, symbol, execution_price),
        "paper_state_file": str(config.paper_state_path),
        "database_file": str(config.trading_db_path),
    }


def export_manual_live_order(
    config: ToolkitConfig,
    symbol: str,
    recommendation: dict,
    execution_price: float,
    current_shares: int,
    avg_price: float,
    available_cash: float,
) -> dict:
    config.live_order_dir.mkdir(parents=True, exist_ok=True)
    live_summary = sync_live_portfolio(
        config=config,
        symbol=symbol,
        current_shares=current_shares,
        avg_price=avg_price,
        available_cash=available_cash,
        market_price=execution_price,
    )

    target_shares = current_shares
    if recommendation["action"] == "BUY" and current_shares == 0:
        lots = int((available_cash * 0.95) // (execution_price * config.lot_size))
        target_shares = lots * config.lot_size
    elif recommendation["action"] == "SELL":
        target_shares = 0

    order = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "strategy": recommendation["strategy"],
        "recommended_action": recommendation["action"],
        "execution_price_reference": round(float(execution_price), 4),
        "current_shares": int(current_shares),
        "current_avg_price": round(float(avg_price), 4),
        "target_shares": int(target_shares),
        "order_shares_delta": int(target_shares - current_shares),
        "available_cash": round(float(available_cash), 2),
        "note": "Manual live order export only. Review before sending to a broker.",
        "rationale": recommendation["rationale"],
    }
    filename = config.live_order_dir / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filename.write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8")
    record_live_order(config, order, str(filename))
    return {
        "order": order,
        "order_file": str(filename),
        "portfolio": live_summary["portfolio"],
        "live_state_file": live_summary["live_state_file"],
        "database_file": str(config.trading_db_path),
    }
