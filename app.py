from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from analysis_service import run_backtest_analysis, run_prediction_analysis
from config import ToolkitConfig
from data_sources import get_kline_data, get_last_close, get_realtime_quote, get_stock_name, get_t0_indicators, normalize_symbol
from llm_service import analyze_t0, load_llm_config, save_llm_config
from strategy_catalog import strategy_options
import trading_agents_service as ta_service
from trading_service import (
    execute_paper_trade,
    export_manual_live_order,
    get_paper_portfolio_summary,
    load_live_state,
    load_paper_state,
    sync_live_portfolio,
)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


def default_form_data() -> dict:
    return {
        "symbol": "601169",
        "label": "601169",
        "label_auto": True,
        "start": "2020-01-01",
        "end": "2026-12-31",
        "lookback": 400,
        "pred_len": 20,
        "backtest_pred_len": 5,
        "rebalance_step": 20,
        "signal_threshold": 0.0,
        "source": "auto",
        "device": "cpu",
        "strategy": ToolkitConfig().recommended_strategy,
        "live_current_shares": 0,
        "live_avg_price": 0.0,
        "live_available_cash": ToolkitConfig().default_paper_cash,
    }


def parse_form(payload: dict) -> dict:
    defaults = default_form_data()
    data = defaults | payload
    data["symbol"] = str(data["symbol"]).strip()
    data["label"] = str(data["label"]).strip()
    data["label_auto"] = str(data.get("label_auto", defaults["label_auto"])).lower() in ("1", "true", "yes", "on")
    data["lookback"] = int(data["lookback"])
    data["pred_len"] = int(data["pred_len"])
    data["backtest_pred_len"] = int(data["backtest_pred_len"])
    data["rebalance_step"] = int(data["rebalance_step"])
    data["signal_threshold"] = float(data["signal_threshold"])
    data["live_current_shares"] = int(data["live_current_shares"])
    data["live_avg_price"] = float(data["live_avg_price"])
    data["live_available_cash"] = float(data["live_available_cash"])
    if data["label_auto"] or not data["label"]:
        data["label"] = data["symbol"]
    return data


def relative_output_path(path: Path, config: ToolkitConfig) -> str:
    return str(path.relative_to(config.output_dir)).replace("\\", "/")


def execute_analysis(form_data: dict) -> dict:
    config = ToolkitConfig()
    config.ensure_directories()

    prediction = run_prediction_analysis(
        config=config,
        symbol=form_data["symbol"],
        label=form_data["label"],
        start=form_data["start"],
        end=form_data["end"],
        lookback=form_data["lookback"],
        pred_len=form_data["pred_len"],
        source=form_data["source"],
        device=form_data["device"],
        strategy_key=form_data["strategy"],
        signal_threshold=form_data["signal_threshold"],
    )
    backtest = run_backtest_analysis(
        config=config,
        symbol=form_data["symbol"],
        start=form_data["start"],
        end=form_data["end"],
        lookback=min(form_data["lookback"], 240),
        pred_len=form_data["backtest_pred_len"],
        rebalance_step=form_data["rebalance_step"],
        signal_threshold=form_data["signal_threshold"],
        source=form_data["source"],
        device=form_data["device"],
        strategy_key=form_data["strategy"],
    )
    market_price = prediction["summary"]["last_close"]
    paper_portfolio = get_paper_portfolio_summary(
        config=config,
        symbol=prediction["summary"]["symbol"],
        market_price=market_price,
    )
    live_portfolio = sync_live_portfolio(
        config=config,
        symbol=prediction["summary"]["symbol"],
        current_shares=form_data["live_current_shares"],
        avg_price=form_data["live_avg_price"],
        available_cash=form_data["live_available_cash"],
        market_price=market_price,
    )

    return {
        "form": form_data,
        "prediction": {
            "summary": prediction["summary"],
            "recommendation": prediction["recommendation"],
            "forecast_image": relative_output_path(prediction["forecast_path"], config),
            "summary_file": relative_output_path(prediction["summary_path"], config),
            "prediction_file": relative_output_path(prediction["prediction_path"], config),
            "history_file": relative_output_path(prediction["history_path"], config),
        },
        "backtest": {
            "summary": backtest["summary"],
            "summary_file": relative_output_path(backtest["summary_path"], config),
            "trades_file": relative_output_path(backtest["trades_path"], config),
            "daily_file": relative_output_path(backtest["daily_path"], config),
        },
        "paper_portfolio": {
            "summary": paper_portfolio["portfolio"],
            "state_file": relative_output_path(Path(paper_portfolio["paper_state_file"]), config),
            "database_file": relative_output_path(Path(paper_portfolio["database_file"]), config),
        },
        "live_portfolio": {
            "summary": live_portfolio["portfolio"],
            "state_file": relative_output_path(Path(live_portfolio["live_state_file"]), config),
            "database_file": relative_output_path(Path(live_portfolio["database_file"]), config),
        },
    }


def _best_execution_price(symbol: str, fallback_close: float) -> tuple[float, str]:
    """Return (price, source_label). Prefers real-time quote during market hours."""
    try:
        quote = get_realtime_quote(symbol)
        if quote and quote.get("is_realtime") and quote.get("last_price"):
            return float(quote["last_price"]), "realtime"
    except Exception:
        pass
    return float(fallback_close), "last_close"


def execute_paper_trade_action(form_data: dict, analysis_result: dict) -> dict:
    config = ToolkitConfig()
    recommendation = analysis_result["prediction"]["recommendation"]
    symbol = analysis_result["prediction"]["summary"]["symbol"]
    fallback = analysis_result["prediction"]["summary"]["last_close"]
    execution_price, price_source = _best_execution_price(symbol, fallback)
    result = execute_paper_trade(
        config=config,
        symbol=symbol,
        recommendation=recommendation,
        execution_price=execution_price,
    )
    result["trade"]["price_source"] = "🔴 实时价" if price_source == "realtime" else "📅 收盘价"

    # Reflection learning: when a SELL is executed, feed the realized return to TA agents
    if result["trade"].get("status") == "sold":
        avg_buy = result["trade"].get("avg_buy_price", 0.0)
        sell_price = result["trade"].get("execution_price", 0.0)
        if avg_buy and avg_buy > 0:
            pnl_pct = round((sell_price / avg_buy - 1.0) * 100, 4)
            try:
                ta_service.reflect(config, pnl_pct)
            except Exception:
                pass

    return result


def execute_live_export_action(form_data: dict, analysis_result: dict) -> dict:
    config = ToolkitConfig()
    recommendation = analysis_result["prediction"]["recommendation"]
    symbol = analysis_result["prediction"]["summary"]["symbol"]
    fallback = analysis_result["prediction"]["summary"]["last_close"]
    execution_price, price_source = _best_execution_price(symbol, fallback)
    export = export_manual_live_order(
        config=config,
        symbol=analysis_result["prediction"]["summary"]["symbol"],
        recommendation=recommendation,
        execution_price=execution_price,
        current_shares=form_data["live_current_shares"],
        avg_price=form_data["live_avg_price"],
        available_cash=form_data["live_available_cash"],
    )
    export["order"]["price_source"] = "🔴 实时价" if price_source == "realtime" else "📅 收盘价"
    export["order_file_relative"] = relative_output_path(Path(export["order_file"]), config)
    export["database_file_relative"] = relative_output_path(Path(export["database_file"]), config)
    return export


@app.route("/", methods=["GET", "POST"])
def index():
    form_data = default_form_data()
    result = None
    error = None
    trade_result = None
    if request.method == "POST":
        try:
            form_data = parse_form(request.form.to_dict())
            result = execute_analysis(form_data)
            action = request.form.get("action", "analyze")
            if action == "paper_trade":
                trade_result = {"paper": execute_paper_trade_action(form_data, result)}
            elif action == "live_export":
                trade_result = {"live": execute_live_export_action(form_data, result)}
        except Exception as exc:
            error = str(exc)
    return render_template(
        "index.html",
        form=form_data,
        result=result,
        error=error,
        trade_result=trade_result,
        strategy_options=strategy_options(),
        now_date=__import__("datetime").date.today().isoformat(),
    )


@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    payload = request.get_json(silent=True) or {}
    try:
        form_data = parse_form(payload)
        result = execute_analysis(form_data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/paper-trade", methods=["POST"])
def paper_trade_api():
    payload = request.get_json(silent=True) or {}
    try:
        form_data = parse_form(payload)
        result = execute_analysis(form_data)
        trade_result = execute_paper_trade_action(form_data, result)
        return jsonify({"analysis": result, "paper_trade": trade_result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/live-export", methods=["POST"])
def live_export_api():
    payload = request.get_json(silent=True) or {}
    try:
        form_data = parse_form(payload)
        result = execute_analysis(form_data)
        trade_result = execute_live_export_action(form_data, result)
        return jsonify({"analysis": result, "live_export": trade_result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/outputs/<path:filename>")
def outputs(filename: str):
    config = ToolkitConfig()
    config.ensure_directories()
    return send_from_directory(config.output_dir, filename)


@app.route("/api/stock-name/<symbol>")
def stock_name_api(symbol: str):
    try:
        sym = normalize_symbol(symbol.strip())
        name = get_stock_name(sym)
        return jsonify({"symbol": sym, "name": name})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/realtime-quote/<symbol>")
def realtime_quote_api(symbol: str):
    try:
        sym = normalize_symbol(symbol.strip())
        quote = get_realtime_quote(sym)
        if quote is None:
            return jsonify({"error": "Quote not available"}), 404
        return jsonify(quote)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/kline/<symbol>")
def kline_api(symbol: str):
    """Return OHLCV data for candlestick chart.
    ?period=daily|weekly|monthly|5min|15min|30min|60min
    ?days=N  (number of calendar days to look back, default 180 for daily)
    """
    try:
        sym = normalize_symbol(symbol.strip())
        period = request.args.get("period", "daily")
        days = int(request.args.get("days", 180))
        data = get_kline_data(sym, period=period, days=days)
        if data is None:
            return jsonify({"error": "Data not available"}), 404
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/llm-config", methods=["GET", "POST"])
def llm_config_api():
    if request.method == "POST":
        try:
            cfg = request.get_json(silent=True) or {}
            save_llm_config(cfg)
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
    cfg = load_llm_config()
    cfg.pop("api_key", None)  # never send key back to browser
    return jsonify(cfg)


@app.route("/api/t0-indicators/<symbol>")
def t0_indicators_api(symbol: str):
    try:
        sym = normalize_symbol(symbol.strip())
        force = request.args.get("force", "0") == "1"
        data = get_t0_indicators(sym, force=force)
        if data is None:
            return jsonify({"error": "Indicator data unavailable — market may be closed"}), 404
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/t0-analysis", methods=["POST"])
def t0_analysis_api():
    payload = request.get_json(silent=True) or {}
    try:
        sym = normalize_symbol(payload.get("symbol", "").strip())
        indicators = get_t0_indicators(sym)
        if indicators is None:
            return jsonify({"error": "No intraday data available"}), 404
        text = analyze_t0(sym, indicators)
        return jsonify({"symbol": sym, "indicators": indicators, "analysis": text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")


@app.route("/api/portfolio-data")
def portfolio_data_api():
    config = ToolkitConfig()
    config.ensure_directories()
    paper_state = load_paper_state(config)
    live_state = load_live_state(config)

    all_symbols: set[str] = set()
    for sym, pos in paper_state.get("positions", {}).items():
        if pos.get("shares", 0) > 0:
            all_symbols.add(sym)
    for sym, pos in live_state.get("positions", {}).items():
        if pos.get("shares", 0) > 0:
            all_symbols.add(sym)

    prices: dict[str, float] = {}
    for sym in all_symbols:
        price = get_last_close(sym, config)
        if price is not None:
            prices[sym] = price

    def build_positions(state: dict) -> dict:
        result: dict = {}
        for sym, pos in state.get("positions", {}).items():
            shares = int(pos.get("shares", 0))
            avg_price = float(pos.get("avg_price", 0.0))
            last_price = prices.get(sym, avg_price)
            market_value = round(shares * last_price, 2)
            cost_basis = round(shares * avg_price, 2)
            unrealized_pnl = round(market_value - cost_basis, 2)
            unrealized_pct = round((last_price / avg_price - 1.0) * 100, 2) if avg_price > 0 and shares > 0 else 0.0
            result[sym] = {
                "name": get_stock_name(sym),
                "shares": shares,
                "avg_price": avg_price,
                "last_price": last_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pct": unrealized_pct,
            }
        return result

    paper_positions = build_positions(paper_state)
    live_positions = build_positions(live_state)

    paper_initial = float(paper_state.get("initial_equity", paper_state.get("initial_cash", 0.0)))
    paper_cash = round(float(paper_state.get("cash", 0.0)), 2)
    paper_total = round(paper_cash + sum(p["market_value"] for p in paper_positions.values()), 2)
    paper_return_pct = round((paper_total / paper_initial - 1.0) * 100, 2) if paper_initial > 0 else 0.0

    live_initial = float(live_state.get("initial_equity", 0.0))
    live_cash = round(float(live_state.get("cash", 0.0)), 2)
    live_total = round(live_cash + sum(p["market_value"] for p in live_positions.values()), 2)
    live_return_pct = round((live_total / live_initial - 1.0) * 100, 2) if live_initial > 0 else 0.0

    paper_curve = []
    if paper_initial > 0:
        paper_curve.append({"t": "Initial", "equity": paper_initial, "status": "initial", "action": None})
    for trade in paper_state.get("trade_history", []):
        equity = round(
            float(trade["cash_after"]) + int(trade["position_after"]["shares"]) * float(trade["execution_price"]),
            2,
        )
        paper_curve.append({
            "t": trade["timestamp"],
            "equity": equity,
            "status": trade["status"],
            "action": trade["action"],
            "symbol": trade["symbol"],
            "price": trade["execution_price"],
            "shares": trade["shares_delta"],
        })

    live_curve = [
        {
            "t": s["timestamp"],
            "equity": s["total_equity"],
            "status": "sync",
            "symbol": s["symbol"],
            "shares": s["shares"],
            "price": s["market_price"],
        }
        for s in live_state.get("sync_history", [])
    ]

    return jsonify({
        "paper": {
            "initial_equity": paper_initial,
            "cash": paper_cash,
            "realized_pnl": round(float(paper_state.get("realized_pnl", 0.0)), 2),
            "total_equity": paper_total,
            "total_return_pct": paper_return_pct,
            "positions": paper_positions,
            "equity_curve": paper_curve,
        },
        "live": {
            "initial_equity": live_initial,
            "cash": live_cash,
            "realized_pnl": round(float(live_state.get("realized_pnl", 0.0)), 2),
            "total_equity": live_total,
            "total_return_pct": live_return_pct,
            "positions": live_positions,
            "equity_curve": live_curve,
        },
    })


@app.route("/api/ta-analysis", methods=["POST"])
def ta_analysis_api():
    """Fire a TradingAgents multi-agent analysis job. Returns job_id immediately."""
    payload = request.get_json(silent=True) or {}
    try:
        raw_symbol = payload.get("symbol", "").strip()
        if not raw_symbol:
            return jsonify({"error": "symbol is required"}), 400
        symbol = normalize_symbol(raw_symbol)
        trade_date = payload.get("date") or ""
        config = ToolkitConfig()
        config.ensure_directories()

        # Quick pre-flight LLM check before queuing the job
        from llm_service import load_llm_config as _llm_cfg
        llm_err = ta_service._check_llm_reachable(_llm_cfg())
        if llm_err:
            return jsonify({"error": llm_err}), 400

        job_id = ta_service.submit_job(config, symbol, trade_date or None)
        return jsonify({"job_id": job_id, "symbol": symbol, "status": "pending"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/ta-status/<job_id>")
def ta_status_api(job_id: str):
    """Poll the status and result of a TradingAgents job."""
    config = ToolkitConfig()
    job = ta_service.get_job(config, job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/ta-latest/<symbol>")
def ta_latest_api(symbol: str):
    """Return the most recent completed TradingAgents analysis for a symbol."""
    try:
        sym = normalize_symbol(symbol.strip())
        config = ToolkitConfig()
        result = ta_service.get_latest(config, sym)
        if result is None:
            return jsonify({"error": "No completed analysis found"}), 404
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/consensus/<symbol>")
def consensus_api(symbol: str):
    """Merge the latest Kronos recommendation with the latest TA decision."""
    try:
        sym = normalize_symbol(symbol.strip())
        config = ToolkitConfig()
        ta_result = ta_service.get_latest(config, sym)
        if ta_result is None:
            return jsonify({"error": "No TA analysis found — run agent analysis first"}), 404

        # Try to load last Kronos recommendation from the most recent summary JSON
        kronos_action = "HOLD"
        summary_path = config.output_dir / f"{sym}_summary.json"
        if summary_path.exists():
            import json as _json
            try:
                summary = _json.loads(summary_path.read_text(encoding="utf-8"))
                kronos_action = summary.get("recommended_action", "HOLD")
            except Exception:
                pass

        consensus = ta_service.build_consensus(kronos_action, ta_result["decision"] or "HOLD")
        return jsonify({
            "symbol":        sym,
            "kronos_action": kronos_action,
            "ta_decision":   ta_result["decision"],
            "ta_trade_date": ta_result["trade_date"],
            "consensus":     consensus,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7080, debug=False)
