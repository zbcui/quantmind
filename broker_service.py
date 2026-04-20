"""Broker service — MiniQMT (xtquant) integration for real broker order execution.

Architecture:
    QuantMind signal (BUY/SELL/HOLD)
        → BrokerService.execute_order()
            → xtquant XtQuantTrader → Broker → Exchange

Supports any broker that provides MiniQMT/QMT access (国金, 华泰, 中泰, 银河, etc).
User configures broker settings via the web UI (⚙ Broker Settings).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import ToolkitConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    LIMIT = "LIMIT"       # 限价单
    MARKET = "MARKET"     # 市价单

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Broker config (per-user, stored in SQLite)
# ---------------------------------------------------------------------------

@dataclass
class BrokerConfig:
    """User's broker connection settings."""
    enabled: bool = False
    mini_qmt_path: str = ""       # Path to MiniQMT userdata dir
    account_id: str = ""          # 资金账号
    broker_name: str = ""         # e.g. "国金证券", "华泰证券"
    max_order_value: float = 50000.0   # 单笔最大下单金额
    max_daily_orders: int = 20         # 每日最大下单次数
    require_confirmation: bool = True  # 下单前需要用户确认

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "mini_qmt_path": self.mini_qmt_path,
            "account_id": self.account_id,
            "broker_name": self.broker_name,
            "max_order_value": self.max_order_value,
            "max_daily_orders": self.max_daily_orders,
            "require_confirmation": self.require_confirmation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BrokerConfig:
        return cls(
            enabled=bool(d.get("enabled", False)),
            mini_qmt_path=str(d.get("mini_qmt_path", "")),
            account_id=str(d.get("account_id", "")),
            broker_name=str(d.get("broker_name", "")),
            max_order_value=float(d.get("max_order_value", 50000.0)),
            max_daily_orders=int(d.get("max_daily_orders", 20)),
            require_confirmation=bool(d.get("require_confirmation", True)),
        )


# ---------------------------------------------------------------------------
# Symbol conversion: QuantMind format → xtquant format
# ---------------------------------------------------------------------------

def to_xtquant_code(symbol: str) -> str:
    """Convert QuantMind symbol to xtquant format.

    QuantMind: SH601169, SZ000651, 601169
    xtquant:   601169.SH, 000651.SZ
    """
    code = symbol.upper().strip()
    if code.startswith("SH"):
        return code[2:] + ".SH"
    if code.startswith("SZ"):
        return code[2:] + ".SZ"
    if "." in code:
        return code
    if len(code) == 6 and code.isdigit():
        if code[0] == "6":
            return code + ".SH"
        return code + ".SZ"
    return code


# ---------------------------------------------------------------------------
# Risk checks
# ---------------------------------------------------------------------------

def _check_risk(
    broker_cfg: BrokerConfig,
    side: OrderSide,
    price: float,
    volume: int,
    daily_order_count: int,
) -> str | None:
    """Return error string if risk check fails, else None."""
    order_value = price * volume
    if order_value > broker_cfg.max_order_value:
        return (
            f"订单金额 ¥{order_value:,.0f} 超过单笔限额 ¥{broker_cfg.max_order_value:,.0f}。"
            f"请在 Broker 设置中调整限额。"
        )
    if daily_order_count >= broker_cfg.max_daily_orders:
        return (
            f"今日已下单 {daily_order_count} 次，达到每日上限 {broker_cfg.max_daily_orders}。"
        )
    if volume <= 0:
        return "下单数量必须大于 0"
    if volume % 100 != 0 and side == OrderSide.BUY:
        return f"A股买入必须为100股整数倍，当前: {volume}"
    if price <= 0:
        return "价格必须大于 0"
    return None


# ---------------------------------------------------------------------------
# XtQuant connection manager (singleton per user session)
# ---------------------------------------------------------------------------

_traders: dict[str, object] = {}  # account_id → XtQuantTrader
_trader_lock = threading.Lock()


def _get_trader(broker_cfg: BrokerConfig):
    """Get or create an XtQuantTrader instance for the given account."""
    try:
        from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
        from xtquant.xttype import StockAccount
    except ImportError:
        raise RuntimeError(
            "xtquant 未安装。请先安装 MiniQMT SDK:\n"
            "  pip install xtquant\n"
            "或从券商获取 xtquant 安装包。"
        )

    account_id = broker_cfg.account_id
    with _trader_lock:
        if account_id in _traders:
            return _traders[account_id], StockAccount(account_id)

        if not broker_cfg.mini_qmt_path:
            raise RuntimeError("未配置 MiniQMT 路径，请在 Broker 设置中填写。")

        path = Path(broker_cfg.mini_qmt_path)
        if not path.exists():
            raise RuntimeError(f"MiniQMT 路径不存在: {path}")

        class _Callback(XtQuantTraderCallback):
            def on_stock_order(self, order):
                log.info("委托回报: %s %s 状态=%s", order.stock_code,
                         order.order_id, order.order_status)

            def on_stock_trade(self, trade):
                log.info("成交回报: %s 成交量=%s 价格=%s", trade.stock_code,
                         trade.traded_volume, trade.traded_price)

            def on_order_error(self, order_error):
                log.error("下单错误: %s %s", order_error.order_id,
                          order_error.error_msg)

            def on_order_stock_async_response(self, response):
                log.info("异步下单响应: order_id=%s, seq=%s",
                         response.order_id, response.seq)

        session_id = int(time.time()) % 100000
        trader = XtQuantTrader(str(path), session_id)
        trader.register_callback(_Callback())
        trader.start()

        connect_result = trader.connect()
        if connect_result != 0:
            raise RuntimeError(
                f"连接 MiniQMT 失败 (code={connect_result})。\n"
                "请确保 MiniQMT 客户端已启动并登录。"
            )

        account = StockAccount(account_id)
        subscribe_result = trader.subscribe(account)
        if subscribe_result != 0:
            log.warning("订阅账户失败 (code=%s), 部分功能可能不可用", subscribe_result)

        _traders[account_id] = trader
        return trader, account


def disconnect_trader(account_id: str) -> None:
    """Disconnect and clean up a trader instance."""
    with _trader_lock:
        trader = _traders.pop(account_id, None)
        if trader:
            try:
                trader.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def execute_order(
    broker_cfg: BrokerConfig,
    symbol: str,
    side: OrderSide,
    volume: int,
    price: float,
    order_type: OrderType = OrderType.LIMIT,
    *,
    daily_order_count: int = 0,
) -> dict:
    """Execute a real broker order via MiniQMT.

    Returns dict with: order_id, status, message, timestamp, details
    """
    timestamp = datetime.now().isoformat(timespec="seconds")

    # Validate config
    if not broker_cfg.enabled:
        return {
            "order_id": None,
            "status": OrderStatus.FAILED.value,
            "message": "实盘交易未启用。请在 Broker 设置中开启。",
            "timestamp": timestamp,
        }

    if not broker_cfg.account_id:
        return {
            "order_id": None,
            "status": OrderStatus.FAILED.value,
            "message": "未配置资金账号。请在 Broker 设置中填写。",
            "timestamp": timestamp,
        }

    # Risk checks
    risk_err = _check_risk(broker_cfg, side, price, volume, daily_order_count)
    if risk_err:
        return {
            "order_id": None,
            "status": OrderStatus.REJECTED.value,
            "message": f"风控拦截: {risk_err}",
            "timestamp": timestamp,
        }

    # Convert symbol
    xt_code = to_xtquant_code(symbol)

    try:
        from xtquant import xtconstant

        trader, account = _get_trader(broker_cfg)

        # Map order side
        xt_side = (xtconstant.STOCK_BUY if side == OrderSide.BUY
                   else xtconstant.STOCK_SELL)

        # Map order type
        xt_price_type = (xtconstant.FIX_PRICE if order_type == OrderType.LIMIT
                         else xtconstant.LATEST_PRICE)

        # Execute async order
        seq = trader.order_stock_async(
            account=account,
            stock_code=xt_code,
            order_type=xt_side,
            order_volume=volume,
            price_type=xt_price_type,
            price=price,
            strategy_name="QuantMind",
            order_remark=f"QuantMind-{side.value}",
        )

        return {
            "order_id": str(seq),
            "status": OrderStatus.SUBMITTED.value,
            "message": f"{'买入' if side == OrderSide.BUY else '卖出'} {xt_code} "
                       f"{volume}股 @ ¥{price:.3f} 已提交",
            "timestamp": timestamp,
            "details": {
                "symbol": xt_code,
                "side": side.value,
                "volume": volume,
                "price": price,
                "order_type": order_type.value,
                "seq": seq,
            },
        }

    except ImportError:
        return {
            "order_id": None,
            "status": OrderStatus.FAILED.value,
            "message": "xtquant 未安装。请安装 MiniQMT SDK: pip install xtquant",
            "timestamp": timestamp,
        }
    except RuntimeError as e:
        return {
            "order_id": None,
            "status": OrderStatus.FAILED.value,
            "message": str(e),
            "timestamp": timestamp,
        }
    except Exception as e:
        log.exception("下单异常")
        return {
            "order_id": None,
            "status": OrderStatus.FAILED.value,
            "message": f"下单异常: {e}",
            "timestamp": timestamp,
        }


# ---------------------------------------------------------------------------
# Account queries
# ---------------------------------------------------------------------------

def query_account_asset(broker_cfg: BrokerConfig) -> dict:
    """Query account cash, total asset, positions."""
    try:
        trader, account = _get_trader(broker_cfg)
        asset = trader.query_stock_asset(account)
        if asset is None:
            return {"error": "查询资产失败，请检查连接状态"}
        return {
            "account_id": broker_cfg.account_id,
            "cash": asset.cash,
            "total_asset": asset.total_asset,
            "market_value": asset.market_value,
            "frozen_cash": asset.frozen_cash,
        }
    except Exception as e:
        return {"error": str(e)}


def query_positions(broker_cfg: BrokerConfig) -> list[dict]:
    """Query current stock positions."""
    try:
        trader, account = _get_trader(broker_cfg)
        positions = trader.query_stock_positions(account)
        if positions is None:
            return []
        return [
            {
                "stock_code": p.stock_code,
                "volume": p.volume,
                "can_use_volume": p.can_use_volume,
                "open_price": p.open_price,
                "market_value": p.market_value,
                "frozen_volume": p.frozen_volume,
            }
            for p in positions
        ]
    except Exception as e:
        log.exception("查询持仓异常")
        return []


def query_orders_today(broker_cfg: BrokerConfig) -> list[dict]:
    """Query today's orders."""
    try:
        trader, account = _get_trader(broker_cfg)
        orders = trader.query_stock_orders(account)
        if orders is None:
            return []
        return [
            {
                "order_id": o.order_id,
                "stock_code": o.stock_code,
                "order_type": o.order_type,
                "order_volume": o.order_volume,
                "price": o.price,
                "traded_volume": o.traded_volume,
                "traded_price": o.traded_price,
                "order_status": o.order_status,
                "order_time": o.order_time,
                "order_remark": getattr(o, "order_remark", ""),
            }
            for o in orders
        ]
    except Exception as e:
        log.exception("查询委托异常")
        return []


def cancel_order(broker_cfg: BrokerConfig, order_id: str) -> dict:
    """Cancel a pending order."""
    try:
        from xtquant.xttype import StockAccount
        trader, account = _get_trader(broker_cfg)
        result = trader.cancel_order_stock_async(account, int(order_id))
        return {"success": True, "message": f"撤单请求已发送 (order_id={order_id})", "seq": result}
    except Exception as e:
        return {"success": False, "message": f"撤单失败: {e}"}


# ---------------------------------------------------------------------------
# High-level: signal → order (used by QuantMind pipeline)
# ---------------------------------------------------------------------------

def execute_signal(
    broker_cfg: BrokerConfig,
    symbol: str,
    action: str,
    price: float,
    *,
    position_shares: int = 0,
    available_cash: float = 0.0,
    lot_size: int = 100,
    daily_order_count: int = 0,
) -> dict:
    """Convert a QuantMind BUY/SELL/HOLD signal into a real broker order.

    This is the main entry point for automated trading.
    """
    action = action.upper().strip()

    if action == "HOLD":
        return {
            "order_id": None,
            "status": "HOLD",
            "message": "信号为 HOLD，不执行交易",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    if action == "BUY":
        if position_shares > 0:
            return {
                "order_id": None,
                "status": "SKIP",
                "message": f"已持有 {position_shares} 股，跳过买入",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        # Calculate buy volume: use 95% of cash, round to lot_size
        budget = available_cash * 0.95
        lots = int(budget // (price * lot_size))
        volume = lots * lot_size
        if volume <= 0:
            return {
                "order_id": None,
                "status": "SKIP",
                "message": f"可用资金不足 (¥{available_cash:,.0f})，无法买入",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        return execute_order(
            broker_cfg, symbol, OrderSide.BUY, volume, price,
            daily_order_count=daily_order_count,
        )

    if action == "SELL":
        if position_shares <= 0:
            return {
                "order_id": None,
                "status": "SKIP",
                "message": "无持仓，跳过卖出",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        return execute_order(
            broker_cfg, symbol, OrderSide.SELL, position_shares, price,
            daily_order_count=daily_order_count,
        )

    return {
        "order_id": None,
        "status": "UNKNOWN",
        "message": f"未知信号: {action}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
