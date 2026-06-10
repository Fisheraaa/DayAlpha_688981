"""
src/backtest/costs.py — 688981 科创板交易成本模型
======================================================
佣金更新：万2.5 (0.00025 per side)
往返总成本明细：
  买入: 佣金 0.025% + 经手费 0.00487% + 滑点 0.02% ≈ 0.0499%
  卖出: 佣金 0.025% + 印花税 0.10%  + 经手费 0.00487% + 滑点 0.02% ≈ 0.1499%
  往返合计 ≈ 0.1998%（原来约 0.31%，降低约35%）
"""
from __future__ import annotations
from functools import lru_cache
from src.utils import load_config

@lru_cache(maxsize=1)
def _c() -> dict:
    return load_config()["backtest"]

def buy_cost(price: float, shares: float) -> float:
    """买入总成本（含佣金、经手费、滑点，不含印花税）。"""
    c = _c()
    rate = c["commission_rate"] + c["exchange_fee_rate"] + c["slippage"]
    return price * shares * (1.0 + rate)

def sell_proceeds(price: float, shares: float) -> float:
    """卖出实收（扣除佣金、印花税、经手费、滑点）。"""
    c = _c()
    rate = (c["commission_rate"] + c["stamp_duty_rate"]
            + c["exchange_fee_rate"] + c["slippage"])
    return price * shares * (1.0 - rate)

def round_trip_cost_rate() -> float:
    """往返单位成本率。"""
    c = _c()
    buy_rate  = c["commission_rate"] + c["exchange_fee_rate"] + c["slippage"]
    sell_rate = (c["commission_rate"] + c["stamp_duty_rate"]
                 + c["exchange_fee_rate"] + c["slippage"])
    return buy_rate + sell_rate

def min_profit_threshold() -> float:
    """最低盈利门槛 = 往返成本 + 安全边际 0.5bp。"""
    return round_trip_cost_rate() + 0.00005

def cost_summary() -> dict:
    """成本明细（单位: bp = 万分之一）。"""
    c = _c()
    rrt = round_trip_cost_rate()
    return {
        "commission_bp":  c["commission_rate"] * 10000,
        "stamp_duty_bp":  c["stamp_duty_rate"] * 10000,
        "exchange_fee_bp":c["exchange_fee_rate"] * 10000,
        "slippage_bp":    c["slippage"] * 10000,
        "round_trip_bp":  rrt * 10000,
        "min_profit_bp":  min_profit_threshold() * 10000,
    }
