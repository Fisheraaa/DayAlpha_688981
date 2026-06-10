"""
src/backtest/metrics.py — 5分钟K线策略绩效指标
================================================
新增：
  - omega_ratio：收益/亏损比（比 profit_factor 更精确）
  - avg_win / avg_loss / payoff_ratio
  - max_consecutive_loss：最大连续亏损次数
  - trade_type_breakdown：各类型平仓占比
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import Counter

BARS_PER_DAY = 8       # 30分钟K线，每天8根（09:30-11:30 = 4根，13:00-15:00 = 4根）


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    ret = equity.pct_change().dropna()
    ex  = ret - rf / (252 * BARS_PER_DAY)
    return float(ex.mean() / ex.std() * np.sqrt(252 * BARS_PER_DAY)) if ex.std() > 0 else 0.0


def sortino(equity: pd.Series, rf: float = 0.0) -> float:
    ret  = equity.pct_change().dropna()
    ex   = ret - rf / (252 * BARS_PER_DAY)
    down = ex[ex < 0]
    return float(ex.mean() / down.std() * np.sqrt(252 * BARS_PER_DAY)) if len(down) > 0 and down.std() > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    dd = (equity - equity.cummax()) / equity.cummax()
    return float(-dd.min())


def calmar(equity: pd.Series) -> float:
    n   = len(equity)
    ann = (equity.iloc[-1] / equity.iloc[0]) ** (252 * BARS_PER_DAY / n) - 1
    mdd = max_drawdown(equity)
    return float(ann / mdd) if mdd > 0 else 0.0


def omega_ratio(equity: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: 超额收益面积 / 亏损面积。"""
    ret = equity.pct_change().dropna() - threshold
    pos = ret[ret > 0].sum()
    neg = abs(ret[ret < 0].sum())
    return float(pos / neg) if neg > 0 else float("inf")


def win_rate(trades: list) -> float:
    return sum(1 for t in trades if t["pnl"] > 0) / len(trades) if trades else 0.0


def profit_factor(trades: list) -> float:
    wins   = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    if wins and losses:
        return float(sum(wins) / abs(sum(losses)))
    return 0.0


def payoff_ratio(trades: list) -> float:
    """平均盈利 / 平均亏损（绝对值）。"""
    wins   = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in trades if t["pnl"] < 0]
    if wins and losses:
        return float(np.mean(wins) / np.mean(losses))
    return 0.0


def max_consecutive_loss(trades: list) -> int:
    """最大连续亏损交易次数。"""
    max_seq = cur = 0
    for t in trades:
        if t["pnl"] < 0:
            cur += 1
            max_seq = max(max_seq, cur)
        else:
            cur = 0
    return max_seq


def trade_type_breakdown(trades: list) -> dict:
    """各平仓类型占比。"""
    cnt = Counter(t["type"] for t in trades)
    total = len(trades)
    return {k: round(v / total * 100, 1) for k, v in cnt.items()} if total else {}


def summary(equity: pd.Series, trades: list, benchmark: pd.Series | None = None) -> dict:
    wins   = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]

    res = {
        "sharpe":        sharpe(equity),
        "sortino":       sortino(equity),
        "omega":         omega_ratio(equity),
        "max_drawdown":  max_drawdown(equity),
        "calmar":        calmar(equity),
        "win_rate":      win_rate(trades),
        "profit_factor": profit_factor(trades),
        "payoff_ratio":  payoff_ratio(trades),
        "n_trades":      len(trades),
        "total_return":  float(equity.iloc[-1] / equity.iloc[0] - 1),
        "avg_hold_bars": float(np.mean([t.get("hold_bars", 0) for t in trades])) if trades else 0.0,
        "avg_win":       float(np.mean(wins)) if wins else 0.0,
        "avg_loss":      float(np.mean(losses)) if losses else 0.0,
        "max_consec_loss": max_consecutive_loss(trades),
        "trade_types":   trade_type_breakdown(trades),
    }

    if benchmark is not None:
        br = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)
        res["benchmark_return"] = br
        res["excess_return"]    = res["total_return"] - br
        # 信息比率
        port_ret = equity.pct_change().dropna()
        bm_ret   = benchmark.pct_change().dropna().reindex(port_ret.index).fillna(0)
        active   = port_ret - bm_ret
        res["information_ratio"] = (
            float(active.mean() / active.std() * np.sqrt(252 * BARS_PER_DAY))
            if active.std() > 0 else 0.0
        )

    return res
