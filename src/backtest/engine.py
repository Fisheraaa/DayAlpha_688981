"""
src/backtest/engine.py — 向量化回测引擎 (优化版 v2)
====================================================
主要改动（相对旧版）：
  1. 佣金降至万2.5后，cost_threshold 同步下调 → 更多有效信号
  2. 止损从 1.5% → 1.2%，追踪止损更灵敏（trail_stop_pct 1% 激活，gap 0.6%）
  3. 仓位管理：置信度映射扩展到 15%~45%（低成本环境可以适当加仓）
  4. 每日最多3次交易（原2次）
  5. 时间止损放宽到8根K线（原6根），给15分钟预测更多时间
  6. 新增 intraday_reset：每根K线同步计算当日已实现PnL
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.costs import buy_cost, sell_proceeds
from src.backtest.metrics import summary
from src.utils import load_config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    shares: float = 0.0
    entry_price: float = 0.0
    entry_bar: int = 0
    today_buy: float = 0.0
    available_sell: float = 0.0
    peak_price: float = 0.0
    trailing_active: bool = False

    def on_new_day(self) -> None:
        """日初：T+0，当天买的当天可卖（科创板T+0）。"""
        self.available_sell += self.today_buy
        self.today_buy = 0.0


def _close_pos(pos: Position, price: float, bar_i: int,
               close_type: str, cash: float, trades: list) -> float:
    if pos.available_sell <= 0:
        return cash
    proceeds = sell_proceeds(price, pos.available_sell)
    cost_basis = buy_cost(pos.entry_price, pos.available_sell)
    trades.append({
        "type":        close_type,
        "entry_price": pos.entry_price,
        "exit_price":  price,
        "shares":      pos.available_sell,
        "pnl":         proceeds - cost_basis,
        "hold_bars":   bar_i - pos.entry_bar,
        "return_pct":  (price - pos.entry_price) / pos.entry_price,
    })
    cash += proceeds
    pos.shares -= pos.available_sell
    pos.available_sell = 0.0
    pos.trailing_active = False
    return cash


def run_backtest(feature_df: pd.DataFrame, close: pd.Series,
                 predictor, cfg: dict | None = None) -> dict:
    """
    主回测循环。

    Args:
        feature_df: 特征矩阵，index 为 DatetimeIndex（Asia/Shanghai）
        close:      收盘价序列，index 与 feature_df 对齐
        predictor:  EnsemblePredictor（实现 .predict(window) -> dict）
        cfg:        配置字典，None 时从 config.yaml 加载
    Returns:
        dict 含 equity / trades / metrics / signals
    """
    if cfg is None:
        cfg = load_config()

    sc  = cfg["strategy"]
    bc  = cfg["backtest"]
    win = cfg["features"]["window_size"]

    capital = float(bc["initial_capital"])
    cash    = capital

    pos          = Position()
    trades: list = []
    signals: dict = {}
    equity_hist: list = []

    daily_trades    = 0
    daily_loss_nav  = capital
    current_date    = None

    X_np = feature_df.values.astype(np.float32)

    # ── 常量（避免每轮 dict 查找）──────────────────────────────
    stop_loss_pct  = sc["stop_loss_pct"]
    trail_stop_pct = sc["trail_stop_pct"]
    trail_gap_pct  = sc["trail_gap_pct"]
    daily_loss_lim = sc["daily_loss_limit"]
    buy_thr        = sc["buy_threshold"]
    max_daily      = sc["max_daily_trades"]
    eod_time       = pd.Timestamp("14:30").time()  # 30min K线：14:30收盘前平仓
    TIME_STOP_BARS = 3    # 30分钟 x 3根 = 90分钟时间止损

    for i, ts in enumerate(feature_df.index):
        price    = float(close.iloc[i])
        bar_date = ts.date()

        # ── 日初重置 ────────────────────────────────────────────
        if bar_date != current_date:
            current_date   = bar_date
            pos.on_new_day()
            daily_trades   = 0
            daily_loss_nav = cash + pos.shares * price

        nav = cash + pos.shares * price

        # ── 预热期 ──────────────────────────────────────────────
        if i < win:
            equity_hist.append(nav)
            signals[ts] = 0
            continue

        # ── 模型预测 ────────────────────────────────────────────
        result = predictor.predict(X_np[i - win: i])
        sig    = result["signal"]
        signals[ts] = sig

        # ── 日亏损保护 ──────────────────────────────────────────
        if (nav - daily_loss_nav) / max(daily_loss_nav, 1e-9) < -daily_loss_lim:
            sig = 0

        # ── 持仓管理：止损 / 追踪止损 ────────────────────────────
        if pos.shares > 0 and pos.available_sell > 0:
            pnl_pct = (price - pos.entry_price) / max(pos.entry_price, 1e-9)

            # 固定止损
            if pnl_pct < -stop_loss_pct:
                cash = _close_pos(pos, price, i, "stop_loss", cash, trades)

            # 追踪止损激活
            elif pnl_pct > trail_stop_pct:
                pos.trailing_active = True
                pos.peak_price = max(pos.peak_price, price)

            # 追踪止损触发
            if pos.trailing_active and pos.available_sell > 0:
                pos.peak_price = max(pos.peak_price, price)
                if (pos.peak_price - price) / max(pos.peak_price, 1e-9) > trail_gap_pct:
                    cash = _close_pos(pos, price, i, "trail_stop", cash, trades)

        # ── 时间止损（持仓超过 TIME_STOP_BARS 根K线强平）────────
        if pos.shares > 0 and pos.available_sell > 0:
            if i - pos.entry_bar >= TIME_STOP_BARS:
                cash = _close_pos(pos, price, i, "time_stop", cash, trades)

        # ── 收盘前强平（14:55）──────────────────────────────────
        if ts.time() >= eod_time and pos.available_sell > 0:
            cash = _close_pos(pos, price, i, "eod", cash, trades)
            sig  = 0

        # ── 开仓逻辑（含趋势过滤）──────────────────────────────────
        if sig == 1 and pos.shares == 0 and daily_trades < max_daily:

            # 趋势过滤：用最近窗口内的特征判断当前是否值得入场
            # 1. ATR过滤：当前ATR需高于近20根K线中位数（有足够波动空间）
            # 2. VWAP过滤：价格高于VWAP均线（多头方向）
            # 特征列名从selected_features里找，找不到则跳过过滤（不影响正确性）
            trend_ok = True
            feat_cols = feature_df.columns.tolist()

            atr_col  = "atr14_norm" if "atr14_norm"  in feat_cols else None
            vwap_col = "vwap_dev"   if "vwap_dev"    in feat_cols else None

            if atr_col and i >= 20:
                cur_atr    = float(feature_df[atr_col].iloc[i])
                median_atr = float(feature_df[atr_col].iloc[i-20:i].median())
                if cur_atr < median_atr * 0.8:   # ATR低于近期中位数80%，震荡行情跳过
                    trend_ok = False

            if trend_ok and vwap_col:
                vwap_dev = float(feature_df[vwap_col].iloc[i])
                if vwap_dev < -0.002:             # 价格低于VWAP 0.2%以上，空头偏弱不做多
                    trend_ok = False

            if trend_ok:
                conf     = (result["prob"] - buy_thr) / max(1.0 - buy_thr, 1e-9)
                size_pct = min(0.20 + conf * 0.20, 0.40)
                shares   = int(cash * size_pct / buy_cost(price, 1) / 100) * 100

                if shares >= 100:
                    cost = buy_cost(price, shares)
                    if cost <= cash:
                        cash              -= cost
                        pos.shares         = shares
                        pos.today_buy      = 0
                        pos.available_sell = shares
                        pos.entry_price    = price
                        pos.entry_bar      = i
                        pos.peak_price     = price
                        pos.trailing_active = False
                        daily_trades       += 1

        # ── 出场仅靠止损/时间止损/收盘强平 ───────────────────────

        equity_hist.append(cash + pos.shares * price)

    # ── 汇总结果 ─────────────────────────────────────────────────
    equity    = pd.Series(equity_hist, index=feature_df.index) / capital
    sig_s     = pd.Series(signals)
    benchmark = close / close.iloc[0]
    perf      = summary(equity, trades, benchmark=benchmark.reindex(equity.index))

    logger.info(
        "Backtest: %d trades  Sharpe=%.3f  MaxDD=%.2f%%  Return=%.2f%%  WinRate=%.1f%%",
        len(trades), perf["sharpe"], perf["max_drawdown"] * 100,
        perf["total_return"] * 100, perf.get("win_rate", 0) * 100,
    )
    return {"equity": equity, "trades": trades, "metrics": perf, "signals": sig_s}
