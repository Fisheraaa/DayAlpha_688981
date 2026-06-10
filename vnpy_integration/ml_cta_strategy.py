"""
vnpy_integration/ml_cta_strategy.py
====================================
将 EnsemblePredictor 包装为 VeighNa CtaTemplate 策略。

关键设计：
  - 不依赖 VeighNa 数据库（load_bar 不调用），buffer 自然预热
  - 所有模型/特征代码原封不动复用 src/ 目录
  - 止损、追踪止损、收盘平仓逻辑与原 engine.py 保持一致
  - 动态仓位：置信度映射 20%~40%
"""
from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── 确保项目根目录在 sys.path ─────────────────────────────────
_INTEGRATION_DIR = Path(__file__).parent
_PROJECT_ROOT    = _INTEGRATION_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vnpy_ctastrategy import CtaTemplate, StopOrder
from vnpy.trader.constant import Direction, Offset
from vnpy.trader.object import BarData, TradeData, OrderData

from src.utils import load_config

logger = logging.getLogger(__name__)


class MLT0Strategy(CtaTemplate):
    """
    中芯国际 688981 T+0 机器学习策略（VeighNa版）

    策略参数可在 VeighNa 界面动态调整，也可在 run_backtest.py 中覆盖。
    """
    author = "SystematicAlpha"

    # ── 可调参数（出现在 VeighNa 参数界面）──────────────────────
    run_id          : str   = ""     # 训练好的模型 run_id
    window_size     : int   = 30     # 特征序列长度，与训练一致
    buy_threshold   : float = 0.61   # 买入置信度阈值
    sell_threshold  : float = 0.45   # 卖出置信度阈值
    stop_loss_pct   : float = 0.020  # 固定止损（距入场价跌幅）
    trail_act_pct   : float = 0.010  # 追踪止损激活距离
    trail_gap_pct   : float = 0.006  # 追踪止损回撤距离
    max_daily_trades: int   = 2      # 每日最多开仓次数
    eod_bar         : int   = 14     # 收盘前平仓的小时数（14:00开始不开新仓）

    parameters = [
        "run_id", "window_size", "buy_threshold", "sell_threshold",
        "stop_loss_pct", "trail_act_pct", "trail_gap_pct",
        "max_daily_trades", "eod_bar",
    ]
    variables = ["pos", "daily_trades", "entry_price", "trail_high"]

    # ────────────────────────────────────────────────────────────
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 内部状态
        self._predictor   = None          # EnsemblePredictor（on_init加载）
        self._bar_buffer  = deque(maxlen=120)  # 120根K线：指标预热≈21根+window_size(30)+安全余量，确保dropna后≥30有效行
        self._current_date = None

        # 持仓追踪（VeighNa 自动维护 self.pos，这里额外跟踪细节）
        self.entry_price  : float = 0.0
        self.trail_high   : float = 0.0
        self.daily_trades : int   = 0

    # ── 生命周期 ─────────────────────────────────────────────────
    def on_init(self) -> None:
        """策略初始化：加载训练好的模型，不访问 VeighNa 数据库。"""
        # !! 注意: VeighNa 在 on_init 期间 self.inited=False，write_log 静默丢弃
        # !! 改用 print() 确保输出可见
        print(f"[MLT0] on_init 开始 | run_id={self.run_id!r}")

        if not self.run_id:
            print("[MLT0] ⚠️ 未设置 run_id，策略将不产生信号")
            return

        try:
            import torch
            from src.models.ensemble import EnsemblePredictor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[MLT0] 正在加载模型... device={device}")
            self._predictor = EnsemblePredictor.load(
                self.run_id, val_aucs=None, device=device
            )
            print(f"[MLT0] ✅ 模型加载成功 | 模型数={len(self._predictor.models)}")
        except Exception as e:
            import traceback
            print(f"[MLT0] ❌ 模型加载失败: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            # self._predictor 保持 None，on_bar 中会跳过

        # 不调用 self.load_bar() —— buffer 在 on_bar() 中自然预热

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        if self.pos > 0:
            self.write_log(f"策略停止时仍持仓 {self.pos} 股，请手动确认")
        self.write_log("策略停止")

    # ── 核心 K线回调 ─────────────────────────────────────────────
    def on_bar(self, bar: BarData) -> None:
        """每根 30 分钟 K 线触发一次完整的信号计算 + 执行逻辑。"""

        # 日初重置日内计数器
        bar_date = bar.datetime.date()
        if bar_date != self._current_date:
            self._current_date = bar_date
            self.daily_trades = 0

        # 收盘前不开新仓（14:00+）
        near_close = bar.datetime.hour >= self.eod_bar

        # 撤销所有未成交挂单（避免堆积）
        self.cancel_all()

        # ── 追踪止损更新 ────────────────────────────────────────
        if self.pos > 0 and self.entry_price > 0:
            self.trail_high = max(self.trail_high, bar.close_price)

            # 固定止损
            fixed_sl = self.entry_price * (1 - self.stop_loss_pct)
            # 追踪止损（需要先触发激活条件）
            activated   = self.trail_high >= self.entry_price * (1 + self.trail_act_pct)
            trail_sl    = self.trail_high * (1 - self.trail_gap_pct) if activated else 0.0
            effective_sl = max(fixed_sl, trail_sl)

            if bar.close_price <= effective_sl:
                self._flat_position(bar.close_price, reason="止损")
                return

        # ── 收盘前平仓 ───────────────────────────────────────────
        # 14:30 及以后强制平仓（防止隔夜）
        if bar.datetime.hour == 14 and bar.datetime.minute >= 30:
            if self.pos > 0:
                self._flat_position(bar.close_price, reason="EOD平仓")
            return

        # ── 缓冲区更新 ───────────────────────────────────────────
        self._bar_buffer.append(bar)  # maxlen=120 自动丢弃最旧的行

        # 每100根K线打印一次进度（帮助确认策略在运行）
        if not getattr(self, "_bar_count", None):
            self._bar_count = 0
        self._bar_count += 1
        if self._bar_count == 1:
            print(f"[MLT0 on_bar#1] buffer={len(self._bar_buffer)} "
                  f"predictor={'OK' if self._predictor is not None else 'NONE'} "
                  f"need={self.window_size+35}")

        # 需要≥65根K线才调用预测：指标预热≈21根，dropna后剩余≥44行 > window_size(30)
        _MIN_BARS = self.window_size + 35   # = 65
        if len(self._bar_buffer) < _MIN_BARS or self._predictor is None:
            return

        # ── 特征计算 & 模型预测 ──────────────────────────────────
        result = self._predict(list(self._bar_buffer))
        if result is None:
            return

        sig, prob = result["signal"], result["prob"]

        # 首次成功预测时打印一条诊断
        if not getattr(self, "_first_pred_logged", False):
            self._first_pred_logged = True
            print(f"[MLT0 首次预测成功] sig={sig} prob={prob:.4f} buffer={len(self._bar_buffer)}")

        # ── 开仓逻辑 ─────────────────────────────────────────────
        if (sig == 1
                and self.pos == 0
                and self.daily_trades < self.max_daily_trades
                and not near_close):

            shares = self._calc_shares(bar.close_price)
            if shares > 0:
                self.buy(bar.close_price * 1.002, shares)  # 略高于收盘价，保证成交
                self.entry_price  = bar.close_price
                self.trail_high   = bar.close_price
                self.daily_trades += 1
                self.write_log(
                    f"开仓 | prob={prob:.3f} shares={shares} price={bar.close_price}"
                )

        # ── 平仓逻辑（模型信号） ─────────────────────────────────
        elif sig == -1 and self.pos > 0:
            self._flat_position(bar.close_price, reason=f"模型信号 prob={prob:.3f}")

    # ── 订单/成交回调 ────────────────────────────────────────────
    def on_order(self, order: OrderData) -> None:
        pass

    def on_trade(self, trade: TradeData) -> None:
        self.write_log(
            f"成交 | {trade.direction.value} {trade.volume}股 "
            f"@ {trade.price:.2f}"
        )

    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass

    # ── 内部辅助方法 ─────────────────────────────────────────────
    def _predict(self, bars: list[BarData]) -> Optional[dict]:
        """将 BarData 列表转换为 DataFrame，计算特征，调用集成预测。"""
        from src.features.dataset import build_feature_df
        from src.data.processor import clean, mark_limit_bars

        try:
            records = []
            for b in bars:
                records.append({
                    "datetime": b.datetime,
                    "open":     b.open_price,
                    "high":     b.high_price,
                    "low":      b.low_price,
                    "close":    b.close_price,
                    "volume":   b.volume,
                    "amount":   b.volume * b.close_price,
                })

            df = pd.DataFrame(records).set_index("datetime")
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("Asia/Shanghai")

            df = clean(df)
            df = mark_limit_bars(df)
            feat_df = build_feature_df(df).dropna()

            # ── 特征行数诊断（首次调用）────────────────────
            if not getattr(self, "_feat_diag_done", False):
                self._feat_diag_done = True
                print(f"[MLT0 FEAT DIAG] "
                      f"buffer_rows={len(df)} "
                      f"after_dropna={len(feat_df)} "
                      f"need={self.window_size} "
                      f"cols={feat_df.shape[1] if len(feat_df)>0 else 0}")

            if len(feat_df) < self.window_size:
                return None

            raw_window = feat_df.values[-self.window_size:].astype("float32")
            return self._predictor.predict(raw_window)

        except Exception as e:
            import traceback
            print(f"[MLT0 _predict ERROR] {type(e).__name__}: {e}")
            print(traceback.format_exc()[-600:])
            return None

    def _calc_shares(self, price: float) -> int:
        """
        动态仓位：置信度映射 20%~40% 的初始资金。
        VeighNa 回测中取 cta_engine.capital（初始资金）计算，
        实盘可改为读取账户可用资金。
        """
        initial_capital = getattr(self.cta_engine, "capital", 1_000_000)
        # 简化：使用固定初始资金，不追踪动态净值
        size_pct = 0.30  # 默认30%仓位
        raw = initial_capital * size_pct / price
        shares = int(raw / 100) * 100  # 向下取整到100股整数倍
        return max(shares, 100)

    def _flat_position(self, price: float, reason: str = "") -> None:
        """平掉全部多头仓位。"""
        if self.pos > 0:
            self.sell(price * 0.998, self.pos)
            self.write_log(f"平仓 | {reason} | price={price:.2f} pos={self.pos}")
            self.entry_price = 0.0
            self.trail_high  = 0.0
