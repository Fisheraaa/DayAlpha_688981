"""
src/tdx/live_trader.py — 通达信模拟盘实时交易循环
===================================================
流程：
  1. 连接通达信行情
  2. 每根5分钟K线结束后，计算特征，调用集成模型预测
  3. 输出信号（仅记录日志 / 模拟下单，不接触真实资金）
  4. 记录信号到 signals.csv 供事后分析

警告：
  live_mode=false（默认）时只记录信号，不执行任何下单操作。
  需要接入券商 API 才能实现真实下单（超出本项目范围）。
"""
from __future__ import annotations
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.tdx.connector import TdxConnector, LiveBarBuffer
from src.features.dataset import build_feature_df, FEATURE_NAMES
from src.features.base import normalize_window
from src.data.processor import clean, mark_limit_bars
from src.utils import load_config, RESULTS

logger = logging.getLogger(__name__)

_SIGNAL_LOG = RESULTS / "live_signals.csv"


class LiveTrader:
    """
    实时交易循环控制器。

    使用方式：
        trader = LiveTrader(predictor)
        trader.run()          # 阻塞循环，按 Ctrl+C 退出
    """

    def __init__(self, predictor, cfg: dict | None = None):
        self._predictor = predictor
        self._cfg       = cfg or load_config()
        self._tdx_cfg   = self._cfg.get("tdx", {})
        self._symbol    = self._tdx_cfg.get("symbol", "sh688981")
        self._win       = self._cfg["features"]["window_size"]
        self._live      = self._tdx_cfg.get("live_mode", False)
        self._signals: list[dict] = []

    def run(self, poll_interval: float = 10.0) -> None:
        """
        实时循环。每 poll_interval 秒轮询一次行情。

        Args:
            poll_interval: 轮询间隔秒数（建议 10s，5分钟K线不需要太频繁）
        """
        logger.info("=== LiveTrader 启动 | symbol=%s | live_mode=%s ===",
                    self._symbol, self._live)

        with TdxConnector() as conn:
            buf = LiveBarBuffer(conn, symbol=self._symbol, capacity=self._win + 20)
            last_bar_time = None

            while True:
                try:
                    updated = buf.update()
                    if not updated or not buf.is_ready(self._win):
                        logger.debug("等待K线缓冲区预热... (%d/%d)", len(buf.get_df()), self._win)
                        time.sleep(poll_interval)
                        continue

                    df     = buf.get_df()
                    latest = df.index[-1]

                    if latest == last_bar_time:
                        time.sleep(poll_interval)
                        continue

                    # 新K线出现
                    last_bar_time = latest
                    self._on_new_bar(df)

                except KeyboardInterrupt:
                    logger.info("LiveTrader 停止（KeyboardInterrupt）")
                    self._save_signals()
                    break
                except Exception as e:
                    logger.error("LiveTrader 错误: %s", e, exc_info=True)
                    time.sleep(30)

    def _on_new_bar(self, df: pd.DataFrame) -> None:
        """处理新K线：特征计算 → 预测 → 信号记录。"""
        try:
            df_clean = clean(df)
            df_clean = mark_limit_bars(df_clean)

            if len(df_clean) < self._win:
                return

            feat_df = build_feature_df(df_clean)
            feat_df = feat_df.dropna()

            if len(feat_df) < self._win:
                return

            # 取最近 window_size 根K线的特征窗口
            window = feat_df.values[-self._win:].astype("float32")
            result = self._predictor.predict(window)

            ts  = df_clean.index[-1]
            sig = result["signal"]
            prob= result["prob"]

            rec = {
                "datetime":   str(ts),
                "signal":     sig,
                "prob":       round(prob, 4),
                "divergence": result.get("divergence", False),
                **{f"p_{k}": round(v, 4) for k, v in result.get("per_model", {}).items()},
            }
            self._signals.append(rec)

            signal_str = {1: "▲ 买入", -1: "▼ 卖出", 0: "─ 观望"}[sig]
            logger.info("[%s] %s  prob=%.3f  %s",
                        ts.strftime("%H:%M"), signal_str, prob,
                        "(分歧过滤)" if result.get("divergence") else "")

            if self._live and sig != 0:
                self._simulate_order(ts, sig, prob)

            # 每10条信号保存一次
            if len(self._signals) % 10 == 0:
                self._save_signals()

        except Exception as e:
            logger.error("_on_new_bar 错误: %s", e)

    def _simulate_order(self, ts, signal: int, prob: float) -> None:
        """模拟下单（仅日志，不接触真实资金）。"""
        action = "BUY" if signal == 1 else "SELL"
        logger.info("[模拟下单] %s %s @ prob=%.3f", ts, action, prob)
        # TODO: 接入券商 API (如 EasyTrader / XTP SDK)

    def _save_signals(self) -> None:
        if not self._signals:
            return
        pd.DataFrame(self._signals).to_csv(_SIGNAL_LOG, index=False, encoding="utf-8-sig")
        logger.info("信号已保存: %s (%d条)", _SIGNAL_LOG, len(self._signals))
