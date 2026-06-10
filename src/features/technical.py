"""
src/features/technical.py — 技术指标特征（优化版，共16个）
============================================================
新增特征（相对旧版12个）：
  13. ADX14          — 趋势强度（区分震荡/趋势）
  14. KAMA           — 自适应移动平均（Kaufman）
  15. DI_diff        — +DI 与 -DI 之差（方向指示）
  16. Pivot_dist     — 与当日枢轴点（Pivot）距离
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.features.base import BaseFeature


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1).rolling(n).mean()


# ── 原有12个特征（保持不变）─────────────────────────────────────
class MA5Deviation(BaseFeature):
    name = "ma5_dev"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma = df["close"].rolling(5).mean().shift(1)
        return (df["close"].shift(1) - ma) / (ma + 1e-9)

class MA20Deviation(BaseFeature):
    name = "ma20_dev"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma = df["close"].rolling(20).mean().shift(1)
        return (df["close"].shift(1) - ma) / (ma + 1e-9)

class EMA12(BaseFeature):
    name = "ema12"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return _ema(df["close"], 12).shift(1)

class BollingerWidth(BaseFeature):
    name = "boll_width"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        return (2 * std / (mid + 1e-9)).shift(1)

class BollingerPosition(BaseFeature):
    name = "boll_pos"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        return ((df["close"] - (mid - 2 * std)) / (4 * std + 1e-9)).shift(1).clip(0, 1)

class RSI14(BaseFeature):
    name = "rsi14"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        d = df["close"].diff()
        g = d.clip(lower=0).rolling(14).mean()
        l = (-d.clip(upper=0)).rolling(14).mean()
        return (100 - 100 / (1 + g / (l + 1e-9))).shift(1) / 100

class MACDSignal(BaseFeature):
    name = "macd_signal"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        macd = _ema(df["close"], 12) - _ema(df["close"], 26)
        return ((macd - _ema(macd, 9)) / (df["close"] + 1e-9)).shift(1)

class ROC5(BaseFeature):
    name = "roc5"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(5).shift(1)

class ATR14Norm(BaseFeature):
    name = "atr14_norm"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return (_atr(df, 14) / (df["close"] + 1e-9)).shift(1)

class StochasticK(BaseFeature):
    name = "stoch_k"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        lo = df["low"].rolling(14).min()
        hi = df["high"].rolling(14).max()
        return ((df["close"] - lo) / (hi - lo + 1e-9)).shift(1)

class CCI20(BaseFeature):
    name = "cci20"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        tp  = (df["high"] + df["low"] + df["close"]) / 3
        ma  = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return (((tp - ma) / (0.015 * mad + 1e-9)).clip(-300, 300) / 300).shift(1)

class WilliamsR(BaseFeature):
    name = "williams_r"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        hi = df["high"].rolling(14).max()
        lo = df["low"].rolling(14).min()
        return ((hi - df["close"]) / (hi - lo + 1e-9)).shift(1)


# ── 新增4个特征 ──────────────────────────────────────────────────
class ADX14(BaseFeature):
    """平均方向指数（0~1 归一化），衡量趋势强度。"""
    name = "adx14"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        atr14 = _atr(df, 14)
        up   = df["high"].diff().clip(lower=0)
        down = (-df["low"].diff()).clip(lower=0)
        dm_pos = up.where(up > down, 0.0).rolling(14).mean()
        dm_neg = down.where(down > up, 0.0).rolling(14).mean()
        di_pos = dm_pos / (atr14 + 1e-9)
        di_neg = dm_neg / (atr14 + 1e-9)
        dx     = ((di_pos - di_neg).abs() / (di_pos + di_neg + 1e-9)) * 100
        adx    = dx.ewm(span=14, adjust=False).mean()
        return (adx / 100).clip(0, 1).shift(1)

class KAMA(BaseFeature):
    """
    Kaufman 自适应移动平均线，对 close 的偏离（归一化）。
    KAMA 跟踪趋势行情，震荡行情接近平均价格。
    """
    name = "kama_dev"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        n = 10
        result = close.copy()
        vals   = close.values.copy()
        res    = vals.copy()
        for i in range(n, len(vals)):
            direction = abs(vals[i] - vals[i - n])
            volatility = sum(abs(vals[j] - vals[j - 1]) for j in range(i - n + 1, i + 1))
            if volatility < 1e-9:
                res[i] = res[i - 1]
                continue
            er   = direction / volatility
            fast, slow = 2.0 / (2 + 1), 2.0 / (30 + 1)
            sc   = (er * (fast - slow) + slow) ** 2
            res[i] = res[i - 1] + sc * (vals[i] - res[i - 1])
        kama = pd.Series(res, index=close.index)
        return ((close - kama) / (kama + 1e-9)).shift(1).clip(-0.1, 0.1)

class DI_Diff(BaseFeature):
    """
    +DI 与 -DI 之差（归一化），正值代表上行趋势占主导。
    """
    name = "di_diff"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        atr14 = _atr(df, 14)
        up    = df["high"].diff().clip(lower=0)
        down  = (-df["low"].diff()).clip(lower=0)
        dm_pos = up.where(up > down, 0.0).rolling(14).mean()
        dm_neg = down.where(down > up, 0.0).rolling(14).mean()
        di_pos = dm_pos / (atr14 + 1e-9)
        di_neg = dm_neg / (atr14 + 1e-9)
        return ((di_pos - di_neg) / (di_pos + di_neg + 1e-9)).shift(1).clip(-1, 1)

class PivotDist(BaseFeature):
    """
    当前 close 与当日枢轴点（Pivot=(H+L+C)/3）之距离，归一化。
    枢轴点用前一根K线的高低收计算（无前瞻）。
    """
    name = "pivot_dist"
    def compute(self, df: pd.DataFrame) -> pd.Series:
        pivot = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
        return ((df["close"].shift(1) - pivot) / (pivot + 1e-9)).clip(-0.05, 0.05)


ALL_TECHNICAL = [
    MA5Deviation(), MA20Deviation(), EMA12(), BollingerWidth(),
    BollingerPosition(), RSI14(), MACDSignal(), ROC5(), ATR14Norm(),
    StochasticK(), CCI20(), WilliamsR(),
    # 新增
    ADX14(), KAMA(), DI_Diff(), PivotDist(),
]
