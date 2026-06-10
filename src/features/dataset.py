"""
src/features/dataset.py — 特征矩阵构建 + Dataset（优化版）
============================================================
改动：
  1. FEATURE_NAMES 随技术特征扩展（从27 → 31个）
  2. prediction_horizon 从1改为3（config中配置，更少噪音）
  3. cost_threshold 从0.003降至0.0018（config中配置，与万2.5佣金匹配）
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.processor import clean, mark_limit_bars
from src.features.base import BaseFeature, normalize_window
from src.features.technical import ALL_TECHNICAL
from src.features.volume import ALL_VOLUME, ALL_MICRO, ALL_TIME
from src.utils import load_config

logger = logging.getLogger(__name__)

ALL_FEATURES  = ALL_TECHNICAL + ALL_VOLUME + ALL_MICRO + ALL_TIME
FEATURE_NAMES = [f.name for f in ALL_FEATURES]   # 现在 31 个（16 technical + 8 vol + 4 micro + 3 time）


def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {feat.name: feat.compute(df) for feat in ALL_FEATURES}
    return pd.DataFrame(cols, index=df.index)


def build_label(close: pd.Series, high: pd.Series | None = None,
                low: pd.Series | None = None,
                threshold: float | None = None,
                horizon: int | None = None) -> pd.Series:
    """
    基于未来H根K线窗口内最大涨幅/最大跌幅构建标签（numpy向量化，无循环）。

    label = 1   未来H根K线最高价相对当前收盘涨幅 > threshold
    label = 0   未来H根K线最低价相对当前收盘跌幅 > threshold（且不满足涨条件）
    label = -1  模糊区间，训练跳过

    与T+0逻辑一致：持仓期间只要价格碰到目标价即可，不需要H根后还在高位。
    horizon=6（30分钟）时正样本率预计 ~55%，样本量充足。
    """
    cfg = load_config()
    thr = threshold if threshold is not None else cfg["features"]["cost_threshold"]
    H   = horizon   if horizon   is not None else cfg["features"].get("prediction_horizon", 6)

    if high is None:
        high = close
    if low is None:
        low = close

    c_arr  = close.values.astype(np.float64)
    h_arr  = high.values.astype(np.float64)
    l_arr  = low.values.astype(np.float64)
    n      = len(c_arr)
    lbl    = np.full(n, -1, dtype=int)

    # 向量化：对每个位置i，取i+1到i+H的max(high)和min(low)
    # 用累积rolling实现：先构建一个矩阵 (n, H)，再取max/min
    # 为节省内存，用stride_tricks
    from numpy.lib.stride_tricks import sliding_window_view
    if n > H:
        # future high: 每个位置i对应high[i+1:i+H+1]的最大值
        h_windows = sliding_window_view(h_arr, H)[1:]   # shape (n-H, H)，从位置1开始
        l_windows = sliding_window_view(l_arr, H)[1:]   # shape (n-H, H)

        fut_max_h = h_windows.max(axis=1)   # shape (n-H,)
        fut_min_l = l_windows.min(axis=1)   # shape (n-H,)

        c_ref  = c_arr[:n - H]
        up_pct   = (fut_max_h - c_ref) / np.where(c_ref > 0, c_ref, 1)
        down_pct = (c_ref - fut_min_l)  / np.where(c_ref > 0, c_ref, 1)

        lbl[:n - H] = np.where(up_pct > thr, 1,
                      np.where(down_pct > thr, 0, -1))

    lbl[-H:] = -1
    return pd.Series(lbl, index=close.index)


class T0Dataset(Dataset):
    def __init__(self, feature_df: pd.DataFrame, label: pd.Series,
                 limit_mask: pd.Series, window_size: int | None = None,
                 selected_features: list[str] | None = None):
        cfg = load_config()
        win = window_size or cfg["features"]["window_size"]

        if selected_features is not None:
            feature_df = feature_df[selected_features]

        X_raw = feature_df.values.astype(np.float32)
        y_raw = label.values
        lim   = limit_mask.values.astype(bool)

        xs, ys = [], []
        for end in range(win, len(X_raw)):
            lbl = y_raw[end]
            if lbl == -1 or lim[end]:
                continue
            w = X_raw[end - win: end]
            if np.isnan(w).any():
                continue
            xs.append(normalize_window(w))
            ys.append(int(lbl))

        self.X = torch.tensor(np.stack(xs), dtype=torch.float32)
        self.y = torch.tensor(ys, dtype=torch.long)
        self.n_features = self.X.shape[2] if self.X.ndim == 3 else feature_df.shape[1]

        pos_rate = 100 * self.y.float().mean().item()
        logger.info(
            "Dataset: %d samples, pos_rate=%.1f%%, features=%d, window=%d",
            len(self.y), pos_rate, self.n_features, win,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def prepare_dataset(raw_df: pd.DataFrame, use_feature_selection: bool = False):
    """
    完整的数据集准备流水线，带特征缓存（跳过每次10分钟的IC重算）。

    Returns:
        feature_df, label, limit_mask, selected_features
    """
    from src.utils import RAW
    import json

    cfg        = load_config()
    ic_thr     = cfg["features"]["ic_threshold"]
    horizon    = cfg["features"].get("prediction_horizon", 3)
    cost_thr   = cfg["features"]["cost_threshold"]

    # 缓存文件：参数变了自动失效
    cache_key  = f"features_ic{ic_thr}_h{horizon}_ct{cost_thr}"
    feat_cache = RAW / f"{cache_key}_features.parquet"
    meta_cache = RAW / f"{cache_key}_meta.json"

    if feat_cache.exists() and meta_cache.exists() and use_feature_selection:
        logger.info("特征缓存命中，跳过IC计算（删除 data/raw/features_*.parquet 可强制重算）")
        feature_df = pd.read_parquet(feat_cache)
        with open(meta_cache, encoding="utf-8") as f:
            meta = json.load(f)
        df         = clean(raw_df)
        df         = mark_limit_bars(df)
        label      = build_label(df["close"], high=df["high"], low=df["low"])
        limit_mask = df["is_limit_bar"]
        return feature_df, label, limit_mask, meta["selected"]

    df         = clean(raw_df)
    df         = mark_limit_bars(df)
    feature_df = build_feature_df(df)
    label      = build_label(df["close"], high=df["high"], low=df["low"])
    limit_mask = df["is_limit_bar"]

    if use_feature_selection:
        from src.features.selector import select_features
        from numpy.lib.stride_tricks import sliding_window_view
        h_arr = df["high"].values.astype(np.float64)
        c_arr = df["close"].values.astype(np.float64)
        if len(h_arr) > horizon:
            h_win     = sliding_window_view(h_arr, horizon)[1:]
            fut_max_h = h_win.max(axis=1)
            c_ref     = c_arr[:len(c_arr) - horizon]
            fut_ret   = (fut_max_h - c_ref) / np.where(c_ref > 0, c_ref, 1)
            future_max = pd.Series(np.append(fut_ret, [np.nan] * horizon), index=df.index)
        else:
            future_max = pd.Series(np.nan, index=df.index)
        selected, ic_summary = select_features(feature_df, future_max.dropna())
        logger.info("IC feature selection: %d / %d kept", len(selected), len(FEATURE_NAMES))

        # 写缓存
        feature_df.to_parquet(feat_cache)
        with open(meta_cache, "w", encoding="utf-8") as f:
            json.dump({"selected": selected}, f, ensure_ascii=False)
        logger.info("特征缓存已写入: %s", feat_cache.name)
    else:
        selected = FEATURE_NAMES

    return feature_df, label, limit_mask, selected
