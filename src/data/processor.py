"""src/data/processor.py — data cleaning and quality flags."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

def clean(df):
    n0 = len(df)
    df = df.copy()
    df = df.dropna(subset=["open","high","low","close","volume"])
    df = df[df["volume"] > 0]
    df = df[~((df["high"] == df["low"]) & (df["high"] == df["open"]))]
    df = df[df["close"] > 0]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    dates = df.index.normalize()
    df["is_first_bar"] = (~dates.duplicated(keep="first")).astype(int)
    df["is_last_bar"]  = (~dates.duplicated(keep="last")).astype(int)
    logger.info("Clean: %d -> %d rows", n0, len(df))
    return df

def mark_limit_bars(df, limit_pct=0.195):
    """Flag STAR-market +/-20% limit bars."""
    df = df.copy()
    chg = (df["close"] - df["close"].shift(1)) / df["close"].shift(1)
    df["is_limit_bar"] = ((chg >= limit_pct) | (chg <= -limit_pct)).astype(int)
    return df
