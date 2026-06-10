"""src/features/base.py — abstract base class and look-ahead validator."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

def normalize_window(window: np.ndarray) -> np.ndarray:
    """Per-feature MinMax normalization. Called identically in training and inference."""
    mn  = window.min(axis=0, keepdims=True)
    mx  = window.max(axis=0, keepdims=True)
    rng = mx - mn
    rng[rng == 0] = 1.0
    return (window - mn) / rng

class BaseFeature(ABC):
    """Contract: compute(df)[t] uses only df rows 0..t-1. Always apply .shift(1)."""
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series: ...

    def validate_no_lookahead(self, df, feature, **kwargs):
        """Asymmetric correlation test. look-ahead features have |corr| > 0.5."""
        ret = df["close"].pct_change()
        common = feature.dropna().index.intersection(ret.dropna().index)
        if len(common) < 50: return
        f, rc = feature[common].values, ret[common].values
        if f.std() < 1e-9 or rc.std() < 1e-9: return
        corr = float(np.corrcoef(f, rc)[0, 1])
        assert abs(corr) < 0.50, (
            f"Look-ahead bias in [{self.name}]: corr={corr:.4f} > 0.50. "
            "Check that compute() applies .shift(1).")
