"""src/features/volume.py — volume/flow (8) + microstructure (4) + time encoding (3)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.features.base import BaseFeature

class VWAPDeviation(BaseFeature):
    name = "vwap_dev"
    def compute(self, df):
        dk=df.index.normalize()
        vwap=df.groupby(dk)["amount"].cumsum()/(df.groupby(dk)["volume"].cumsum()+1e-9)
        return ((df["close"]-vwap)/(vwap+1e-9)).shift(1)
class VolumeRatio5(BaseFeature):
    name = "vol_ratio5"
    def compute(self, df):
        return (df["volume"]/(df["volume"].rolling(5).mean()+1e-9)).shift(1).clip(0,10)/10
class OBVNorm(BaseFeature):
    name = "obv_norm"
    def compute(self, df):
        obv=(np.sign(df["close"].diff())*df["volume"]).cumsum()
        mu=obv.rolling(20).mean(); sig=obv.rolling(20).std()
        return ((obv-mu)/(sig+1e-9)).shift(1).clip(-3,3)/3
class AmountShare(BaseFeature):
    name = "amount_share"
    def compute(self, df):
        dk=df.index.normalize(); cum=df.groupby(dk)["amount"].cumsum()
        return (df["amount"]/(cum.shift(1)+1e-9)).shift(1).clip(0,1)
class PriceVolCorr10(BaseFeature):
    name = "pv_corr10"
    def compute(self, df):
        return df["close"].pct_change().rolling(10).corr(df["volume"]).shift(1).clip(-1,1)
class TickImpact(BaseFeature):
    name = "tick_impact"
    def compute(self, df):
        imp=(df["high"]-df["low"])/(df["volume"]+1e-9)
        mu=imp.rolling(20).mean(); sig=imp.rolling(20).std()
        return ((imp-mu)/(sig+1e-9)).shift(1).clip(-3,3)/3
class ClosePosition(BaseFeature):
    name = "close_pos"
    def compute(self, df):
        return ((df["close"]-df["low"])/(df["high"]-df["low"]+1e-9)).shift(1)
class MFI14(BaseFeature):
    name = "mfi14"
    def compute(self, df):
        tp=(df["high"]+df["low"]+df["close"])/3; mf=tp*df["volume"]; pt=tp.shift(1)
        pos=mf.where(tp>pt,0).rolling(14).sum(); neg=mf.where(tp<pt,0).rolling(14).sum()
        return (100-100/(1+pos/(neg+1e-9))).shift(1)/100
class SpreadProxy(BaseFeature):
    name = "spread_proxy"
    def compute(self, df): return ((df["high"]-df["low"])/(df["close"]+1e-9)).shift(1)
class OrderFlowImbalance(BaseFeature):
    name = "ofi"
    def compute(self, df):
        return (2*(df["close"]-df["open"])/(df["high"]-df["low"]+1e-9)-1).shift(1)
class GapSize(BaseFeature):
    name = "gap_size"
    def compute(self, df):
        return ((df["open"]-df["close"].shift(1))/(df["close"].shift(1)+1e-9)).shift(1)
class OvernightGap(BaseFeature):
    name = "overnight_gap"
    def compute(self, df):
        gap=(df["open"]-df["close"].shift(1))/(df["close"].shift(1)+1e-9)
        return gap.where(df["is_first_bar"].astype(bool),0.0).shift(1)
class TimeEncodingSin(BaseFeature):
    name = "time_sin"
    def compute(self, df):
        idx=df.groupby(df.index.normalize()).cumcount()
        return np.sin(2*np.pi*idx/48)
class TimeEncodingCos(BaseFeature):
    name = "time_cos"
    def compute(self, df):
        idx=df.groupby(df.index.normalize()).cumcount()
        return np.cos(2*np.pi*idx/48)
class IsFirstBar(BaseFeature):
    name = "is_first_bar_feat"
    def compute(self, df): return df["is_first_bar"].astype(float)

ALL_VOLUME=[VWAPDeviation(),VolumeRatio5(),OBVNorm(),AmountShare(),
            PriceVolCorr10(),TickImpact(),ClosePosition(),MFI14()]
ALL_MICRO=[SpreadProxy(),OrderFlowImbalance(),GapSize(),OvernightGap()]
ALL_TIME=[TimeEncodingSin(),TimeEncodingCos(),IsFirstBar()]
