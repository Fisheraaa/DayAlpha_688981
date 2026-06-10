"""src/features/selector.py — IC-based automated feature selection."""
from __future__ import annotations
import logging
import numpy as np, pandas as pd
from scipy import stats
from src.utils import load_config
logger = logging.getLogger(__name__)

def compute_ic(feature_df, forward_ret, window=None):
    cfg = load_config(); win = window or cfg["features"]["ic_window"]
    n = len(feature_df); ic_records = []
    for end in range(win, n):
        sl=slice(end-win,end); f=feature_df.iloc[sl]; r=forward_ret.iloc[sl]
        mask=f.notna().all(axis=1)&r.notna(); f,r=f[mask],r[mask]
        if len(f)<win//2: continue
        row={col: (stats.spearmanr(f[col].values,r.values)[0] or 0.0) for col in feature_df.columns}
        ic_records.append(row)
    return pd.DataFrame(ic_records, columns=feature_df.columns)

def select_features(feature_df, forward_ret, ic_threshold=None, window=None, verbose=True):
    cfg=load_config(); threshold=ic_threshold or cfg["features"]["ic_threshold"]
    ic_df=compute_ic(feature_df,forward_ret,window=window)
    rows=[]
    for col in ic_df.columns:
        vals=ic_df[col].dropna(); mu=float(vals.mean()); std=float(vals.std())+1e-9
        rows.append({"feature":col,"ic_mean":round(mu,5),"ic_std":round(std,5),
                     "ic_ir":round(mu/std,4),"ic_pos_pct":round(float((vals>0).mean())*100,1),
                     "selected":abs(mu)>=threshold})
    ic_summary=(pd.DataFrame(rows).sort_values("ic_mean",key=abs,ascending=False).reset_index(drop=True))
    selected=ic_summary.loc[ic_summary["selected"],"feature"].tolist()
    dropped=ic_summary.loc[~ic_summary["selected"],"feature"].tolist()
    if verbose: logger.info("Feature selection (|IC|>=%.4f): keep %d, drop %d: %s",threshold,len(selected),len(dropped),dropped)
    return selected, ic_summary

def filter_feature_df(df, selected):
    missing=[f for f in selected if f not in df.columns]
    if missing: raise ValueError(f"Features missing from matrix: {missing}")
    return df[selected]
