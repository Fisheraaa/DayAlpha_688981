"""src/backtest/walk_forward.py — Walk-Forward OOS validation（带断点续传）"""
from __future__ import annotations
import json
import logging
import numpy as np
import pandas as pd
import torch

from src.features.dataset import T0Dataset, prepare_dataset, FEATURE_NAMES
from src.models.ensemble import build_all_models, EnsemblePredictor
from src.training.trainer import train_one_model
from src.training.optuna_search import run_optuna
from src.backtest.engine import run_backtest
from src.utils import load_config, RESULTS

logger = logging.getLogger(__name__)

# 断点续传文件路径
_CKPT_DIR  = RESULTS / "wf_checkpoint"
_CKPT_FILE = _CKPT_DIR / "progress.json"


def _save_checkpoint(wi: int, results: list, cfg_hash: str) -> None:
    """保存当前进度到checkpoint文件。"""
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # equity是pd.Series不能直接JSON序列化，只保存metrics
    serializable = []
    for r in results:
        serializable.append({
            "window":  r["window"],
            "metrics": {k: (v if not isinstance(v, dict) else str(v))
                        for k, v in r["metrics"].items()},
        })
    with open(_CKPT_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_completed_window": wi, "results": serializable,
                   "cfg_hash": cfg_hash}, f, indent=2)
    logger.info("Checkpoint saved: window %d completed", wi + 1)


def _load_checkpoint(cfg_hash: str) -> tuple[int, list]:
    """
    读取checkpoint。
    返回 (last_completed_window_index, partial_results)。
    如果没有checkpoint或cfg变了则返回 (-1, [])。
    """
    if not _CKPT_FILE.exists():
        return -1, []
    try:
        with open(_CKPT_FILE, encoding="utf-8") as f:
            ckpt = json.load(f)
        if ckpt.get("cfg_hash") != cfg_hash:
            logger.info("Config changed, ignoring old checkpoint")
            return -1, []
        last = ckpt["last_completed_window"]
        logger.info("Checkpoint found: resuming from window %d", last + 2)
        return last, ckpt["results"]
    except Exception as e:
        logger.warning("Failed to load checkpoint: %s", e)
        return -1, []


def _clear_checkpoint() -> None:
    if _CKPT_FILE.exists():
        _CKPT_FILE.unlink()


def _cfg_hash(cfg: dict) -> str:
    """生成config的简单hash，用于判断参数是否变化。"""
    import hashlib, json as _json
    relevant = {
        "data":     cfg.get("data", {}),
        "features": cfg.get("features", {}),
        "walk_forward": cfg.get("walk_forward", {}),
    }
    return hashlib.md5(_json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:8]


def run_walk_forward(raw_df, cfg=None, use_optuna=False,
                     use_feature_selection=False, device=None) -> dict:
    """
    Walk-Forward OOS validation（带断点续传）。

    中途关机后重新运行会自动从上次完成的窗口之后继续，不重跑已完成的窗口。
    config参数变化时自动放弃旧checkpoint重头开始。
    """
    if cfg is None: cfg = load_config()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wf        = cfg["walk_forward"]
    oos_start = pd.Timestamp(cfg["data"]["oos_start"], tz="Asia/Shanghai")
    oos_end   = pd.Timestamp(cfg["data"]["oos_end"],   tz="Asia/Shanghai")

    feature_df, label, limit_mask, selected = prepare_dataset(
        raw_df, use_feature_selection=use_feature_selection)
    close   = raw_df["close"].reindex(feature_df.index)
    n_feats = len(selected)
    logger.info("Walk-Forward using %d features", n_feats)

    # ── 构建窗口列表 ─────────────────────────────────────────────────
    windows, test_start = [], oos_start
    while test_start + pd.DateOffset(months=wf["test_months"]) <= oos_end:
        windows.append({
            "train_start": test_start - pd.DateOffset(months=wf["train_months"]),
            "train_end":   test_start - pd.Timedelta(days=1),
            "test_start":  test_start,
            "test_end":    test_start + pd.DateOffset(months=wf["test_months"]),
        })
        test_start += pd.DateOffset(months=wf["step_months"])

    logger.info("Walk-Forward: %d windows", len(windows))

    # ── 断点续传：读取上次进度 ───────────────────────────────────────
    ch = _cfg_hash(cfg)
    last_done, partial_results = _load_checkpoint(ch)
    start_wi = last_done + 1   # 从下一个未完成的窗口开始

    if start_wi > 0:
        logger.info("Resuming from window %d/%d (skipping %d completed)",
                    start_wi + 1, len(windows), start_wi)

    # equity_pieces只保存本次运行的，用于最终拼接
    results       = partial_results.copy()   # 含历史metrics（无equity）
    equity_pieces = []                        # 只含本次运行的equity序列

    for wi in range(start_wi, len(windows)):
        w = windows[wi]
        logger.info("Window %d/%d  train %s~%s  test %s~%s",
                    wi+1, len(windows),
                    w["train_start"].date(), w["train_end"].date(),
                    w["test_start"].date(),  w["test_end"].date())

        tr_mask  = (feature_df.index >= w["train_start"]) & (feature_df.index <= w["train_end"])
        oos_mask = (feature_df.index >= w["test_start"])  & (feature_df.index <= w["test_end"])
        if tr_mask.sum() < 500 or oos_mask.sum() < 50:
            logger.warning("Window %d: insufficient data, skipping", wi + 1)
            continue

        ds = T0Dataset(feature_df[tr_mask], label[tr_mask], limit_mask[tr_mask],
                       selected_features=selected)
        if len(ds) < 100:
            continue
        X_all, y_all = ds.X, ds.y

        models = build_all_models(cfg, n_features=n_feats)
        if use_optuna:
            opt = run_optuna(X_all, y_all, cfg=cfg, device=device, n_features=n_feats)
            bp  = opt["best_params"]
            cfg["model"].update({
                "d_model":     bp.get("d_model", 64),
                "nhead":       bp.get("nhead", 4),
                "num_transformer_layers": bp.get("num_layers", 2),
                "lstm_hidden": bp.get("lstm_hidden", 64),
                "dropout":     bp.get("dropout", 0.3),
            })
            cfg["training"].update({
                "lr":         bp.get("lr", 0.001),
                "pos_weight": bp.get("pos_weight", 1.5),
                "batch_size": bp.get("batch_size", 128),
            })
            models = build_all_models(cfg, n_features=n_feats)

        val_aucs = {}
        for name, model in models.items():
            trained, auc = train_one_model(model, X_all, y_all, cfg=cfg, device=device)
            models[name]  = trained
            val_aucs[name] = auc
            logger.info("  [%s] AUC=%.4f", name, auc)

        predictor = EnsemblePredictor(models, val_aucs, device=device,
                                      selected_features=selected)
        bt = run_backtest(feature_df[oos_mask], close[oos_mask], predictor, cfg=cfg)

        results.append({"window": wi+1, "metrics": bt["metrics"], "equity": bt["equity"]})
        equity_pieces.append(bt["equity"])

        # ── 每个窗口完成后保存checkpoint ─────────────────────────────
        _save_checkpoint(wi, results, ch)

    if not results:
        return {"window_results": [], "summary": {}, "all_equity": pd.Series(dtype=float)}

    keys = ["sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "total_return"]
    summ = {}
    for k in keys:
        vals = [r["metrics"].get(k, np.nan) for r in results]
        summ[f"{k}_mean"] = float(np.nanmean(vals))
        summ[f"{k}_std"]  = float(np.nanstd(vals))

    logger.info("Walk-Forward summary: Sharpe %.3f+/-%.3f  MaxDD %.2f%%+/-%.2f%%",
                summ["sharpe_mean"], summ["sharpe_std"],
                summ["max_drawdown_mean"]*100, summ["max_drawdown_std"]*100)

    # ── equity拼接（只用本次运行的部分）─────────────────────────────
    if equity_pieces:
        combined = equity_pieces[0]
        for p in equity_pieces[1:]:
            combined = pd.concat([combined, p * combined.iloc[-1]])
        combined = combined[~combined.index.duplicated(keep="first")]
    else:
        combined = pd.Series(dtype=float)

    # 全部完成，清除checkpoint
    _clear_checkpoint()
    logger.info("Walk-Forward complete. Checkpoint cleared.")

    return {"window_results": results, "summary": summ, "all_equity": combined}
