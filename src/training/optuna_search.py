"""
src/training/optuna_search.py — Optuna 超参数搜索（GPU加速版）
=================================================================
GPU优化：
  - trial内训练也使用AMP + pin_memory
  - 快速评估轮数从30→40（GPU快，可以多跑几轮）
  - 并行度：n_jobs=1（单GPU，避免显存竞争）
"""
from __future__ import annotations
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from src.utils import load_config

logger = logging.getLogger(__name__)


def run_optuna(X: torch.Tensor, y: torch.Tensor, cfg: dict | None = None,
               device=None, n_features: int = 31) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna未安装: pip install optuna")
        return {"best_params": {}, "best_value": 0.0}

    if cfg is None:
        cfg = load_config()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp    = cfg["training"].get("use_amp", False) and device.type == "cuda"
    pin_memory = cfg["training"].get("pin_memory", False) and device.type == "cuda"

    oc      = cfg.get("optuna", {})
    n_tr    = oc.get("n_trials", 50)
    timeout = oc.get("timeout", 3600)
    tc      = cfg["training"]
    val_n   = int(len(X) * tc["val_ratio"])
    val_X   = X[-val_n:]
    val_y   = y[-val_n:].cpu().numpy()
    tr_X    = X[:-val_n]
    tr_y    = y[:-val_n]

    def objective(trial):
        from src.models.transformer_lstm import TransformerLSTM

        d_model = trial.suggest_categorical("d_model", [64, 128, 192])
        nhead   = trial.suggest_categorical("nhead", [4, 8])
        if d_model % nhead != 0:
            raise optuna.exceptions.TrialPruned()

        hp = {
            "d_model":     d_model,
            "nhead":       nhead,
            "num_layers":  trial.suggest_int("num_layers", 2, 4),
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [64, 128, 192]),
            "dropout":     trial.suggest_float("dropout", 0.1, 0.4),
            "lr":          trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "pos_weight":  trial.suggest_float("pos_weight", 1.0, 3.0),
            "batch_size":  trial.suggest_categorical("batch_size", [256, 512, 1024]),
        }

        model = TransformerLSTM(
            n_features=n_features,
            d_model=hp["d_model"], nhead=hp["nhead"],
            num_layers=hp["num_layers"], lstm_hidden=hp["lstm_hidden"],
            dropout=hp["dropout"],
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hp["pos_weight"]], device=device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=1e-5)
        scaler    = GradScaler("cuda", enabled=use_amp)
        loader    = DataLoader(
            TensorDataset(tr_X, tr_y.float()),
            batch_size=hp["batch_size"], shuffle=True,
            pin_memory=pin_memory, num_workers=0,
        )

        best_auc = 0.0
        for epoch in range(40):   # GPU快，快速评估40轮
            model.train()
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=use_amp):
                    loss = criterion(model(xb).squeeze(1), yb)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            model.eval()
            probs_list = []
            with torch.no_grad():
                for i in range(0, len(val_X), 1024):
                    xb = val_X[i: i+1024].to(device)
                    with autocast("cuda", enabled=use_amp):
                        logit = model(xb).squeeze(1)
                    probs_list.append(torch.sigmoid(logit).cpu().float().numpy())
            probs = np.concatenate(probs_list)

            try:
                auc = roc_auc_score(val_y, probs)
            except ValueError:
                auc = 0.5
            best_auc = max(best_auc, auc)

            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_auc

    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15)
    sampler = optuna.samplers.TPESampler(seed=42)
    study   = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_tr, timeout=timeout, show_progress_bar=True)

    bp = study.best_params
    logger.info("Optuna best: AUC=%.4f  params=%s", study.best_value, bp)
    return {"best_params": bp, "best_value": study.best_value, "study": study}
