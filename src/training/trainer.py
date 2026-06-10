"""
src/training/trainer.py — 单模型训练器（GPU加速版）
=====================================================
GPU优化项：
  1. AMP 混合精度（torch.cuda.amp）：显存减半，RTX系列速度提升30~50%
  2. pin_memory=True：CPU→GPU数据传输走锁页内存，减少等待
  3. num_workers=4：DataLoader多进程并行预加载，GPU不空等
  4. optimizer.zero_grad(set_to_none=True)：比置零快，减少内存写操作
  5. 验证时分批推理（避免大val集一次性OOM）
  6. epoch日志从每20轮改为每10轮，方便观察训练曲线
"""
from __future__ import annotations
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import roc_auc_score

from src.utils import load_config

logger = logging.getLogger(__name__)


def _val_predict(model: nn.Module, val_X: torch.Tensor,
                 device, use_amp: bool, batch: int = 1024) -> np.ndarray:
    """分批验证推理，防止大val集OOM。"""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(val_X), batch):
            xb = val_X[i: i + batch].to(device)
            with autocast("cuda", enabled=use_amp):
                logit = model(xb).squeeze(1)
            probs.append(torch.sigmoid(logit).cpu().float().numpy())
    return np.concatenate(probs)


def train_one_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                    cfg: dict | None = None, device=None) -> tuple[nn.Module, float]:
    """
    训练单个模型（GPU加速版）。

    Args:
        model:  PyTorch 模型
        X:      (N, window, features)  float32
        y:      (N,)                   long
        cfg:    配置字典
        device: 训练设备（None→自动选CUDA）

    Returns:
        (trained_model, best_val_auc)
    """
    if cfg is None:
        cfg = load_config()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tc          = cfg["training"]
    lr          = tc["lr"]
    batch_size  = tc["batch_size"]
    max_epochs  = tc["max_epochs"]
    patience    = tc["patience"]
    pos_weight  = tc["pos_weight"]
    val_ratio   = tc["val_ratio"]
    grad_clip   = tc.get("grad_clip", 1.0)
    use_amp     = tc.get("use_amp", False) and device.type == "cuda"
    num_workers = tc.get("num_workers", 0) if device.type == "cuda" else 0
    pin_memory  = tc.get("pin_memory", False) and device.type == "cuda"

    model = model.to(device)

    # ── 数据集分割（时序：前段训练，后段验证）────────────────────
    n     = len(X)
    split = int(n * (1 - val_ratio))
    ds    = TensorDataset(X, y)
    tr_loader = DataLoader(
        Subset(ds, range(split)),
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        persistent_workers = (num_workers > 0),
    )
    val_X = X[split:]          # 保留在CPU，分批推理时再搬GPU
    val_y = y[split:].cpu().numpy()

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=lr * 0.01
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    if use_amp:
        logger.info("  AMP混合精度已启用（CUDA）")

    best_auc   = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # ── 训练 ────────────────────────────────────────────────
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.float().to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)          # ★ 比 zero_grad() 快
            with autocast("cuda", enabled=use_amp):
                loss = criterion(model(xb).squeeze(1), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # ── 验证 ────────────────────────────────────────────────
        probs = _val_predict(model, val_X, device, use_amp)
        try:
            auc = roc_auc_score(val_y, probs)
        except ValueError:
            auc = 0.5

        if auc > best_auc + 1e-4:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            logger.info("    Epoch %3d/%d  AUC=%.4f  best=%.4f  no_improve=%d",
                        epoch, max_epochs, auc, best_auc, no_improve)

        if no_improve >= patience:
            logger.info("    Early stop @ epoch %d  (best AUC=%.4f)", epoch, best_auc)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().to(device)
    return model, best_auc
