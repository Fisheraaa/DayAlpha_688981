"""src/explain/shap_analysis.py — SHAP attribution, visualization, and feature stability."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from src.features.dataset import FEATURE_NAMES

logger = logging.getLogger(__name__)


def compute_shap_values(model, X_background, X_explain, device=None):
    """
    计算 SHAP 值。

    优先使用 GradientExplainer（对 LSTM/LayerNorm 等复杂模块更健壮）。
    DeepExplainer 不支持 LSTM/LayerNorm 内部操作，会因加和性校验失败报错，
    GradientExplainer 直接走反向传播梯度，与 Integrated Gradients 原理相近，
    适合本项目的 Transformer-LSTM 结构。

    注意：调用前建议关闭 cuDNN（torch.backends.cudnn.enabled = False），
    否则 cuDNN LSTM 在 eval 模式下不支持 backward。
    """
    if not HAS_SHAP:
        logger.warning("shap not installed: pip install shap")
        return np.zeros((len(X_explain), X_explain.shape[1], X_explain.shape[2]))

    dev = device or torch.device("cpu")
    model.eval().to(dev)

    X_bg  = X_background.to(dev)
    X_exp = X_explain.to(dev)

    # ── 优先：GradientExplainer（兼容 LSTM/LayerNorm）────────────
    try:
        explainer = shap.GradientExplainer(model, X_bg)
        vals      = explainer.shap_values(X_exp)
        result    = vals[0] if isinstance(vals, list) else vals
        result    = np.array(result)
        # GradientExplainer 对单输出模型会多一个尾部维度 (..., 1)
        if result.ndim == 4 and result.shape[-1] == 1:
            result = result[..., 0]   # (n, seq, features, 1) → (n, seq, features)
        logger.info("SHAP (GradientExplainer) computed: shape=%s", result.shape)
        return result

    except Exception as e_grad:
        logger.warning("GradientExplainer 失败 (%s)，尝试 DeepExplainer...", e_grad)

    # ── 备用：DeepExplainer（跳过加和性校验）────────────────────
    try:
        explainer = shap.DeepExplainer(model, X_bg)
        # 部分版本支持 check_additivity=False
        try:
            vals = explainer.shap_values(X_exp, check_additivity=False)
        except TypeError:
            vals = explainer.shap_values(X_exp)
        result = vals[0] if isinstance(vals, list) else vals
        logger.info("SHAP (DeepExplainer) computed: shape=%s", np.array(result).shape)
        return np.array(result)

    except Exception as e_deep:
        logger.warning("DeepExplainer 也失败 (%s)，返回零矩阵", e_deep)
        return np.zeros((len(X_explain), X_explain.shape[1], X_explain.shape[2]))


def global_importance(shap_values, feature_names=None):
    names = feature_names or FEATURE_NAMES[:shap_values.shape[2]]
    imp   = np.abs(shap_values).mean(axis=(0, 1))
    return (pd.DataFrame({"feature": names, "shap_importance": imp})
              .sort_values("shap_importance", ascending=False)
              .reset_index(drop=True))


def plot_global_importance(imp_df, save_path=None, top_n=20):
    if not HAS_MPL:
        return
    df  = imp_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(df["feature"][::-1], df["shap_importance"][::-1],
                   color="#2563EB", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(f"SHAP Global Feature Importance (Top {top_n})", fontsize=14, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, df["shap_importance"][::-1]):
        ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP importance plot saved: %s", save_path)
    plt.close(fig)


def plot_temporal_heatmap(shap_values, feature_names=None, save_path=None, top_n_features=15):
    if not HAS_MPL:
        return
    names = feature_names or FEATURE_NAMES[:shap_values.shape[2]]
    heat  = np.abs(shap_values).mean(axis=0)
    top_i = np.argsort(heat.mean(axis=0))[::-1][:top_n_features]
    heat_top  = heat[:, top_i].T
    names_top = [names[i] for i in top_i]
    fig, ax = plt.subplots(figsize=(14, max(5, top_n_features * 0.4)))
    im = ax.imshow(heat_top, aspect="auto", cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean |SHAP value|")
    ax.set_yticks(range(len(names_top)))
    ax.set_yticklabels(names_top, fontsize=9)
    ax.set_xlabel("Bar index in window (0=oldest, 59=latest)", fontsize=11)
    ax.set_title("SHAP Temporal Heatmap", fontsize=13, pad=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP heatmap saved: %s", save_path)
    plt.close(fig)


def plot_attention_weights(attn_weights, save_path=None):
    if not HAS_MPL or attn_weights is None:
        return
    seq = attn_weights.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_weights, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention weight")
    ax.set_xlabel("Key bar position", fontsize=11)
    ax.set_ylabel("Query bar position", fontsize=11)
    ax.set_title("Transformer Attention Weights (averaged)", fontsize=13, pad=12)
    ax.axhline(seq - 1, color="white", lw=1, ls="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Attention plot saved: %s", save_path)
    plt.close(fig)


def feature_stability_analysis(model, X, y, n_windows=5, device=None):
    """
    Time-slice permutation importance.
    stability_score = importance_mean / importance_std  (higher = more stable).
    """
    from sklearn.metrics import roc_auc_score
    dev = device or torch.device("cpu")
    model.eval().to(dev)
    rng, window_imps = np.random.default_rng(42), []

    for w in range(n_windows):
        win_size = len(X) // n_windows
        Xw = X[w * win_size: (w + 1) * win_size].to(dev)
        yw = y[w * win_size: (w + 1) * win_size].numpy()
        if len(np.unique(yw)) < 2:
            continue
        with torch.no_grad():
            base_probs = torch.sigmoid(model(Xw).squeeze(1)).cpu().numpy()
        base_auc, feat_drops = roc_auc_score(yw, base_probs), []
        for fi in range(X.shape[2]):
            Xp = Xw.clone()
            Xp[:, :, fi] = Xp[rng.permutation(len(Xp)), :, fi]
            with torch.no_grad():
                p = torch.sigmoid(model(Xp).squeeze(1)).cpu().numpy()
            try:    feat_drops.append(base_auc - roc_auc_score(yw, p))
            except: feat_drops.append(0.0)
        window_imps.append(np.array(feat_drops))

    if not window_imps:
        return pd.DataFrame()
    arr   = np.stack(window_imps)
    names = FEATURE_NAMES[:arr.shape[1]]
    df = pd.DataFrame({
        "feature":         names,
        "importance_mean": arr.mean(axis=0),
        "importance_std":  arr.std(axis=0),
        "stability_score": arr.mean(axis=0) / (arr.std(axis=0) + 1e-9),
    }).sort_values("stability_score", ascending=False).reset_index(drop=True)
    df["stability_rank"] = range(1, len(df) + 1)
    logger.info("Feature stability done. Top: %s (score=%.3f)",
                df.iloc[0]["feature"], df.iloc[0]["stability_score"])
    return df


def permutation_importance(model, X, y, n_repeats=3, device=None, feature_names=None):
    from sklearn.metrics import roc_auc_score
    dev   = device or torch.device("cpu")
    names = feature_names or FEATURE_NAMES[:X.shape[2]]
    model.eval().to(dev)
    rng   = np.random.default_rng(42)
    with torch.no_grad():
        base_probs = torch.sigmoid(model(X.to(dev)).squeeze(1)).cpu().numpy()
    base_auc = roc_auc_score(y.numpy(), base_probs)
    rows = []
    for fi, name in enumerate(names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.clone()
            Xp[:, :, fi] = Xp[rng.permutation(len(Xp)), :, fi]
            with torch.no_grad():
                p = torch.sigmoid(model(Xp.to(dev)).squeeze(1)).cpu().numpy()
            try:    drops.append(base_auc - roc_auc_score(y.numpy(), p))
            except: drops.append(0.0)
        rows.append({"feature": name, "drop_mean": float(np.mean(drops)),
                     "drop_std": float(np.std(drops))})
    return (pd.DataFrame(rows).sort_values("drop_mean", ascending=False).reset_index(drop=True))
