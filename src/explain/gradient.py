"""src/explain/gradient.py — Integrated Gradients + Transformer attention extraction."""
from __future__ import annotations
import logging, numpy as np, torch, torch.nn as nn, pandas as pd
from src.features.dataset import FEATURE_NAMES
logger=logging.getLogger(__name__)

def integrated_gradients(model, x, baseline=None, n_steps=50, device=None):
    dev=device or torch.device("cpu"); model.eval().to(dev); x=x.to(dev)
    if baseline is None: baseline=torch.zeros_like(x)
    baseline=baseline.to(dev)
    alphas=torch.linspace(0,1,n_steps+1,device=dev)
    interp=(baseline+alphas.view(-1,1,1,1)*(x-baseline)).squeeze(1)
    interp.requires_grad_(True)
    torch.sigmoid(model(interp)).sum().backward()
    avg_grads=interp.grad.detach().mean(dim=0)
    return (avg_grads*(x.squeeze(0)-baseline.squeeze(0))).cpu().numpy()

def batch_integrated_gradients(model, X, n_steps=50, device=None):
    dev=device or torch.device("cpu")
    attrs=[]
    for i in range(len(X)):
        attrs.append(integrated_gradients(model,X[i:i+1],n_steps=n_steps,device=dev))
        if (i+1)%50==0: logger.info("  IG: %d/%d",i+1,len(X))
    return np.stack(attrs)

def ig_global_importance(attrs, feature_names=None):
    names=feature_names or FEATURE_NAMES[:attrs.shape[2]]
    imp=np.abs(attrs).mean(axis=(0,1))
    return (pd.DataFrame({"feature":names,"ig_importance":imp})
              .sort_values("ig_importance",ascending=False).reset_index(drop=True))

class AttentionHook:
    def __init__(self, model):
        self.weights=[]; self._handles=[]
        for _,module in model.named_modules():
            if isinstance(module,nn.MultiheadAttention):
                self._handles.append(module.register_forward_hook(self._hook_fn))
    def _hook_fn(self,module,input,output):
        _,attn_w=output
        if attn_w is not None: self.weights.append(attn_w.detach().cpu())
    def get_weights(self): return torch.stack(self.weights,dim=0).numpy() if self.weights else None
    def clear(self): self.weights.clear()
    def remove(self):
        for h in self._handles: h.remove()
        self._handles.clear()

def extract_attention_weights(model, X, device=None):
    dev=device or torch.device("cpu"); model.eval().to(dev)
    hook=AttentionHook(model)
    with torch.no_grad(): _=model(X.to(dev))
    weights=hook.get_weights(); hook.remove()
    if weights is None: return None
    return weights.mean(axis=(0,1))

def attention_bar_importance(mean_attn):
    return mean_attn[-1,:] if mean_attn is not None else None
