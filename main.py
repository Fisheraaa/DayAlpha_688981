"""
main.py — 688981 T+0 策略主入口（GPU加速版）
=============================================
用法：
  python main.py --mode full            # 完整训练+回测
  python main.py --mode walk_forward    # Walk-Forward验证
  python main.py --mode walk_forward --optuna  # +Optuna超参搜索
  python main.py --mode live --run_id <id>     # 通达信模拟盘
  python main.py --check_gpu                   # 只检查GPU信息后退出
"""
from __future__ import annotations
import argparse
import logging
import sys
import torch

from src.utils import load_config, setup_logging, RESULTS
from src.backtest.costs import cost_summary


# ── GPU诊断 ──────────────────────────────────────────────────────
def _check_and_print_gpu() -> torch.device:
    """打印GPU信息，返回最优device。"""
    print("\n🖥️  设备检测:")
    if not torch.cuda.is_available():
        print("  ⚠️  未检测到CUDA，将使用CPU训练（速度较慢）")
        print("     若您有NVIDIA独显，请确认已安装CUDA版PyTorch：")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return torch.device("cpu")

    idx  = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    mem  = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    print(f"  ✅ GPU: {name}")
    print(f"  💾 显存: {mem:.1f} GB")

    # 自动根据显存调整batch_size建议
    if mem >= 10:
        bs_tip = 512
    elif mem >= 6:
        bs_tip = 256
    else:
        bs_tip = 128
    print(f"  📐 建议batch_size: {bs_tip}（当前config: {load_config()['training']['batch_size']}）")

    # 检查AMP支持
    cc = torch.cuda.get_device_properties(idx).major
    if cc >= 7:
        print(f"  ⚡ 支持AMP混合精度（Compute Capability {cc}.x，Tensor Core加速）")
    else:
        print(f"  ℹ️  Compute Capability {cc}.x，AMP可用但无Tensor Core加速")

    # 检查torch.compile支持（PyTorch 2.0+）
    if hasattr(torch, "compile"):
        print("  🚀 torch.compile 可用（PyTorch 2.0+）")
    print()
    return torch.device("cuda")


def _print_cost_summary():
    cs = cost_summary()
    print("📊 交易成本明细:")
    print(f"  佣金:      {cs['commission_bp']:.2f} bp  (万{cs['commission_bp']/10:.1f})")
    print(f"  印花税:    {cs['stamp_duty_bp']:.2f} bp（仅卖出）")
    print(f"  经手费:    {cs['exchange_fee_bp']:.3f} bp")
    print(f"  滑点:      {cs['slippage_bp']:.2f} bp")
    print(f"  往返合计:  {cs['round_trip_bp']:.2f} bp ≈ {cs['round_trip_bp']/100:.4f}%")
    print(f"  最低盈利:  {cs['min_profit_bp']:.2f} bp")
    print()


# ── 训练辅助：可选 torch.compile ─────────────────────────────────
def _maybe_compile(model, device):
    """
    PyTorch 2.0+ 在CUDA上尝试编译模型，可额外提速10~30%。
    Windows上Triton不可用，自动跳过（不影响正确性）。
    """
    import platform
    if platform.system() == "Windows":
        return model   # Windows无Triton，torch.compile不可用
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logging.getLogger("main").info("  torch.compile 已启用（reduce-overhead模式）")
        except Exception:
            pass
    return model


# ── 模式：full ────────────────────────────────────────────────────
def mode_full(cfg, device, run_id: str = ""):
    from src.data.fetcher import fetch_bars
    from src.features.dataset import prepare_dataset, T0Dataset
    from src.models.ensemble import build_all_models, EnsemblePredictor
    from src.backtest.engine import run_backtest
    import pandas as pd, json, time

    logger = logging.getLogger("main.full")

    logger.info("Step 1: 获取数据...")
    raw_df = fetch_bars()

    logger.info("Step 2: 特征工程（IC特征选择启用）...")
    feature_df, label, limit_mask, selected = prepare_dataset(raw_df, use_feature_selection=True)
    logger.info("  选中特征 %d 个: %s", len(selected), selected)

    oos_start = pd.Timestamp(cfg["data"]["oos_start"], tz="Asia/Shanghai")
    oos_mask  = feature_df.index >= oos_start
    n_feats   = len(selected)

    # ── 加载已有模型 or 重新训练 ─────────────────────────────────
    if run_id:
        logger.info("Step 3: 加载已有模型 run_id=%s（跳过训练）...", run_id)
        predictor  = EnsemblePredictor.load(run_id, val_aucs=None, device=device)
        new_run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") + "_rebacktest"
    else:
        from src.training.trainer import train_one_model
        tr_mask = feature_df.index < oos_start
        ds      = T0Dataset(feature_df[tr_mask], label[tr_mask],
                            limit_mask[tr_mask], selected_features=selected)

        logger.info("Step 3: 训练四个模型（n_features=%d，device=%s）...", n_feats, device)
        models   = build_all_models(cfg, n_features=n_feats)
        val_aucs = {}

        for name, model in models.items():
            model = _maybe_compile(model, device)
            t0    = time.time()
            logger.info("  ▶ 训练 [%s] ...", name)
            trained, auc   = train_one_model(model, ds.X, ds.y, cfg=cfg, device=device)
            models[name]   = trained
            val_aucs[name] = auc
            elapsed = time.time() - t0
            logger.info("  ✅ [%s]  val AUC=%.4f  耗时=%.1fs", name, auc, elapsed)

        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(device) / 1024**3
            resv  = torch.cuda.memory_reserved(device) / 1024**3
            logger.info("  GPU显存: 已用 %.2fGB / 保留 %.2fGB", alloc, resv)

        predictor  = EnsemblePredictor(models, val_aucs, device=device, selected_features=selected)
        new_run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        predictor.save(new_run_id)

    logger.info("Step 4: OOS 回测（%s ~ %s）...",
                cfg["data"]["oos_start"], cfg["data"]["oos_end"])
    close  = raw_df["close"].reindex(feature_df.index)
    result = run_backtest(feature_df[oos_mask], close[oos_mask], predictor, cfg=cfg)

    perf = result["metrics"]
    print("\n" + "═"*48)
    print("  📈 OOS 回测结果")
    print("═"*48)
    print(f"  总收益:     {perf['total_return']*100:+.2f}%")
    print(f"  基准收益:   {perf.get('benchmark_return',0)*100:+.2f}%  (买入持有688981)")
    print(f"  超额收益:   {perf.get('excess_return',0)*100:+.2f}%")
    print(f"  夏普比率:   {perf['sharpe']:.3f}")
    print(f"  索提诺比率: {perf['sortino']:.3f}")
    print(f"  Omega比率:  {perf['omega']:.3f}")
    print(f"  最大回撤:   {perf['max_drawdown']*100:.2f}%")
    print(f"  Calmar比率: {perf['calmar']:.3f}")
    print(f"  胜率:       {perf['win_rate']*100:.1f}%")
    print(f"  盈亏比:     {perf.get('payoff_ratio',0):.3f}")
    print(f"  交易次数:   {perf['n_trades']}")
    print(f"  平仓类型:   {perf.get('trade_types',{})}")
    print("═"*48 + "\n")

    out_dir = RESULTS / new_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result["equity"].to_csv(out_dir / "equity.csv")
    pd.DataFrame(result["trades"]).to_csv(out_dir / "trades.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        m_clean = {k: (v if not isinstance(v, dict) else str(v)) for k, v in perf.items()}
        json.dump(m_clean, f, indent=2, ensure_ascii=False)

    logger.info("结果已保存 → results/%s/   run_id=%s", new_run_id, new_run_id)

    # ── Step 5: 可解释性分析（自动在 full 模式末尾运行）────────
    logger.info("Step 5: 可解释性分析（SHAP/IG/排列重要性）...")
    try:
        mode_explain(cfg, device, new_run_id)
    except Exception as e:
        logger.warning("可解释性分析未完成（不影响回测结果）: %s", e)
        logger.warning("可单独重跑: python main.py --mode explain --run_id %s", new_run_id)

    return new_run_id


# ── 模式：explain ─────────────────────────────────────────────────
def mode_explain(cfg, device, run_id: str):
    """
    Step 5: 可解释性分析
    ├── SHAP全局特征重要性 + 时序热力图
    ├── 集成梯度（Integrated Gradients）特征归因
    ├── 排列重要性（Permutation Importance）
    └── 特征稳定性分析（跨时间窗口）

    输出到 results/<run_id>/explain/
    单独调用：python main.py --mode explain --run_id <id>
    """
    import json
    import pandas as pd

    from src.data.fetcher import fetch_bars
    from src.features.dataset import prepare_dataset, T0Dataset
    from src.models.ensemble import EnsemblePredictor
    from src.explain.shap_analysis import (
        compute_shap_values, global_importance,
        plot_global_importance, plot_temporal_heatmap,
        feature_stability_analysis, permutation_importance,
    )
    from src.explain.gradient import batch_integrated_gradients, ig_global_importance

    logger = logging.getLogger("main.explain")

    # ── 数据准备 ──────────────────────────────────────────────────
    logger.info("[Explain] 加载数据 & 特征工程...")
    raw_df     = fetch_bars()
    feature_df, label, limit_mask, selected = prepare_dataset(
        raw_df, use_feature_selection=True
    )
    oos_start = pd.Timestamp(cfg["data"]["oos_start"], tz="Asia/Shanghai")
    oos_mask  = feature_df.index >= oos_start

    oos_ds = T0Dataset(
        feature_df[oos_mask], label[oos_mask],
        limit_mask[oos_mask], selected_features=selected
    )
    n = len(oos_ds)
    logger.info("[Explain] OOS样本: %d，特征: %d", n, len(selected))

    # ── 加载主模型（transformer_lstm） ────────────────────────────
    logger.info("[Explain] 加载模型 run_id=%s ...", run_id)
    predictor  = EnsemblePredictor.load(run_id, val_aucs=None, device=device)
    main_model = predictor.models["transformer_lstm"]

    # ── 输出目录 ──────────────────────────────────────────────────
    out_dir = RESULTS / run_id / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 样本分配：SHAP 慢，严格限量 ──────────────────────────────
    X_all = oos_ds.X
    y_all = oos_ds.y
    n_bg   = min(80,  n // 5)   # SHAP 背景集
    n_shap = min(50,  n // 8)   # SHAP 解释集（DeepExplainer O(n·bg) 复杂度）
    n_ig   = min(120, n // 3)   # 集成梯度
    n_perm = min(400, n)        # 排列重要性
    n_stab = min(500, n)        # 特征稳定性

    X_bg   = X_all[:n_bg]
    X_shap = X_all[n_bg : n_bg + n_shap]
    X_ig   = X_all[:n_ig];   y_ig   = y_all[:n_ig]
    X_perm = X_all[:n_perm]; y_perm = y_all[:n_perm]
    X_stab = X_all[:n_stab]; y_stab = y_all[:n_stab]

    summary = {}   # 汇总各方法的 Top-3 特征，写入 JSON

    # ── 5a: SHAP ─────────────────────────────────────────────────
    logger.info("[Explain 5a] SHAP分析 (bg=%d exp=%d)...", n_bg, n_shap)
    try:
        import torch.backends.cudnn as _cudnn
        _cudnn_was = _cudnn.enabled
        _cudnn.enabled = False          # cuDNN LSTM 不支持 eval mode 下的反向传播
        try:
            shap_vals = compute_shap_values(main_model, X_bg, X_shap, device=device)
        finally:
            _cudnn.enabled = _cudnn_was  # 恢复原来的设置
        imp_df    = global_importance(shap_vals, feature_names=selected)
        plot_global_importance(imp_df, save_path=out_dir / "shap_importance.png")
        plot_temporal_heatmap(
            shap_vals, feature_names=selected,
            save_path=out_dir / "shap_heatmap.png"
        )
        imp_df.to_csv(out_dir / "shap_importance.csv", index=False)
        summary["shap_top3"] = imp_df["feature"].head(3).tolist()
        logger.info("[Explain 5a] ✅ SHAP完成")
    except Exception as e:
        logger.warning("[Explain 5a] SHAP跳过: %s", e)
        imp_df = None

    # ── 5b: 集成梯度 ─────────────────────────────────────────────
    logger.info("[Explain 5b] 集成梯度 (n=%d)...", n_ig)
    try:
        import torch.backends.cudnn as _cudnn
        _cudnn_was = _cudnn.enabled
        _cudnn.enabled = False          # 同 SHAP，关闭 cuDNN 以支持 eval backward
        try:
            ig_attrs = batch_integrated_gradients(main_model, X_ig, device=device)
        finally:
            _cudnn.enabled = _cudnn_was
        ig_imp   = ig_global_importance(ig_attrs, feature_names=selected)
        ig_imp.to_csv(out_dir / "ig_importance.csv", index=False)
        summary["ig_top3"] = ig_imp["feature"].head(3).tolist()
        logger.info("[Explain 5b] ✅ IG完成")
    except Exception as e:
        logger.warning("[Explain 5b] IG跳过: %s", e)
        ig_imp = None

    # ── 5c: 排列重要性 ───────────────────────────────────────────
    logger.info("[Explain 5c] 排列重要性 (n=%d n_repeats=3)...", n_perm)
    try:
        perm_imp = permutation_importance(
            main_model, X_perm, y_perm,
            n_repeats=3, device=device, feature_names=selected
        )
        perm_imp.to_csv(out_dir / "perm_importance.csv", index=False)
        summary["perm_top3"] = perm_imp["feature"].head(3).tolist()
        logger.info("[Explain 5c] ✅ 排列重要性完成")
    except Exception as e:
        logger.warning("[Explain 5c] 排列重要性跳过: %s", e)
        perm_imp = None

    # ── 5d: 特征稳定性 ───────────────────────────────────────────
    logger.info("[Explain 5d] 特征稳定性分析 (n=%d)...", n_stab)
    try:
        stab = feature_stability_analysis(
            main_model, X_stab, y_stab, n_windows=5, device=device
        )
        stab.to_csv(out_dir / "feature_stability.csv", index=False)
        logger.info("[Explain 5d] ✅ 稳定性分析完成")
    except Exception as e:
        logger.warning("[Explain 5d] 特征稳定性跳过: %s", e)

    # ── 写入汇总 JSON ────────────────────────────────────────────
    with open(out_dir / "explain_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 打印结果 ──────────────────────────────────────────────────
    print("\n" + "═"*50)
    print("  🔍 可解释性分析结果")
    print("═"*50)
    if imp_df is not None:
        print("  SHAP Top-5 特征:")
        for _, r in imp_df.head(5).iterrows():
            print(f"    {r['feature']:<26} {r['shap_importance']:.4f}")
    if ig_imp is not None:
        print("  集成梯度 Top-5 特征:")
        for _, r in ig_imp.head(5).iterrows():
            print(f"    {r['feature']:<26} {r['ig_importance']:.4f}")
    if perm_imp is not None:
        print("  排列重要性 Top-5 特征 (AUC下降量):")
        val_col = "drop_mean" if "drop_mean" in perm_imp.columns else perm_imp.columns[1]
        for _, r in perm_imp.head(5).iterrows():
            print(f"    {r['feature']:<26} {r[val_col]:+.4f}")
    print(f"\n  📁 输出目录: results/{run_id}/explain/")
    print("     shap_importance.png / shap_heatmap.png")
    print("     ig_importance.csv / perm_importance.csv / feature_stability.csv")
    print("═"*50 + "\n")

    return out_dir


# ── 模式：walk_forward ────────────────────────────────────────────
def mode_walk_forward(cfg, device, use_optuna=False):
    from src.data.fetcher import fetch_bars
    from src.backtest.walk_forward import run_walk_forward
    import json

    logger = logging.getLogger("main.wf")
    logger.info("获取数据...")
    raw_df = fetch_bars()

    logger.info("Walk-Forward 开始（optuna=%s，device=%s）...", use_optuna, device)
    wf_result = run_walk_forward(
        raw_df, cfg=cfg,
        use_optuna=use_optuna,
        use_feature_selection=True,
        device=device,
    )

    summ = wf_result["summary"]
    print("\n" + "═"*48)
    print("  📊 Walk-Forward 汇总")
    print("═"*48)
    for k, v in summ.items():
        print(f"  {k:<22}: {v:.4f}")
    print("═"*48 + "\n")

    out = RESULTS / "walk_forward_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)
    logger.info("Walk-Forward 结果 → %s", out)


# ── 模式：live ────────────────────────────────────────────────────
def mode_live(cfg, run_id: str, device):
    from src.models.ensemble import EnsemblePredictor
    from src.tdx.live_trader import LiveTrader

    logger = logging.getLogger("main.live")
    if not cfg.get("tdx", {}).get("enabled", False):
        logger.error("通达信未启用，请在 config.yaml 设置 tdx.enabled: true")
        sys.exit(1)

    logger.info("加载模型 run_id=%s ...", run_id)
    predictor = EnsemblePredictor.load(run_id, val_aucs={
        "transformer_lstm": 0.6, "vanilla_lstm": 0.58,
        "cnn_lstm": 0.57, "mlp": 0.55,
    }, device=device)

    logger.info("启动实时信号循环（Ctrl+C 退出）...")
    LiveTrader(predictor, cfg=cfg).run()


# ── 入口 ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="688981 T+0 策略")
    parser.add_argument("--mode",      choices=["full", "walk_forward", "live", "explain"], default="full")
    parser.add_argument("--optuna",    action="store_true")
    parser.add_argument("--run_id",    type=str, default="")
    parser.add_argument("--log",       type=str, default="run.log")
    parser.add_argument("--check_gpu", action="store_true", help="仅打印GPU信息后退出")
    args = parser.parse_args()

    setup_logging(log_file=args.log)

    if args.check_gpu:
        _check_and_print_gpu()
        return

    device = _check_and_print_gpu()
    _print_cost_summary()

    cfg = load_config()
    logging.getLogger("main").info("模式: %s  device: %s", args.mode, device)

    if args.mode == "full":
        mode_full(cfg, device, run_id=args.run_id)
    elif args.mode == "walk_forward":
        mode_walk_forward(cfg, device, use_optuna=args.optuna)
    elif args.mode == "explain":
        if not args.run_id:
            print("错误：--mode explain 需要指定 --run_id")
            sys.exit(1)
        mode_explain(cfg, device, run_id=args.run_id)
    elif args.mode == "live":
        if not args.run_id:
            print("错误：--mode live 需要指定 --run_id")
            sys.exit(1)
        mode_live(cfg, args.run_id, device)


if __name__ == "__main__":
    main()
