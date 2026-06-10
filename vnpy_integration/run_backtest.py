"""
vnpy_integration/run_backtest.py
==================================
使用 VeighNa BacktestingEngine 运行 ML T+0 策略回测。

核心设计：
  - 跳过 engine.load_data()，直接注入 parquet 数据
    （不需要 VeighNa 数据库、不修改任何 data/ 文件）
  - 结果额外保存到 results/<run_id>_vnpy/ 保持与原有格式兼容
  - 不影响已有的 results/ 目录和模型文件

用法
----
python vnpy_integration/run_backtest.py
python vnpy_integration/run_backtest.py --run_id 20260603_165651
python vnpy_integration/run_backtest.py --run_id 20260603_165651 --show_chart
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Interval

from src.utils import load_config, RESULTS
from vnpy_integration.ml_cta_strategy import MLT0Strategy
from vnpy_integration.data_loader import load_bars_from_parquet


# ── 成本常量 ─────────────────────────────────────────────────────
# VeighNa 的 rate 是单边佣金率（含经手费约 0.00487bp）
# stamp_duty（0.1% 卖出方）VeighNa 不单独区分；
# 这里用买入成本率近似，保持与原 config 一致
_COMMISSION_RATE  = 0.00025   # 万2.5，与 config.yaml 一致
# slippage: VeighNa 单位为价格（元/股），约 0.02% × 均价 70 = 0.014，取 0.01
_SLIPPAGE_YUAN    = 0.01      # 每股每侧滑点（元）
_SIZE             = 1         # 1 合约 = 1 股
_PRICETICK        = 0.01      # 688981 最小价格变动单位


def _find_latest_run_id() -> str:
    dirs = sorted(
        [d.name for d in RESULTS.iterdir()
         if d.is_dir() and not d.name.endswith("_vnpy") and (d / "metrics.json").exists()],
        reverse=True,
    )
    if not dirs:
        raise RuntimeError("results/ 目录下没有找到有效的 run_id，请先运行 main.py")
    return dirs[0]


def run_vnpy_backtest(
    run_id: str = "",
    show_chart: bool = False,
    extra_params: dict | None = None,
) -> dict:
    """
    运行 VeighNa 回测并返回统计指标字典。

    Parameters
    ----------
    run_id : str
        要加载的模型 run_id。空字符串时自动选择最新的。
    show_chart : bool
        是否调用 engine.show_chart() 展示 VeighNa 内置图表（需要 Qt 环境）。
    extra_params : dict, optional
        额外覆盖策略参数，如 {"buy_threshold": 0.65}。

    Returns
    -------
    dict
        VeighNa 计算的统计指标。
    """
    cfg = load_config()

    if not run_id:
        run_id = _find_latest_run_id()
    print(f"\n{'='*55}")
    print(f"  VeighNa 回测 | run_id = {run_id}")
    print(f"{'='*55}")

    # ── 加载历史数据（不使用 VeighNa 数据库）────────────────────
    oos_start = datetime.fromisoformat(cfg["data"]["oos_start"])
    oos_end   = datetime.fromisoformat(cfg["data"]["oos_end"])

    bars = load_bars_from_parquet(start=oos_start, end=oos_end)
    if not bars:
        raise RuntimeError("没有可用的 K 线数据，请检查 parquet 文件和日期配置")

    # ── 配置回测引擎 ──────────────────────────────────────────────
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=f"{cfg['data']['symbol']}.SSE",
        interval=Interval.MINUTE,
        start=oos_start,
        end=oos_end,
        rate=_COMMISSION_RATE,
        slippage=_SLIPPAGE_YUAN,
        size=_SIZE,
        pricetick=_PRICETICK,
        capital=cfg["backtest"]["initial_capital"],
    )

    # ── 策略参数 ──────────────────────────────────────────────────
    strategy_params = {
        "run_id":           run_id,
        "window_size":      cfg["features"]["window_size"],
        "buy_threshold":    cfg["strategy"]["buy_threshold"],
        "sell_threshold":   cfg["strategy"]["sell_threshold"],
        "stop_loss_pct":    cfg["strategy"]["stop_loss_pct"],
        "max_daily_trades": cfg["strategy"]["max_daily_trades"],
    }
    if extra_params:
        strategy_params.update(extra_params)
        print(f"  覆盖参数: {extra_params}")

    engine.add_strategy(MLT0Strategy, strategy_params)

    # ── 注入数据（绕过数据库，直接赋值 history_data）────────────
    # run_backtesting() 直接遍历 history_data，不会再调用 load_data()
    engine.history_data = bars
    print(f"  已注入 {len(bars)} 根 K 线（跳过 VeighNa 数据库）")

    # ── 执行回测 ──────────────────────────────────────────────────
    print("\n  开始回放历史数据...")
    engine.run_backtesting()

    # ── 计算结果 ──────────────────────────────────────────────────
    daily_df = engine.calculate_result()
    stats    = engine.calculate_statistics()

    # ── 打印核心指标 ──────────────────────────────────────────────
    print(f"\n{'─'*45}")
    print("  VeighNa 回测结果")
    print(f"{'─'*45}")

    key_map = {
        "start_date":            "起始日期",
        "end_date":              "结束日期",
        "total_return":          "总收益率",
        "annual_return":         "年化收益率",
        "sharpe_ratio":          "夏普比率",
        "ewm_sharpe":            "EWM夏普",
        "return_drawdown_ratio": "收益回撤比",
        "max_ddpercent":         "最大回撤(%)",
        "total_trade_count":     "总交易次数",
        "profit_days":           "盈利天数",
        "loss_days":             "亏损天数",
        "total_commission":      "总佣金",
        "total_slippage":        "总滑点",
        "end_balance":           "最终余额",
    }
    for k, label in key_map.items():
        v = stats.get(k, "N/A")
        if isinstance(v, float):
            print(f"  {label:<16} {v:.4f}")
        else:
            print(f"  {label:<16} {v}")

    # ── 与原有回测对比 ────────────────────────────────────────────
    original_metrics_path = RESULTS / run_id / "metrics.json"
    if original_metrics_path.exists():
        with open(original_metrics_path, encoding="utf-8") as f:
            orig = json.load(f)
        print(f"\n{'─'*45}")
        print("  对比原自研回测引擎")
        print(f"{'─'*45}")
        # VeighNa total_return/max_ddpercent 以百分数存储(4.44=4.44%)
        # 原引擎以小数存储(0.0494=4.94%)，对比前统一换算成百分数显示
        rows = [
            ("总收益率",
             f"{stats.get('total_return', 0):+.2f}%",
             f"{orig.get('total_return', 0)*100:+.2f}%",
             stats.get('total_return',0) - orig.get('total_return',0)*100),
            ("Sharpe",
             f"{stats.get('sharpe_ratio', 0):+.4f}",
             f"{orig.get('sharpe', 0):+.4f}",
             stats.get('sharpe_ratio',0) - orig.get('sharpe',0)),
            ("最大回撤",
             f"{abs(stats.get('max_ddpercent', 0)):+.2f}%",
             f"{orig.get('max_drawdown', 0)*100:+.2f}%",
             abs(stats.get('max_ddpercent',0)) - orig.get('max_drawdown',0)*100),
            ("成交(来回)",
             f"{stats.get('total_trade_count',0)//2} 笔",
             f"{orig.get('n_trades', 0)} 笔",
             None),
        ]
        for label, v_v, v_o, delta in rows:
            d = f"  差={delta:+.2f}" if delta is not None else ""
            print(f"  {label:<10} VeighNa={v_v:<10} 原引擎={v_o}{d}")
        print()
        print("  差异原因: VeighNa T+1成交 / 原引擎T+0收盘 / Sharpe计算基期不同")

    # ── 保存结果到 results/<run_id>_vnpy/ ─────────────────────────
    out_dir = RESULTS / f"{run_id}_vnpy"
    out_dir.mkdir(parents=True, exist_ok=True)

    if daily_df is not None and not daily_df.empty:
        daily_df.to_csv(out_dir / "vnpy_daily.csv", encoding="utf-8")

    # 将 stats 中的 Timestamp 转为字符串，方便 JSON 序列化
    # VeighNa stats can contain numpy int64/float64 — convert before JSON dump
    stats_serializable = {}
    for k, v in stats.items():
        if hasattr(v, "isoformat"):           stats_serializable[k] = str(v)
        elif isinstance(v, np.integer):        stats_serializable[k] = int(v)
        elif isinstance(v, np.floating):       stats_serializable[k] = float(v)
        else:                                  stats_serializable[k] = v

    def _np_default(obj):
        """Handle numpy scalar types that stdlib json can't serialize."""
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if hasattr(obj, "isoformat"):    return str(obj)
        raise TypeError(f"Not serializable: {type(obj).__name__}")

    with open(out_dir / "vnpy_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_serializable, f, ensure_ascii=False, indent=2,
                  default=_np_default)

    print(f"\n  结果保存到: {out_dir}")
    print(f"{'='*55}\n")

    if show_chart:
        try:
            engine.show_chart()
        except Exception as e:
            print(f"  ⚠️ show_chart 需要 Qt 环境: {e}")

    return stats


# ── 参数优化（网格搜索）──────────────────────────────────────────
def optimize_vnpy(run_id: str = "") -> None:
    """
    使用 VeighNa 内置的参数优化（遗传算法/网格）。
    示例：对 buy_threshold 和 stop_loss_pct 做网格搜索。
    """
    from vnpy_ctastrategy.backtesting import OptimizationSetting

    cfg = load_config()
    if not run_id:
        run_id = _find_latest_run_id()

    oos_start = datetime.fromisoformat(cfg["data"]["oos_start"])
    oos_end   = datetime.fromisoformat(cfg["data"]["oos_end"])
    bars      = load_bars_from_parquet(start=oos_start, end=oos_end)

    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=f"{cfg['data']['symbol']}.SSE",
        interval=Interval.MINUTE,
        start=oos_start,
        end=oos_end,
        rate=_COMMISSION_RATE,
        slippage=_SLIPPAGE_YUAN,
        size=_SIZE,
        pricetick=_PRICETICK,
        capital=cfg["backtest"]["initial_capital"],
    )
    engine.add_strategy(MLT0Strategy, {
        "run_id":        run_id,
        "window_size":   cfg["features"]["window_size"],
        "sell_threshold":cfg["strategy"]["sell_threshold"],
        "max_daily_trades": cfg["strategy"]["max_daily_trades"],
    })
    engine.history_data = bars

    setting = OptimizationSetting()
    setting.add_parameter("buy_threshold",  0.57, 0.65, 0.02)   # 4 个值
    setting.add_parameter("stop_loss_pct",  0.015, 0.030, 0.005)# 4 个值
    setting.set_target("sharpe_ratio")

    print("开始参数优化（网格搜索 buy_threshold × stop_loss_pct）...")
    results = engine.run_ga_optimization(setting, output=True)

    out_path = RESULTS / f"{run_id}_vnpy" / "opt_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results[:20], f, ensure_ascii=False, indent=2, default=str)
    print(f"优化结果已保存到: {out_path}")


# ── CLI ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VeighNa BacktestingEngine 回测 688981 T+0 策略"
    )
    parser.add_argument("--run_id",     type=str, default="",
                        help="模型 run_id（默认使用最新）")
    parser.add_argument("--show_chart", action="store_true",
                        help="展示 VeighNa 内置图表（需要 Qt 桌面环境）")
    parser.add_argument("--optimize",   action="store_true",
                        help="运行参数网格优化")
    args = parser.parse_args()

    if args.optimize:
        optimize_vnpy(run_id=args.run_id)
    else:
        run_vnpy_backtest(run_id=args.run_id, show_chart=args.show_chart)
