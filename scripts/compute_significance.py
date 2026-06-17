"""
scripts/compute_significance.py
对指定 run_id 的回测结果计算两种显著性检验：
  方法A: 逐笔交易收益率 t检验 (trades.csv)，H0: 平均每笔收益=0
  方法B: 逐bar净值收益率 -> 年化Sharpe的t检验 (equity.csv)，H0: Sharpe=0
          （Lo 2002近似: t = SR_bar * sqrt(N)）

用法:
    python scripts/compute_significance.py 20260603_165651 20260615_165941_rebacktest
"""
import sys
import pandas as pd
import numpy as np
from scipy import stats


def analyze(run_id: str) -> None:
    trades = pd.read_csv(f"results/{run_id}/trades.csv")
    eq     = pd.read_csv(f"results/{run_id}/equity.csv", index_col=0)

    print(f"\n========== {run_id} ==========")

    # 方法A：逐笔
    r = trades["return_pct"].values
    n = len(r)
    mean, std = r.mean(), r.std(ddof=1)
    t_a = mean / (std / np.sqrt(n))
    p_a = stats.t.sf(abs(t_a), df=n - 1) * 2
    print(f"[逐笔]  n={n:4d}  mean={mean*100:+.4f}%  std={std*100:.4f}%  "
          f"t={t_a:+.3f}  p={p_a:.6f}")

    # 方法B：逐bar
    nav = eq.iloc[:, 0]
    ret = nav.pct_change().dropna()
    N = len(ret)
    sr_bar = ret.mean() / ret.std(ddof=1)
    sr_ann = sr_bar * np.sqrt(252 * 8)   # 30min K线，每日8根
    t_b = sr_bar * np.sqrt(N)
    p_b = stats.t.sf(abs(t_b), df=N - 1) * 2
    print(f"[逐bar] N={N:4d}  年化Sharpe={sr_ann:+.4f}  t={t_b:+.3f}  p={p_b:.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/compute_significance.py <run_id1> [run_id2] ...")
        sys.exit(1)
    for rid in sys.argv[1:]:
        analyze(rid)