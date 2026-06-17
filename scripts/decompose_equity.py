"""
scripts/decompose_equity.py
按月拆解净值曲线，判断是逐月衰减还是某个时间点突然大幅回撤。

用法:
    python scripts/decompose_equity.py results/20260615_165941_rebacktest/equity.csv
"""
import sys
import pandas as pd


def max_drawdown(nav: pd.Series) -> float:
    roll_max = nav.cummax()
    dd = (nav - roll_max) / roll_max
    return dd.min()


def main(path: str) -> None:
    eq = pd.read_csv(path, index_col=0)
    nav = eq.iloc[:, 0]
    nav.index = pd.to_datetime(nav.index, utc=True).tz_convert("Asia/Shanghai")

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    print(f"区间: {nav.index[0].date()} ~ {nav.index[-1].date()}")
    print(f"整体收益: {total_ret*100:+.2f}%   整体最大回撤: {max_drawdown(nav)*100:.2f}%\n")

    print("=== 按月拆解 ===")
    print(f"{'月份':<10}{'收益':>10}{'区间最大回撤':>14}{'bars':>8}")
    for period, grp in nav.resample("ME"):
        if len(grp) < 2:
            continue
        ret = grp.iloc[-1] / grp.iloc[0] - 1
        dd  = max_drawdown(grp)
        print(f"{period.strftime('%Y-%m'):<10}{ret*100:>+9.2f}%{dd*100:>13.2f}%{len(grp):>8d}")


if __name__ == "__main__":
    main(sys.argv[1])