"""
vnpy_integration/data_loader.py
================================
将项目现有的 Baostock parquet 文件转换为 VeighNa BarData 列表。

不依赖 VeighNa 数据库（无需 load_data()），直接注入到
BacktestingEngine.history_data。这样迁移后的回测不改动任何
原始 data/ 目录或 parquet 文件。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData

from src.utils import RAW, load_config


# 通达信实盘暂不需要区分 SSE/SZSE 内部代码，这里统一用 Exchange.SSE
_EXCHANGE_MAP = {
    "6": Exchange.SSE,   # 上交所（包含 688981 科创板）
    "0": Exchange.SZSE,  # 深交所
    "3": Exchange.SZSE,
}


def _infer_exchange(symbol: str) -> Exchange:
    return _EXCHANGE_MAP.get(symbol[0], Exchange.SSE)


def _interval_from_freq(bar_freq: int) -> Interval:
    """
    将分钟数映射到 VeighNa Interval 枚举。

    VeighNa 4.x 的 Interval 只有：MINUTE / HOUR / DAILY / WEEKLY / TICK，
    没有 MINUTE_5 / MINUTE_30 等细分枚举。
    由于我们直接注入 parquet 数据（跳过数据库），interval 仅作 BarData 标签，
    不影响回测逻辑，映射到最近合法值即可。
    """
    if bar_freq >= 60:
        return Interval.HOUR
    return Interval.MINUTE  # 5 / 15 / 30 分钟统一用 MINUTE（4.x 无细分枚举）


def load_bars_from_parquet(
    parquet_path: Optional[Path] = None,
    symbol: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[BarData]:
    """
    将 parquet 文件读取为 VeighNa BarData 列表。

    Parameters
    ----------
    parquet_path : Path, optional
        显式指定 parquet 路径。为 None 时自动从 config.yaml 推断：
        data/raw/{symbol}_{bar_frequency}min.parquet
    symbol : str, optional
        股票代码（如 "688981"），为 None 时从 config 读取
    start : datetime, optional
        回测起始日期（含）。为 None 时取全量
    end : datetime, optional
        回测结束日期（含）。为 None 时取全量

    Returns
    -------
    list[BarData]
        按时间升序排列的 BarData 列表
    """
    cfg      = load_config()
    sym      = symbol or cfg["data"]["symbol"]
    bar_freq = cfg["data"]["bar_frequency"]   # 30

    if parquet_path is None:
        parquet_path = RAW / f"{sym}_{bar_freq}min.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"找不到历史数据文件: {parquet_path}\n"
            f"请先运行: python main.py --mode data  （从 Baostock 拉取数据）"
        )

    df = pd.read_parquet(parquet_path)

    # 确保时区统一
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("Asia/Shanghai")

    # 按日期过滤
    if start is not None:
        start_ts = pd.Timestamp(start).tz_localize("Asia/Shanghai") \
                   if start.tzinfo is None else pd.Timestamp(start)
        df = df[df.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end).tz_localize("Asia/Shanghai") \
                 if end.tzinfo is None else pd.Timestamp(end)
        df = df[df.index <= end_ts]

    if df.empty:
        raise ValueError(f"指定时间段内无数据: {start} ~ {end}")

    exchange = _infer_exchange(sym)
    interval = _interval_from_freq(bar_freq)

    bars: list[BarData] = []
    for dt, row in df.iterrows():
        bar = BarData(
            symbol=sym,
            exchange=exchange,
            datetime=dt,
            interval=interval,
            volume=float(row.get("volume", 0)),
            open_interest=0.0,                   # 股票无持仓量
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            gateway_name="PARQUET",
        )
        bars.append(bar)

    bars.sort(key=lambda b: b.datetime)
    print(f"[DataLoader] 加载 {len(bars)} 根 {bar_freq}min K线 "
          f"({bars[0].datetime.date()} ~ {bars[-1].datetime.date()})")
    return bars
