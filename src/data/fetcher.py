"""src/data/fetcher.py — 支持多周期K线（5min / 30min）从Baostock拉取。"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import pandas as pd

from src.utils import RAW, load_config

logger = logging.getLogger(__name__)

_SESSION = [("09:31", "11:30"), ("13:01", "15:00")]


def _in_session(t) -> bool:
    from datetime import time as dtime
    for s, e in _SESSION:
        if dtime.fromisoformat(s) <= t <= dtime.fromisoformat(e):
            return True
    return False


def _to_bs_code(symbol: str) -> str:
    if symbol.startswith("sh.") or symbol.startswith("sz."):
        return symbol
    return f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}"


def _parse_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    t6       = time_series.str[8:14]
    time_str = t6.str[:2] + ":" + t6.str[2:4] + ":" + t6.str[4:6]
    return (pd.to_datetime(date_series + " " + time_str, format="%Y-%m-%d %H:%M:%S")
              .dt.tz_localize("Asia/Shanghai"))


def _fetch_year_worker(bs_code: str, year: int, frequency: str, result: dict) -> None:
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        result["error"] = f"login failed: {lg.error_msg}"
        return
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,time,open,high,low,close,volume,amount",
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            frequency=frequency,
            adjustflag="3",
        )
        if rs.error_code != "0":
            result["error"] = f"query failed: {rs.error_msg}"
            return
        rows, count = [], 0
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
            count += 1
            if count % 5000 == 0:
                result["progress"] = count
        result["rows"] = rows
    except Exception as e:
        result["error"] = str(e)
    finally:
        bs.logout()


def _fetch_year_with_timeout(bs_code: str, year: int, frequency: str,
                              timeout: int = 90) -> pd.DataFrame:
    result: dict = {"rows": None, "error": None, "progress": 0}
    t = threading.Thread(
        target=_fetch_year_worker, args=(bs_code, year, frequency, result), daemon=True
    )
    t.start()
    deadline, last_prog = time.time() + timeout, 0
    while t.is_alive():
        t.join(timeout=2)
        prog = result.get("progress", 0)
        if prog > last_prog:
            logger.info("    ... %d rows so far", prog)
            last_prog = prog
        if time.time() > deadline:
            logger.warning("  Year %d timed out after %ds, retrying ...", year, timeout)
            return pd.DataFrame()
    if result.get("error"):
        logger.warning("  Year %d error: %s", year, result["error"])
        return pd.DataFrame()
    rows = result.get("rows") or []
    if not rows:
        logger.warning("  Year %d: baostock returned 0 rows.", year)
        return pd.DataFrame()
    cols = ["date", "time", "open", "high", "low", "close", "volume", "amount"]
    df = pd.DataFrame(rows, columns=cols)
    logger.info("  Year %d sample row -> date=%s time=%s close=%s (total raw rows: %d)",
                year, df["date"].iloc[0], df["time"].iloc[0],
                df["close"].iloc[0], len(df))
    return df


def fetch_bars(
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
    frequency: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    通用K线拉取函数，支持任意Baostock频率（"5","15","30","60","d"）。
    frequency默认从config读取。
    """
    import baostock as bs

    cfg       = load_config()
    symbol    = symbol    or cfg["data"]["symbol"]
    start     = start     or cfg["data"]["start"]
    end       = end       or cfg["data"]["oos_end"]
    frequency = frequency or str(cfg["data"].get("bar_frequency", "30"))

    freq_label  = f"{frequency}min" if frequency != "d" else "daily"
    final_cache = RAW / f"{symbol}_{freq_label}.parquet"

    if final_cache.exists() and not force_refresh:
        df = pd.read_parquet(final_cache)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("Asia/Shanghai")
        start_date = pd.Timestamp(start).date()
        end_date   = pd.Timestamp(end).date()
        if df.index.min().date() <= start_date and df.index.max().date() >= end_date:
            logger.info("Using cached data: %s (%d rows)", final_cache, len(df))
            return df
        logger.info("Cache incomplete, re-fetching missing years.")

    bs_code  = _to_bs_code(symbol)
    start_yr = datetime.strptime(start, "%Y-%m-%d").year
    end_yr   = datetime.strptime(end,   "%Y-%m-%d").year

    logger.info("Fetching %s %smin data from Baostock (%s ~ %s), unadjusted ...",
                bs_code, frequency, start, end)

    yearly_dfs = []
    for year in range(start_yr, end_yr + 1):
        year_cache = RAW / f"{symbol}_{freq_label}_{year}.parquet"

        if year_cache.exists() and not force_refresh:
            df_yr = pd.read_parquet(year_cache)
            if df_yr.index.tzinfo is None:
                df_yr.index = df_yr.index.tz_localize("Asia/Shanghai")
            logger.info("  Year %d: loaded from cache (%d rows)", year, len(df_yr))
            yearly_dfs.append(df_yr)
            continue

        logger.info("  Fetching year %d (timeout=90s) ...", year)
        t0    = time.time()
        chunk = _fetch_year_with_timeout(bs_code, year, frequency, timeout=90)

        if len(chunk) == 0:
            logger.info("  Year %d: retrying ...", year)
            time.sleep(3)
            chunk = _fetch_year_with_timeout(bs_code, year, frequency, timeout=90)

        if len(chunk) == 0:
            logger.warning("  Year %d: skipping (no data after retry).", year)
            continue

        chunk["datetime"] = _parse_datetime(chunk["date"], chunk["time"])
        float_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in float_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = (chunk.set_index("datetime")
                      [["open", "high", "low", "close", "volume", "amount"]]
                      .sort_index())
        chunk = chunk[~chunk.index.duplicated(keep="first")]
        chunk = chunk[chunk["volume"] > 0]

        if frequency != "d":
            mask  = pd.Series([_in_session(t.time()) for t in chunk.index], index=chunk.index)
            chunk = chunk[mask]

        logger.info("  Year %d done: %d in-session rows (%.1fs) -> caching",
                    year, len(chunk), time.time() - t0)

        chunk_save = chunk.copy()
        chunk_save.index = chunk_save.index.tz_localize(None)
        chunk_save.to_parquet(year_cache)
        yearly_dfs.append(chunk)

    if not yearly_dfs:
        raise RuntimeError(f"No data fetched for {bs_code}.")

    raw = pd.concat(yearly_dfs).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]

    start_ts = pd.Timestamp(start).tz_localize("Asia/Shanghai")
    end_ts   = pd.Timestamp(end).tz_localize("Asia/Shanghai") + pd.Timedelta(days=1)
    raw = raw[(raw.index >= start_ts) & (raw.index < end_ts)]

    raw_save = raw.copy()
    raw_save.index = raw_save.index.tz_localize(None)
    raw_save.to_parquet(final_cache)
    logger.info("Saved %d rows -> %s", len(raw), final_cache)

    for year in range(start_yr, end_yr + 1):
        p = RAW / f"{symbol}_{freq_label}_{year}.parquet"
        if p.exists():
            p.unlink()

    return raw


# 向后兼容别名
def fetch_5min(**kwargs) -> pd.DataFrame:
    return fetch_bars(frequency="5", **kwargs)
