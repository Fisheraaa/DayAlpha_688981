"""
src/tdx/connector.py — 通达信模拟盘实时行情接入
=================================================
通达信使用 pytdx 库的 HQ（行情）协议，端口 7709。
本模块提供：
  1. TdxConnector   — 连接管理、心跳、断线重连
  2. TdxBar5Min     — 获取最新5分钟K线
  3. LiveBarBuffer  — 维护滚动K线缓冲区（配合预测器）

依赖：
  pip install pytdx pandas

使用方式（config.yaml 中 tdx.enabled: true）：
  from src.tdx.connector import TdxConnector, LiveBarBuffer
  conn = TdxConnector()
  conn.connect()
  buf  = LiveBarBuffer(conn, symbol="sh688981", capacity=100)
  buf.update()          # 每次5分钟K线结束后调用
  latest = buf.get_df() # 返回最近 N 根K线的 DataFrame
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, date
import pandas as pd

from src.utils import load_config

logger = logging.getLogger(__name__)

try:
    from pytdx.hq import TdxHq_API
    HAS_PYTDX = True
except ImportError:
    HAS_PYTDX = False
    logger.warning(
        "pytdx 未安装，通达信接口不可用。请执行: pip install pytdx"
    )


def _mkt_code(symbol: str) -> tuple[int, str]:
    """
    解析市场代码。
    'sh688981' → (1, '688981')
    'sz000001' → (0, '000001')
    """
    if symbol.startswith("sh"):
        return 1, symbol[2:]
    elif symbol.startswith("sz"):
        return 0, symbol[2:]
    # 纯数字
    return (1, symbol) if symbol.startswith("6") else (0, symbol)


_FALLBACK_SERVERS = [
    ("218.75.126.9",   7709),
    ("119.147.212.81", 7709),
    ("121.14.110.210", 7709),
    ("106.14.95.149",  7709),
]

class TdxConnector:
    """
    通达信行情连接器。

    支持断线重连和心跳保活。
    """

    def __init__(self, host: str | None = None, port: int | None = None):
        cfg         = load_config().get("tdx", {})
        self._host  = host or cfg.get("host", "218.75.126.9")
        self._port  = port or cfg.get("port", 7709)
        self._retry = cfg.get("reconnect_attempts", 3)
        self._hb_iv = cfg.get("heartbeat_interval", 30)
        self._api: "TdxHq_API | None" = None
        self._last_hb = 0.0

    

    def connect(self) -> bool:
        if not HAS_PYTDX:
            logger.error("pytdx 未安装，无法连接通达信")
            return False
        # 先试配置的主服务器，失败后依次试fallback列表
        candidates = [(self._host, self._port)] + [
            s for s in _FALLBACK_SERVERS if s != (self._host, self._port)
        ]
        for host, port in candidates:
            try:
                self._api = TdxHq_API()
                self._api.connect(host, port)
                # 用轻量查询验证连接有效
                test = self._api.get_security_count(1)
                if test and test > 0:
                    self._host = host  # 记住成功的服务器
                    logger.info("通达信连接成功: %s:%d", host, port)
                    self._last_hb = time.time()
                    return True
                self._api.disconnect()
            except Exception as e:
                logger.warning("连接 %s:%d 失败: %s", host, port, e)
        logger.error("所有通达信服务器均连接失败")
        return False

    def disconnect(self) -> None:
        if self._api is not None:
            try:
                self._api.disconnect()
            except Exception:
                pass
            self._api = None
            logger.info("通达信连接已断开")

    def heartbeat(self) -> None:
        """每隔 heartbeat_interval 秒发送心跳（防断线）。"""
        if self._api is None:
            return
        if time.time() - self._last_hb > self._hb_iv:
            try:
                self._api.get_security_count(0)   # 轻量级查询作为心跳
                self._last_hb = time.time()
            except Exception:
                logger.warning("心跳失败，尝试重连...")
                self.connect()

    @property
    def api(self) -> "TdxHq_API | None":
        return self._api

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


class TdxBar5Min:
    """
    获取指定股票的5分钟K线数据。
    """

    def __init__(self, connector: TdxConnector):
        self._conn = connector

    def fetch(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """
        获取最新 count 根5分钟K线。

        Args:
            symbol: 如 'sh688981' 或 '688981'
            count:  K线根数（最多800）

        Returns:
            DataFrame，列: open/high/low/close/volume/amount，index: DatetimeIndex(tz=Asia/Shanghai)
        """
        self._conn.heartbeat()
        api = self._conn.api
        if api is None:
            logger.error("未连接通达信，无法获取数据")
            return pd.DataFrame()

        mkt, code = _mkt_code(symbol)
        try:
            raw = api.get_security_bars(
                category=2,    #30分钟
                market=mkt,
                code=code,
                start=0,
                count=min(count, 800),
            )
        except Exception as e:
            logger.error("获取K线失败: %s", e)
            return pd.DataFrame()

        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        # 通达信返回字段: datetime, open, high, low, close, vol, amount
        rename = {"vol": "volume", "datetime": "dt"}
        df     = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize("Asia/Shanghai")
            df = df.set_index("dt").sort_index()

        # 保留所需字段
        keep = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]
        df   = df[keep].astype(float)
        df   = df[df["volume"] > 0]

        logger.debug("TdxBar5Min: 获取 %d 根K线 (%s)", len(df), symbol)
        return df


class LiveBarBuffer:
    """
    实时K线缓冲区，维护最近 capacity 根5分钟K线。
    每次调用 update() 追加最新K线。
    """

    def __init__(self, connector: TdxConnector, symbol: str | None = None,
                 capacity: int = 32):
        cfg          = load_config().get("tdx", {})
        self._symbol = symbol or cfg.get("symbol", "sh688981")
        self._cap    = capacity
        self._fetcher = TdxBar5Min(connector)
        self._buf: pd.DataFrame = pd.DataFrame()

    def update(self) -> bool:
        """拉取最新K线，追加到缓冲区，返回是否有更新。"""
        new_df = self._fetcher.fetch(self._symbol, count=self._cap)
        if new_df.empty:
            return False

        if self._buf.empty:
            self._buf = new_df
        else:
            combined  = pd.concat([self._buf, new_df])
            combined  = combined[~combined.index.duplicated(keep="last")]
            self._buf = combined.sort_index().tail(self._cap)

        return True

    def get_df(self, n: int | None = None) -> pd.DataFrame:
        """返回最近 n 根K线（None 则返回全部缓冲区）。"""
        if n is None:
            return self._buf.copy()
        return self._buf.tail(n).copy()

    def is_ready(self, min_bars: int = 30) -> bool:
        """缓冲区是否满足最少K线数量（预热）。"""
        return len(self._buf) >= min_bars
