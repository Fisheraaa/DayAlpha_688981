# 通达信接入完整指南
## 688981 T+0策略 · 实盘信号接入

---

## 目录

1. [架构说明](#1-架构说明)
2. [前置条件](#2-前置条件)
3. [pytdx 安装与验证](#3-pytdx-安装与验证)
4. [通达信客户端配置](#4-通达信客户端配置)
5. [config.yaml 配置](#5-configyaml-配置)
6. [⚠️ 关键 Bug：K线频率不匹配](#6️-关键-bugk线频率不匹配)
7. [修复方案](#7-修复方案)
8. [运行实盘信号循环](#8-运行实盘信号循环)
9. [连接远程通达信服务器](#9-连接远程通达信服务器)
10. [信号输出格式](#10-信号输出格式)
11. [常见问题排查](#11-常见问题排查)
12. [接入券商下单 API（进阶）](#12-接入券商下单-api进阶)

---

## 1. 架构说明

项目的通达信接入分两层：

```
通达信行情客户端（本地安装 or 远程服务器）
    │ TCP 7709端口（pytdx HQ协议）
    ▼
src/tdx/connector.py          ← 连接管理、心跳、断线重连
    │ LiveBarBuffer.update()
    ▼
src/tdx/live_trader.py        ← 每根K线触发特征计算 + 模型预测
    │ predictor.predict(window)
    ▼
结果记录到 results/live_signals.csv
    │ （live_mode=true时才模拟下单）
    ▼
券商下单API（预留接口，需自行接入）
```

**重要说明**：
- 默认 `live_mode: false`，只记录信号，**不操作资金**
- 真实下单需要接入券商 API（见第12节）
- 本接入属于行情订阅，不是通达信的 Level-2 接口

---

## 2. 前置条件

| 条件 | 要求 |
|------|------|
| Python | 3.8+ |
| 操作系统 | Windows（推荐）/ Linux / macOS |
| 通达信客户端 | 需要安装并**保持运行**（本地模式）|
| 网络 | 连接到通达信行情服务器（境内网络）|
| pytdx | `pip install pytdx` |

> **注意**：科创板股票（688981）行情在部分低版本通达信里有延迟，建议用最新版通达信或申万宏源、东方财富等大券商的通达信行情版本。

---

## 3. pytdx 安装与验证

### 安装

```bash
pip install pytdx
```

### 快速连接测试

```python
from pytdx.hq import TdxHq_API

api = TdxHq_API()

# 连接通达信行情（本地客户端）
api.connect("127.0.0.1", 7709)

# 获取 688981 最新5分钟K线（最近5根）
data = api.get_security_bars(
    category=0,    # 0=5分钟, 4=30分钟, 5=60分钟, 6=日线
    market=1,      # 1=上海, 0=深圳
    code="688981",
    start=0,       # 从最新开始
    count=5,
)
print(data)
api.disconnect()
```

### K线 category 对照表

| category | 周期 |
|----------|------|
| 0 | 5分钟 |
| 1 | 15分钟 |
| 2 | 30分钟 ← 本项目使用 |
| 3 | 60分钟 |
| 4 | 日线 |
| 5 | 周线 |
| 6 | 月线 |

---

## 4. 通达信客户端配置

### 本地模式（推荐测试用）

1. 打开通达信客户端，确保**已登录并持续运行**
2. 通达信默认在本地开放 `127.0.0.1:7709` 行情端口
3. 部分版本通达信需要手动开启本地服务：
   - 菜单 → 系统设置 → 网络通信设置 → 勾选"允许第三方接入"

### 验证端口是否开放

```bash
# Windows
netstat -ano | findstr 7709

# Linux/Mac
netstat -tlnp | grep 7709
# 或
nc -zv 127.0.0.1 7709
```

---

## 5. config.yaml 配置

在项目根目录的 `config.yaml` 中修改 `tdx` 部分：

```yaml
tdx:
  enabled: true                  # 开启通达信接入
  host: "127.0.0.1"             # 本地通达信客户端IP
  port: 7709                    # 默认行情端口
  heartbeat_interval: 30        # 心跳间隔（秒），防止断线
  reconnect_attempts: 3         # 断线重连次数
  symbol: "sh688981"            # 上交所科创板代码
  bar_frequency: 30             # ⚠️ 必须与项目训练数据一致：30分钟
  live_mode: false              # 测试阶段保持false，不操作资金
```

---

## 6. ⚠️ 关键 Bug：K线频率不匹配

**这是当前项目中最重要的待修复问题。**

### 问题描述

`src/tdx/connector.py` 中的 `TdxBar5Min` 类：

```python
# ❌ 现在的代码 - 硬编码了 category=0（5分钟K线）
raw = api.get_security_bars(
    category=0,    # 0 = 5分钟  ← 错误！
    market=mkt,
    code=code,
    start=0,
    count=min(count, 800),
)
```

但整个项目（training/backtest）使用的是 **30分钟K线**（`config.yaml: bar_frequency: 30`）。

**后果**：
- 拉回来的是5分钟K线数据
- 传给 `build_feature_df()` 之前**没有重采样**
- 所有特征（ATR14、MACD等）周期语义完全错误
- 模型预测的是5分钟趋势，但在30分钟尺度上执行信号 → 信号混乱

---

## 7. 修复方案

### 方案A（最简单）：直接改 connector.py

修改 `TdxBar5Min.fetch()` 方法，支持可配置频率：

```python
# src/tdx/connector.py 修改版

_CATEGORY_MAP = {
    5:  0,   # 5分钟
    15: 1,   # 15分钟
    30: 2,   # 30分钟  ← 本项目需要这个
    60: 3,   # 60分钟
}

class TdxBar30Min:  # 重命名，职责更清晰
    def __init__(self, connector: TdxConnector):
        self._conn = connector
        cfg = load_config().get("tdx", {})
        bar_freq = cfg.get("bar_frequency", 30)
        self._category = _CATEGORY_MAP.get(bar_freq, 2)  # 默认30分钟

    def fetch(self, symbol: str, count: int = 100) -> pd.DataFrame:
        self._conn.heartbeat()
        api = self._conn.api
        if api is None:
            return pd.DataFrame()

        mkt, code = _mkt_code(symbol)
        try:
            raw = api.get_security_bars(
                category=self._category,  # ✅ 从config读取
                market=mkt,
                code=code,
                start=0,
                count=min(count, 800),
            )
        except Exception as e:
            logger.error("获取K线失败: %s", e)
            return pd.DataFrame()

        # ... 后续处理与原来相同 ...
```

同步修改 `LiveBarBuffer.__init__()` 中的引用：
```python
# 原来
self._fetcher = TdxBar5Min(connector)
# 改为
self._fetcher = TdxBar30Min(connector)
```

### 方案B（更健壮）：实时重采样

如果要保留5分钟数据精度，可拉5分钟后重采样到30分钟：

```python
def _resample_to_30min(df_5min: pd.DataFrame) -> pd.DataFrame:
    """将5分钟K线聚合为30分钟K线"""
    return df_5min.resample("30T").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
        "amount": "sum",
    }).dropna()
```

**推荐方案A**，直接获取30分钟K线，避免聚合带来的误差。

---

## 8. 运行实盘信号循环

### 第一步：训练模型，获得 run_id

```bash
# 训练完整模型（需要Baostock历史数据）
python main.py --mode full
# 输出类似：run_id = 20260603_165651
```

### 第二步：开启通达信客户端

确保通达信行情客户端正在运行，且港股/A股行情已登录。

### 第三步：启动实时信号循环

```bash
python main.py --mode live --run_id 20260603_165651
```

### 预期输出

```
=== LiveTrader 启动 | symbol=sh688981 | live_mode=False ===
通达信连接成功: 127.0.0.1:7709
等待K线缓冲区预热... (15/50)
等待K线缓冲区预热... (30/50)
[09:30] ─ 观望  prob=0.421  
[10:00] ▲ 买入  prob=0.623  
[10:30] ─ 观望  prob=0.512  (分歧过滤)
[11:00] ─ 观望  prob=0.489  
```

### 信号说明

| 信号 | 含义 | 条件 |
|------|------|------|
| `▲ 买入 (1)` | 集成模型看涨 | prob > buy_threshold AND 3/4模型一致 |
| `─ 观望 (0)` | 无信号 | 概率居中 or 模型分歧 |
| `▼ 卖出 (-1)` | 集成模型看跌 | prob < sell_threshold |

---

## 9. 连接远程通达信服务器

如果不想在本地安装通达信客户端，可以直接连接通达信的公共行情服务器：

```python
# 可用的通达信公共服务器列表（可能随时变化）
SERVERS = [
    ("119.147.212.81",  7709),   # 广发行情
    ("218.80.248.229",  7709),   # 通达信上海
    ("202.108.253.130", 7709),   # 同花顺
    ("60.28.23.80",     7709),
]
```

修改 `config.yaml`：
```yaml
tdx:
  host: "119.147.212.81"   # 使用公共服务器
  port: 7709
```

> **注意**：
> - 公共服务器不稳定，断线较多，依赖 `reconnect_attempts` 重连
> - 部分服务器已停用，需要自行测试
> - 生产环境建议使用券商提供的专用行情服务器
> - 可用服务器列表可从 `pytdx.hq_utils` 的 `get_static_data()` 获取

### 查找可用服务器

```python
from pytdx.util.best_ip import search_best_ip

# 自动测试延迟并返回最优服务器（需要有网络连接）
best = search_best_ip()
print(best)
```

---

## 10. 信号输出格式

所有实盘信号自动保存到 `results/live_signals.csv`：

```
datetime,signal,prob,divergence,p_transformer_lstm,p_vanilla_lstm,p_cnn_lstm,p_mlp
2026-06-03 09:30:00+08:00,0,0.4218,True,0.3891,0.4502,0.4011,0.4472
2026-06-03 10:00:00+08:00,1,0.6234,False,0.6512,0.6001,0.5978,0.6445
```

字段说明：
- `signal`: 1=买入, 0=观望, -1=卖出
- `prob`: 集成加权概率
- `divergence`: True表示模型分歧（信号被过滤）
- `p_xxx`: 各子模型的原始概率

---

## 11. 常见问题排查

### Q1: `pytdx 未安装，通达信接口不可用`

```bash
pip install pytdx
# 如果pip安装失败，尝试
pip install pytdx --no-binary pytdx
```

### Q2: `连接失败 (attempt 1/3): [Errno 111] Connection refused`

- 检查通达信客户端是否正在运行
- 检查防火墙是否放行了 7709 端口
- 尝试 `telnet 127.0.0.1 7709` 验证端口开放
- 如果使用远程服务器，检查网络连通性

### Q3: 获取到的K线数据为空

```python
# 常见原因：非交易时段
# 通达信只在交易时间（09:30-15:00）提供实时数据
# 测试时可先获取历史K线验证连接
data = api.get_security_bars(category=2, market=1, code="688981", start=0, count=10)
print(len(data), "根K线")
```

### Q4: 心跳失败频繁重连

调整 `heartbeat_interval` 为更短的间隔（如10秒），或检查网络稳定性：

```yaml
tdx:
  heartbeat_interval: 10  # 更频繁的心跳
  reconnect_attempts: 5   # 更多重连次数
```

### Q5: 模型加载失败

```
FileNotFoundError: models/20260603_165651/transformer_lstm.pt
```

确保 `--run_id` 指向已经训练完成的目录：
```bash
ls results/          # 查看可用的run_id
ls models/           # 确认模型文件存在
```

### Q6: 缓冲区长时间无法预热（一直停在"等待缓冲区"）

原因：拉回来的K线不够（少于 `window_size + 20 = 50` 根）

- 当日开盘不久时属于正常（需要等待约25小时积累30分钟K线）
- 检查是否连接到了正确的行情服务器
- 尝试降低预热门槛：在 `live_trader.py` 中修改 `buf.is_ready(self._win)` 里的阈值

---

## 12. 接入券商下单 API（进阶）

当前 `live_trader.py` 的 `_simulate_order()` 只是记录日志，要真正下单需要接入券商接口。

### 选项A：EasyTrader（通用，免费）

适合个人账户，通过模拟输入指令控制通达信/同花顺客户端：

```bash
pip install easytrader
```

```python
# 在 _simulate_order() 中添加：
import easytrader

trader = easytrader.use("tdx")          # 通达信
trader.connect("c:/hmbz/TdxW.exe")      # 通达信客户端路径

def _simulate_order(self, ts, signal: int, prob: float) -> None:
    shares_to_trade = 1000  # 示例：每次交易1000股
    price = self._get_current_price()   # 获取当前价格
    
    if signal == 1:
        trader.buy("688981", price=price, amount=shares_to_trade)
    elif signal == -1:
        trader.sell("688981", price=price, amount=shares_to_trade)
```

> ⚠️ EasyTrader 通过模拟鼠标/键盘操作，需要窗口保持前台可见，不适合服务器部署。

### 选项B：XTP SDK（机构级）

需要向中泰证券申请开通 XTP 接口：

```python
# XTP SDK Python binding
import xtpwrapper as xtp
# 需要有效的账户和授权
```

### 选项C：QMT / miniQMT（个人投资者）

迅投 miniQMT 支持Python直接下单，适合个人：

```python
from xtquant import xttrader

xt_trader = xttrader.XtQuantTrader(path, session_id)
xt_trader.order_stock("688981", xttrader.XT_STOCK_BUY, 1000, xttrader.XT_PRICE_LATEST)
```

---

## 总结清单

- [ ] `pip install pytdx` 安装依赖
- [ ] 通达信客户端运行并开放 7709 端口
- [ ] `config.yaml` 中 `tdx.enabled: true`
- [ ] **修复 connector.py 中的 K线频率 Bug**（把 `category=0` 改为 `category=2`）
- [ ] `python main.py --mode full` 训练模型，记录 run_id
- [ ] `python main.py --mode live --run_id <your_run_id>` 启动信号循环
- [ ] 验证 `results/live_signals.csv` 输出是否正常
- [ ] 如需真实下单，接入券商 API（EasyTrader / XTP / miniQMT）

---

*本指南对应项目版本 v2.0（万2.5佣金 · 30分钟K线 · 四模型集成）*
