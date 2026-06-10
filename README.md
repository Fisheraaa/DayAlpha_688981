<div align="center">

# DayAlpha (688981)

**A股日内机器学习量化交易策略 · Intraday ML Quantitative Strategy**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![VeighNa](https://img.shields.io/badge/VeighNa-4.x-blue?style=flat)](https://github.com/vnpy/vnpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

<br/>

**[ [中文](#中文) · [English](#english) ]**

</div>

---

## 中文

### 项目简介

基于四模型集成的中芯国际（688981）T+0 日内量化策略，覆盖完整研究链路：数据构建 → IC 特征选择 → 深度学习集成模型 → PurgedKFold Walk-Forward 验证 → SHAP 可解释性 → VeighNa 平台回测 → 通达信实盘行情接入。

**OOS 样本外结果（2024 全年，含完整交易成本）**

| 指标 | 数值 |
|------|------|
| 总收益率 | **+4.94%** |
| 年化 Sharpe | **0.804** |
| 最大回撤 | **6.37%** |
| 胜率 | **42.4%** |
| 盈亏比 | **1.62** |
| 往返成本 | ≈20bp（万2.5佣金 + 印花税 + 0.02%滑点）|
| Kelly 期望 | **正** |

> **局限性说明**：OOS 仅一年（125 笔来回），Sharpe t 检验 p ≈ 0.22，统计显著性不足；时间编码特征为主要 alpha 来源，下一步方向为接入 Level-2 数据与多标的截面化。

---

### 架构概览

```
Baostock 行情数据
    │
    ▼
数据清洗 + 涨跌停标记
    │
    ▼
27 个原始特征 ──► IC 滚动筛选（|IC| ≥ 0.008）──► 28 维有效特征
    │
    ▼
滑动窗口序列（30 根 × 28 维）
    │
    ├── Transformer-LSTM（主模型）
    ├── VanillaLSTM + 注意力
    ├── CNN-LSTM + SE 通道注意力
    └── MLP + 残差块
         │
         ▼
    Softmax 加权集成
         │
         ▼
    买入/观望/卖出 信号
         │
    ┌────┴─────┐
    │          │
回测引擎     实盘信号
(VeighNa)  (通达信)
```

---

### 技术栈

| 模块 | 技术 |
|------|------|
| 数据 | Baostock API、pandas、pytdx |
| 模型 | PyTorch 2.0+、TransformerEncoder（Pre-LN）、双向 LSTM |
| 训练 | Optuna TPE（50次Trial）、PurgedKFold Walk-Forward、BCE + CosineAnnealingWarmRestarts |
| 可解释性 | SHAP GradientExplainer、集成梯度、排列重要性、特征稳定性分析 |
| 回测 | VeighNa 4.x BacktestingEngine + 自研回测引擎（对比验证） |
| 实盘 | pytdx + 通达信 30 分钟行情（category=2） |

---

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt
pip install vnpy vnpy_ctastrategy  # VeighNa 回测（可选）
pip install shap                   # 可解释性分析（可选）

# 完整训练 + 回测 + 可解释性（全流程）
python main.py --mode full

# 仅 Walk-Forward 验证
python main.py --mode walk_forward

# Walk-Forward + Optuna 超参搜索
python main.py --mode walk_forward --optuna

# 单独运行可解释性分析
python main.py --mode explain --run_id <run_id>

# 生成可视化 HTML 报告（含 SHAP 图表、净值曲线、交易分析）
python report_generator.py --run_id <run_id>

# VeighNa 框架回测（对比验证）
python vnpy_integration/run_backtest.py --run_id <run_id>
```

**实盘信号接入（通达信）**

```bash
# 1. 安装依赖
pip install pytdx

# 2. 开启通达信客户端（需本地运行，默认端口 7709）
# 3. 修改 config.yaml
#    tdx:
#      enabled: true
#      host: "127.0.0.1"
#      port: 7709
#      live_mode: false   # 先用 false 测试，确认信号正常再改 true

# 4. 训练模型
python main.py --mode full

# 5. 启动实盘信号循环（仅信号，不自动下单）
python main.py --mode live --run_id <run_id>
```

---

### 项目结构

```
DayAlpha/
├── config.yaml                  # 所有参数（万2.5佣金、30分钟K线、阈值等）
├── main.py                      # 主入口（full / walk_forward / explain / live）
├── report_generator.py          # 可视化 HTML 报告生成（含 SHAP 图、中英切换）
├── requirements.txt
├── vnpy_integration/            # VeighNa 平台层（不修改原有代码）
│   ├── ml_cta_strategy.py       # CtaTemplate 策略包装
│   ├── data_loader.py           # Parquet → VeighNa BarData
│   └── run_backtest.py          # BacktestingEngine 回测入口
└── src/
    ├── data/
    │   ├── fetcher.py           # Baostock 30 分钟 K 线获取
    │   └── processor.py         # 数据清洗、涨跌停标记
    ├── features/
    │   ├── technical.py         # 16 个技术指标（MA/EMA/Bollinger/RSI/MACD/ATR/ADX/KAMA 等）
    │   ├── volume.py            # 量价微结构 + VWAP + 时间编码
    │   ├── dataset.py           # 滑动窗口序列、标签构建（T0Dataset）
    │   ├── selector.py          # Spearman IC 滚动特征选择
    │   └── base.py              # BaseFeature 抽象类
    ├── models/
    │   ├── transformer_lstm.py  # 主模型（d_model=128, nhead=8, 3层 Pre-LN）
    │   ├── baselines.py         # VanillaLSTM、CNN-LSTM（SE注意力）、MLP
    │   └── ensemble.py          # Softmax 加权集成预测器
    ├── training/
    │   ├── trainer.py           # 单模型训练（早停 + CosineWarmRestart）
    │   └── optuna_search.py     # Optuna TPE 超参搜索（50次Trial）
    ├── backtest/
    │   ├── engine.py            # T+0 回测引擎（动态仓位 + 三层止损）
    │   ├── costs.py             # 成本模型（佣金 + 印花税 + 经手费 + 滑点）
    │   ├── metrics.py           # 绩效指标（Sharpe/Sortino/Omega/Calmar/信息比率）
    │   └── walk_forward.py      # 滚动窗口验证（断点续传）
    ├── explain/
    │   ├── shap_analysis.py     # SHAP GradientExplainer + 排列重要性 + 特征稳定性
    │   └── gradient.py          # 集成梯度（Integrated Gradients）
    └── tdx/
        ├── connector.py         # 通达信连接（心跳/断线重连，30min K线 category=2）
        └── live_trader.py       # 实时信号循环
```

---

### 可解释性分析结果

三种方法高度一致，揭示核心 alpha 来源：

| 排名 | SHAP | 集成梯度 | 排列重要性 |
|------|------|---------|-----------|
| 1 | time_cos | time_cos | time_sin |
| 2 | time_sin | time_sin | cci20 |
| 3 | ema12 | ma20_dev | vol_ratio5 |

**核心洞察**：日内时间编码特征（time_cos/time_sin，代表一天中的时刻）稳居第一，说明 688981 存在显著的日内时段规律性。

---

### 成本模型

```
往返交易总成本 ≈ 20bp

  买入成本：万2.5佣金（0.025%）+ 经手费（0.00487%）+ 滑点（0.02%）
  卖出成本：万2.5佣金（0.025%）+ 印花税（0.1%）+ 经手费（0.00487%）+ 滑点（0.02%）
  ─────────────────────────────────────────────────────────────────
  合计：约 0.1997%，最低盈利阈值约 0.2047%
```

---

<br/>

## English

### Overview

DayAlpha is an intraday T+0 quantitative trading strategy for SMIC (688981) built on a four-model ML ensemble. It covers the full research pipeline: data construction → IC-based feature selection → deep learning ensemble → PurgedKFold Walk-Forward validation → SHAP explainability → VeighNa backtesting → TongDaXin live data feed.

**Out-of-Sample Results (Full Year 2024, After All Transaction Costs)**

| Metric | Value |
|--------|-------|
| Total Return | **+4.94%** |
| Annualized Sharpe | **0.804** |
| Max Drawdown | **6.37%** |
| Win Rate | **42.4%** |
| Payoff Ratio | **1.62** |
| Round-Trip Cost | ≈20bp (0.025% commission + stamp duty + 0.02% slippage) |
| Kelly Expectation | **Positive** |

> **Limitations**: OOS covers only one year (125 round trips). Sharpe t-test yields p ≈ 0.22, so statistical significance is not established. Time-of-day encoding features are the dominant alpha source. Future work includes Level-2 order book data and cross-sectional portfolio construction.

---

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install vnpy vnpy_ctastrategy  # VeighNa backtesting (optional)
pip install shap                   # Explainability (optional)

# Full pipeline: train → backtest → explainability
python main.py --mode full

# Walk-Forward validation only
python main.py --mode walk_forward

# Run explainability analysis on an existing run
python main.py --mode explain --run_id <run_id>

# Generate HTML report (equity curve, SHAP charts, trade analysis)
python report_generator.py --run_id <run_id>

# VeighNa framework backtest (cross-validation against custom engine)
python vnpy_integration/run_backtest.py --run_id <run_id>
```

**Live Signal Feed (TongDaXin)**

```bash
pip install pytdx

# Enable in config.yaml:
#   tdx:
#     enabled: true
#     host: "127.0.0.1"
#     port: 7709
#     live_mode: false   # signals only, no auto-execution

python main.py --mode live --run_id <run_id>
```

---

### Architecture

| Layer | Detail |
|-------|--------|
| Data | Baostock 30-min OHLCV, cleaned and limit-move filtered |
| Features | 27 raw → 28 selected (Spearman IC rolling filter, \|IC\| ≥ 0.008) |
| Input | 30-bar sliding window × 28 features |
| Models | Transformer-LSTM (d=128, 8 heads, 3 Pre-LN layers) + VanillaLSTM + CNN-LSTM + MLP |
| Ensemble | Softmax-weighted prediction aggregation |
| Training | Optuna TPE (50 trials), BCE loss, CosineAnnealingWarmRestarts, early stopping on val AUC |
| Validation | PurgedKFold Walk-Forward (8-month train / 1-month test) |
| Explainability | SHAP GradientExplainer, Integrated Gradients, Permutation Importance |
| Backtesting | Custom T+0 engine + VeighNa BacktestingEngine (cross-verification) |
| Live Feed | pytdx → TongDaXin 30-min bars (category=2) |

---

### Key Design Decisions

**Why PurgedKFold Walk-Forward?**
Standard k-fold leaks future information in time series. PurgedKFold removes the overlap period between training and validation sets, preventing look-ahead bias. Walk-Forward then tests across multiple time windows to verify the strategy generalises beyond the training period.

**Why four models?**
Each architecture captures different aspects of the signal: Transformer handles long-range temporal dependencies, LSTM captures sequential patterns, CNN extracts local feature interactions, and MLP serves as a nonlinear baseline. Softmax-weighted ensembling reduces variance without overfitting to any single model's bias.

**Why time encoding features dominate?**
SHAP, IG, and permutation importance all rank `time_cos`/`time_sin` first. This points to a clear intraday seasonality: SMIC exhibits distinct trading patterns at different times of day, likely driven by institutional order flow and market microstructure. This is a known alpha factor, though its persistence depends on market regime stability.

---

*Built with PyTorch · VeighNa · SHAP · Optuna*
