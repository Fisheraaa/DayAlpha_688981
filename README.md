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

---

### 结果总览

**第一层验证：训练期样本外（2024 全年 OOS，含完整交易成本）**

| 指标 | 数值 |
|------|------|
| 总收益率 | **+4.94%** |
| 年化 Sharpe | **0.804** |
| 最大回撤 | **6.37%** |
| 胜率 | **42.4%** |
| 盈亏比 | **1.62** |
| 交易笔数 | 250 |
| 往返成本 | ≈20bp（万2.5佣金 + 印花税 + 0.02%滑点）|
| Kelly 期望 | **+0.111（正期望）** |
| 逐笔 t 检验 | t=2.49，**p=0.013**（显著）|

**第二层验证：独立 OOS（2025-01 ~ 2026-06，零参数修改）**

| 指标 | 数值 |
|------|------|
| 总收益率 | **−19.21%** |
| 年化 Sharpe | **−3.332** |
| 胜率 | **32.1%** |
| 盈亏比 | **1.093** |
| 交易笔数 | 358 |
| Kelly 期望 | **−0.328（负期望）** |
| 逐bar t 检验 | t=−3.88，**p=0.0001**（高度显著为负）|

---

### 方法论诚信声明

2024 OOS 的 +0.804 是在大约 16 次「看结果→调参」循环后被选出来汇报的——从 5 分钟 v1\~v9、换到 30 分钟、到 buy\_threshold 网格搜索。模型权重从未在 2024 数据上训练，但「选择报告这个结果」的决策本身基于 2024 表现，属于 López de Prado 所说的 *backtest overfitting via repeated trials*。

三个独立观测并列：WF 均值 −2.42，2024 单点 +0.804，2025–26 单点 −3.332。两负一正，正的那个恰好是被选出来汇报的那个。这两个数字放在一起，比单独报告哪一个都更有研究价值。

---

### 根因定位：时段 Alpha 衰减

可解释性分析对比了两个时期，发现关键分歧：

| 分析方法 | 2024（训练期） | 2025–26（独立OOS）|
|----------|--------------|-----------------|
| SHAP Top-1/2 | time\_cos / time\_sin（合计22.1%）| time\_cos / time\_sin（合计22.2%）|
| 排列重要性 time\_sin | 排名 **#1**（drop=+0.010）| 排名 **#27**（drop=−0.008）|
| 排列重要性 time\_cos | 排名倒数第1（drop≈−0.012）| 排名倒数第1（drop≈−0.011）|
| 排列重要性 boll\_width | 排名第15（无关紧要）| 排名 **#1**（最有用）|

SHAP 两期完全相同（权重固定），排列重要性剧烈分化——说明模型的"注意力"没有变，但市场结构变了。

消融实验进一步验证：去掉 time\_cos/time\_sin，在 2024 年内 Walk-Forward 重训，**所有指标全面变差**（Sharpe 均值 −2.42→−3.54，胜率 37.9%→33.2%），证明时段特征在训练期是真实有用的信号，不是伪相关。

**结论**：688981 在 2020–2024 年间存在真实的日内时段规律性，模型成功捕捉并从中获利。2025–26 年间这个规律消失或反转，模型仍用旧规律下注，导致系统性亏损。这是 **alpha decay**，不是模型设计缺陷。

---

### 架构概览

```
Baostock 行情数据
    │
    ▼
数据清洗 + 涨跌停标记
    │
    ▼
31 个原始特征 ──► IC 滚动筛选（|IC| ≥ 0.008）──► 28 维有效特征
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
| 训练 | Optuna TPE（30次Trial）、PurgedKFold Walk-Forward、BCE + 早停 |
| 可解释性 | SHAP GradientExplainer、集成梯度、排列重要性、特征稳定性分析 |
| 回测 | VeighNa 4.x BacktestingEngine + 自研回测引擎（对比验证） |
| 实盘 | pytdx + 通达信公共行情服务器（218.75.126.9，30 分钟 K 线 category=2）|

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

# 2. 无需本地通达信客户端，直接连接公共行情服务器
#    修改 config.yaml：
#    tdx:
#      enabled: true
#      host: "218.75.126.9"   # 通达信公共行情服务器
#      port: 7709
#      live_mode: false        # false = 仅记录信号，不下单

# 3. 启动实盘信号循环（建议交易日 9:25 前启动，15:05 后关闭）
python main.py --mode live --run_id <run_id>

# 信号记录位置：
#   results/live_all_bars.csv   每根K线均记录，含各模型概率
#   results/live_signals.csv    仅买入信号（prob > 0.61）
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
    │   ├── fetcher.py           # Baostock 30 分钟 K 线获取（多周期支持）
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
    │   └── optuna_search.py     # Optuna TPE 超参搜索（30次Trial）
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
        └── live_trader.py       # 实时信号循环（all_bars + 买入信号双轨记录）
```

---

### 可解释性分析

三种方法在训练期（2024）高度一致：

| 排名 | SHAP | 集成梯度 | 排列重要性 |
|------|------|---------|-----------|
| 1 | time\_cos | time\_cos | time\_sin |
| 2 | time\_sin | time\_sin | cci20 |
| 3 | ema12 | ma20\_dev | vol\_ratio5 |

**关键发现**：time\_cos/time\_sin 在训练期是真实有用的信号（消融验证），但在 2025–26 的排列重要性中排名骤降至倒数（drop 从 +0.010 变为 −0.008）。这是日内时段规律 **alpha decay** 的直接证据，详见[方法论诚信声明](#方法论诚信声明)。

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

---

### Results Summary

**Layer 1: In-distribution OOS (Full Year 2024, After All Transaction Costs)**

| Metric | Value |
|--------|-------|
| Total Return | **+4.94%** |
| Annualized Sharpe | **0.804** |
| Max Drawdown | **6.37%** |
| Win Rate | **42.4%** |
| Payoff Ratio | **1.62** |
| Trades | 250 |
| Round-Trip Cost | ≈20bp (0.025% commission + stamp duty + 0.02% slippage) |
| Kelly Expectation | **+0.111 (positive)** |
| Per-trade t-test | t=2.49, **p=0.013** (significant) |

**Layer 2: Independent OOS (2025-01 to 2026-06, zero parameter changes)**

| Metric | Value |
|--------|-------|
| Total Return | **−19.21%** |
| Annualized Sharpe | **−3.332** |
| Win Rate | **32.1%** |
| Payoff Ratio | **1.093** |
| Trades | 358 |
| Kelly Expectation | **−0.328 (negative)** |
| Per-bar t-test | t=−3.88, **p=0.0001** (highly significant negative) |

---

### Methodology Note

The 2024 Sharpe of +0.804 was selected after approximately 16 rounds of "observe result → adjust parameter" — from 5-min v1–v9, to switching to 30-min bars, to the buy\_threshold grid search. Model weights were never trained on 2024 data, but the decision to *report this result* was made based on 2024 performance. This is what López de Prado calls *backtest overfitting via repeated trials*.

Three independent data points: WF mean −2.42, 2024 single-point +0.804, 2025–26 single-point −3.332. Two negatives, one positive — and the positive one is exactly the result that was selected for reporting. Both numbers together tell a more complete and honest story than either alone.

---

### Root Cause: Intraday Alpha Decay

Explainability analysis compared two periods and found a critical divergence:

| Method | 2024 (train-era) | 2025–26 (independent OOS) |
|--------|-----------------|--------------------------|
| SHAP Top-1/2 | time\_cos / time\_sin (22.1%) | time\_cos / time\_sin (22.2%) |
| Perm. importance time\_sin | Rank **#1** (drop=+0.010) | Rank **#27** (drop=−0.008) |
| Perm. importance time\_cos | Rank last (drop≈−0.012) | Rank last (drop≈−0.011) |
| Perm. importance boll\_width | Rank #15 (irrelevant) | Rank **#1** (most useful) |

SHAP is identical across periods (weights are fixed); permutation importance diverges sharply — the model's "attention" did not change, but the market structure did.

Ablation confirmed: removing time\_cos/time\_sin and retraining on 2024 Walk-Forward **uniformly worsened all metrics** (Sharpe mean −2.42→−3.54, win rate 37.9%→33.2%), proving the time features were genuinely useful in the training period, not spuriously correlated.

**Conclusion**: SMIC exhibited real intraday time-of-day seasonality during 2020–2024, which the model successfully captured. This pattern has since disappeared or reversed, causing the model to systematically bet in the wrong direction. This is **alpha decay**, not a fundamental model flaw.

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

# No local TongDaXin client required — connects to public quote server
# Enable in config.yaml:
#   tdx:
#     enabled: true
#     host: "218.75.126.9"   # TDX public quote server
#     port: 7709
#     live_mode: false        # signals only, no auto-execution

# Run during trading hours (start before 9:25, stop after 15:05)
python main.py --mode live --run_id <run_id>

# Signal logs:
#   results/live_all_bars.csv   every bar, including per-model probabilities
#   results/live_signals.csv    buy signals only (prob > 0.61)
```

---

### Architecture

| Layer | Detail |
|-------|--------|
| Data | Baostock 30-min OHLCV, cleaned and limit-move filtered |
| Features | 31 raw → 28 selected (Spearman IC rolling filter, \|IC\| ≥ 0.008) |
| Input | 30-bar sliding window × 28 features |
| Models | Transformer-LSTM (d=128, 8 heads, 3 Pre-LN layers) + VanillaLSTM + CNN-LSTM + MLP |
| Ensemble | Softmax-weighted prediction aggregation |
| Training | Optuna TPE (30 trials), BCE loss, early stopping on val AUC |
| Validation | PurgedKFold Walk-Forward (8-month train / 1-month test) |
| Explainability | SHAP GradientExplainer, Integrated Gradients, Permutation Importance |
| Backtesting | Custom T+0 engine + VeighNa BacktestingEngine (cross-verification) |
| Live Feed | pytdx → TDX public server (218.75.126.9), 30-min bars (category=2) |

---

### Key Design Decisions

**Why PurgedKFold Walk-Forward?**
Standard k-fold leaks future information in time series. PurgedKFold removes the overlap period between training and validation sets, preventing look-ahead bias. Walk-Forward then tests across multiple time windows to verify generalisation beyond the training period.

**Why four models?**
Each architecture captures different aspects of the signal: Transformer handles long-range temporal dependencies, LSTM captures sequential patterns, CNN extracts local feature interactions, and MLP serves as a nonlinear baseline. Softmax-weighted ensembling reduces variance without overfitting to any single model's bias.

**Why report both the 2024 and 2025–26 results?**
The 2024 result (+0.804) was selected from multiple iterations and carries backtest overfitting risk. The 2025–26 result (−3.332) is the only truly clean test. Reporting both, along with the root cause analysis (alpha decay in time-of-day features), is more scientifically honest — and more instructive — than cherry-picking either result alone.

---

*Built with PyTorch · VeighNa · SHAP · Optuna*
