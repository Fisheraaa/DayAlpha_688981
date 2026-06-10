# 688981 中芯国际 T+0 策略 (优化版 v2.0)

## 主要改动摘要

### 🔥 成本调整（最核心）
| 参数 | 旧版 | 新版 | 说明 |
|------|------|------|------|
| 佣金率 | 万10 (0.001) | **万2.5 (0.00025)** | 面试官给出的新条件 |
| 往返总成本 | ≈0.31% | **≈0.20%** | 降低约35% |
| 标签阈值 `cost_threshold` | 0.003 | **0.0018** | 与新成本匹配，正样本率提高 |

### 📈 策略参数优化
| 参数 | 旧版 | 新版 | 原因 |
|------|------|------|------|
| `buy_threshold` | 0.60 | **0.58** | 成本低，可适当放宽入场条件 |
| `max_daily_trades` | 2 | **3** | 轮动机会更多 |
| `stop_loss_pct` | 1.5% | **1.2%** | 更快截损，减少单笔损失 |
| `trail_stop_pct` | 1.5% | **1.0%** | 更早锁定利润 |
| `trail_gap_pct` | 0.8% | **0.6%** | 追踪更紧 |
| `prediction_horizon` | 1根 | **3根 (15min)** | 减少噪音，提升预测质量 |
| 时间止损 | 6根 | **8根** | 配合15分钟预测 |

### 🧠 模型升级
| 参数 | 旧版 | 新版 |
|------|------|------|
| `d_model` | 64 | **128** |
| `nhead` | 4 | **8** |
| Transformer 层数 | 2 | **3** (Pre-LN) |
| `lstm_hidden` | 64 | **128** |
| 特征数量 | 27 | **31** (+ADX14, KAMA, DI差, 枢轴点) |
| 集成权重 | AUC差值归一化 | **Softmax归一化** |

### 🔌 通达信模拟盘接入（新增）
```bash
# 1. 安装依赖
pip install pytdx

# 2. 在 config.yaml 开启
# tdx:
#   enabled: true
#   host: "127.0.0.1"
#   port: 7709

# 3. 先训练模型得到 run_id
python main.py --mode full

# 4. 启动实盘信号循环
python main.py --mode live --run_id 20260531_120000
```

## 快速开始

```bash
pip install -r requirements.txt

# 完整训练 + 回测
python main.py --mode full

# Walk-Forward 验证
python main.py --mode walk_forward

# Walk-Forward + Optuna 超参搜索
python main.py --mode walk_forward --optuna
```

## 项目结构

```
project_optimized/
├── config.yaml              # 所有参数配置（含万2.5佣金）
├── main.py                  # 主入口
├── requirements.txt
└── src/
    ├── utils.py             # 路径/配置/日志
    ├── data/
    │   ├── fetcher.py       # Baostock 5分钟K线获取
    │   └── processor.py     # 数据清洗、涨跌停标记
    ├── features/
    │   ├── technical.py     # 16个技术指标（+4新增）
    │   ├── volume.py        # 量价微结构+时间编码（15个）
    │   ├── dataset.py       # 特征矩阵、标签构建
    │   ├── selector.py      # IC特征选择
    │   └── base.py          # BaseFeature抽象类
    ├── models/
    │   ├── transformer_lstm.py  # 主模型（128d, 8head, 3层）
    │   ├── baselines.py         # VanillaLSTM+注意力, CNN+SE, MLP+残差
    │   └── ensemble.py          # 集成预测器（Softmax权重）
    ├── training/
    │   ├── trainer.py           # 单模型训练（CosineWarmRestart调度）
    │   └── optuna_search.py     # 50次Optuna搜索
    ├── backtest/
    │   ├── engine.py            # 回测引擎（T+0、动态仓位）
    │   ├── costs.py             # 成本模型（万2.5佣金）
    │   ├── metrics.py           # 绩效指标（+Omega, 信息比率）
    │   └── walk_forward.py      # 滚动验证
    ├── explain/
    │   ├── shap_analysis.py     # SHAP分析
    │   └── gradient.py          # 集成梯度
    └── tdx/                     # 通达信接口（新增）
        ├── connector.py         # 连接/心跳/断线重连
        └── live_trader.py       # 实时信号循环
```

## 成本变化对策略的影响

```
旧成本（万10）往返≈0.31%
  → 需要较高阈值 buy_threshold=0.60
  → 每日2次交易，避免频繁交易
  → 标签阈值 0.003 才能覆盖成本

新成本（万2.5）往返≈0.20%
  → 可以适当放宽入场 buy_threshold=0.58
  → 每日3次交易机会
  → 标签阈值降至 0.0018，正样本更多，模型有更多可学习信号
  → 止损更紧（1.2%），因为每次入场需要覆盖的成本更少
```
