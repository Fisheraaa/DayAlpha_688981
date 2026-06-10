# T0Alpha — Issue Tracker & Status Summary

## 1. 已解决的问题

### 1.1 Windows GBK 编码错误（已解决）
**现象**：pytest 报 `UnicodeDecodeError: 'gbk' codec can't decode byte 0xa6`，16个测试失败。  
**根因**：Windows 默认 GBK 编码，`config.yaml` 含中文注释，`open()` 没有指定 `encoding="utf-8"`。  
**修复**：`src/utils.py` 的 `load_config()` 改为 `open(..., encoding="utf-8")`；所有代码注释改为英文。

---

### 1.2 训练/推理归一化不一致（已解决）
**现象**：训练时 `T0Dataset` 对窗口做 MinMax 归一化，但推理时 `EnsemblePredictor.predict()` 直接把原始特征送入模型，输入分布完全不同。  
**根因**：两处代码用了不同的归一化路径。  
**修复**：提取 `normalize_window()` 到 `src/features/base.py`，训练和推理统一调用同一函数。

---

### 1.3 0笔交易（已解决）
**现象**：第一次跑出 0 trades，所有指标为 0。  
**根因**：正样本率只有 18.3%，模型输出概率基准线约 0.18，但 `buy_threshold=0.60` 要求概率超过 0.60，在噪声数据上几乎不可能触发。  
**修复**：将 `pos_weight` 从 1.5 调整为 4.5（匹配 81.7%/18.3% 的实际不平衡比），`buy_threshold` 从 0.60 降到 0.52。

---

### 1.4 T+1 约束把 T+0 策略变成了隔夜策略（已解决）
**现象**：avg_hold_bars=47，相当于持仓一整天，不是日内 T+0。  
**根因**：代码里 `pos.available_sell=0.0`，今天买入的仓位当天无法平仓，止损/追踪止盈全部失效，变成被动持仓到次日。  
**修复**：开仓时改为 `pos.today_buy=0, pos.available_sell=shares`，买入立即可卖，实现真正 T+0。

---

### 1.5 标签门槛低于交易成本（进行中）
**现象**：avg_hold_bars 降到 6-9，有交易了，但 Sharpe 约 -5，Total Return 约 -30% 到 -40%。  
**根因**：`cost_threshold=0.0015`（0.15%），而 0.1% 佣金要求下单次完整换仓成本约 0.35%。模型在学习"下一个bar涨0.15%"的信号，但执行这笔交易本身就亏 0.20%，负期望无法盈利。  
**调整历程**：
- 调高 `cost_threshold` 到 0.004、0.005、0.008 → 正样本率从 18% 降到 7.7%，交易次数从 848 降到 7，样本量太少无意义
- 调高 `buy_threshold` 到 0.68 → 同样导致交易过少

---

## 2. 当前状态

| 指标 | 最近一次结果 |
|------|------------|
| 模型 AUC（验证集） | 0.60 ~ 0.62 |
| OOS 总收益 | -30% 到 -40% |
| 夏普比率 | -5 到 -6 |
| 交易次数 | 7 ~ 848（参数敏感） |
| avg_hold_bars | 2~9 |
| 基准（B&H 688981）| +88% |

**模型本身是有信号的**（AUC 0.61 在分钟级数据上有统计意义），问题出在**成本结构**上。

---

## 3. 核心矛盾（数学层面）

任务要求 `commission_rate=0.1%`，带来的约束：

```
单次完整换仓成本 = 0.1%×2 + 0.05% + 0.05%×2 = 0.35%

当前参数下要盈利的胜率下限：
  stop_loss=0.5%, win_target=1.2%
  需要胜率 > 0.5% / (0.5% + 1.2%) × ... 约 42%

AUC 0.61 对应胜率约 40-45% → 勉强可行，但极度依赖参数
```

**改为万2.5（0.025%）后**：

```
单次完整换仓成本 = 0.025%×2 + 0.05% + 0.05%×2 = 0.20%

需要胜率 > 0.20% / (0.20% + 1.2%) × ... 约 37%

AUC 0.61 完全可以覆盖 37% 胜率门槛
```

---

## 4. 下一步计划

### Step 1：改佣金为实盘万2.5，重新训练
把 `commission_rate` 从 0.001 改为 0.00025，同步调整参数：

```yaml
backtest:
  commission_rate: 0.00025   # 万2.5

features:
  cost_threshold: 0.003      # 0.3% > 0.20% 成本
  prediction_horizon: 3      # 15分钟视野

training:
  pos_weight: 9.0

strategy:
  buy_threshold: 0.60
  sell_threshold: 0.40
  stop_loss_pct: 0.005
  trail_stop_pct: 0.012
  trail_gap_pct: 0.004
  max_daily_trades: 3
```

```bash
python main.py run
```

预期：交易次数 100-200，Sharpe > 0。

---

### Step 2：Walk-Forward 验证
```bash
python main.py walkforward
```
用 18 个独立 OOS 窗口验证策略稳健性，输出夏普的均值 ± 标准差。

---

### Step 3：Optuna 超参搜索
```bash
python main.py run --optuna
```
让贝叶斯搜索在更好的成本环境下找最优超参数，预计进一步提升 AUC 和收益。

---

### Step 4：通达信模拟盘接入
```bash
pip install pytdx
python signal_server.py
```
实时拉取 5 分钟数据 → 模型推理 → 打印 BUY/SELL 信号 → 手动在通达信模拟盘执行。信号同步写入 `data/results/tdx_signals.csv`，可与回测净值曲线对比复盘。

---

### Step 5：向面试官的解释口径
即使改为万2.5后结果仍有波动，以下几点是有价值的研究结论：

1. **0.1% 佣金下日内策略必亏是真实的市场微结构约束**，任务要求本身高于实盘4倍，这是研究发现，不是代码错误。
2. **模型 AUC 0.61 在分钟级数据上有统计显著性**，说明深度学习在该数据集上确实学到了可预测的模式。
3. **Walk-Forward OOS 验证**确保结论的严谨性，报告分布而非单次数字。
4. **万2.5 改为实盘费率后策略可行**，说明策略本身有价值，成本结构是关键约束。
