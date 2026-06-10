# 需求满足度分析报告
## 688981 T+0 策略 vs 原始需求对照

---

## ① 数据构建

| 要求 | 状态 | 实现位置 | 备注 |
|------|------|----------|------|
| 构建多类技术指标 | ✅ | `src/features/technical.py` | 16个：MA/EMA/Bollinger/RSI/MACD/ROC/ATR/Stoch/CCI/Williams/ADX/KAMA/DI差/枢轴点 |
| 自动化特征工程 | ✅ | `src/features/dataset.py` `build_feature_df()` | 全量31个特征自动计算 |
| 提取时序特征 | ✅ | `src/features/volume.py` | 量价微结构(4) + VWAP + 时间编码(3) |
| IC特征选择 | ✅ | `src/features/selector.py` | Spearman IC滚动计算，\|IC\|≥0.008过滤 |
| 滚动时间窗口 | ✅ | `src/features/dataset.py` `T0Dataset` | window_size=30根K线 |
| 历史序列作为模型输入 | ✅ | `src/features/dataset.py` | shape=(batch, 30, n_features) |

**数据构建：满分 ✅**

---

## ② 模型构建

| 要求 | 状态 | 实现位置 | 备注 |
|------|------|----------|------|
| 混合Transformer-LSTM模型 | ✅ | `src/models/transformer_lstm.py` | d_model=128, nhead=8, 3层Pre-LN + 双向LSTM |
| 自注意力机制 | ✅ | `TransformerEncoder` | MultiheadAttention内置 |
| 时序建模能力 | ✅ | `nn.LSTM` + 双向 | 捕捉长短期时序依赖 |
| 多模型集成系统 | ✅ | `src/models/ensemble.py` | 4个模型：TransformerLSTM / VanillaLSTM / CNN_LSTM / MLP |
| 普通LSTM | ✅ | `src/models/baselines.py` `VanillaLSTM` | 双向LSTM + 自注意力聚合 |
| CNN | ✅ | `src/models/baselines.py` `CNN_LSTM` | 多尺度CNN(3/5/7) + SE通道注意力 + LSTM |
| MLP | ✅ | `src/models/baselines.py` `MLP` | 4层 + 残差块 |
| 残差连接 | ✅ | 所有模型 | TransformerLSTM有LSTM残差投影；MLP有两级残差块 |
| 批量归一化 | ✅ | 所有模型 | `nn.BatchNorm1d` 在输出层前 |
| Dropout层 | ✅ | 所有模型 | dropout=0.17（可配置） |
| Xavier初始化 | ✅ | `_init_weights()` | `nn.init.xavier_uniform_` 用于Linear层 |
| 正交初始化 | ✅ | `_init_weights()` | `nn.init.orthogonal_` 用于LSTM hidden权重 |

**模型构建：满分 ✅**

---

## ③ 模型训练

| 要求 | 状态 | 实现位置 | 备注 |
|------|------|----------|------|
| 时间序列交叉验证 | ✅ | `src/backtest/walk_forward.py` | 滚动窗口：train 8个月 → test 1个月 |
| Optuna超参优化 | ✅ | `src/training/optuna_search.py` | TPE采样 + Median剪枝，30次trial |
| 学习率搜索 | ✅ | Optuna | 范围 1e-4 ~ 5e-3（log scale） |
| LSTM单元数搜索 | ✅ | Optuna | [64, 128, 192] |
| Dropout率搜索 | ✅ | Optuna | 0.1 ~ 0.4 |
| 早停机制 | ✅ | `src/training/trainer.py` | patience=40，监控val AUC |
| 监控验证集AUC | ✅ | `trainer.py` | `roc_auc_score`，best_state保存最优 |
| BCE损失函数 | ✅ | `trainer.py` | `BCEWithLogitsLoss` + pos_weight |
| Adam优化器 | ✅ | `trainer.py` | `torch.optim.Adam` + weight_decay=1e-5 |
| 学习率调度 | ✅（额外） | `trainer.py` | CosineAnnealingWarmRestarts（T_0=25, T_mult=2） |

**模型训练：满分 ✅**

---

## ④ 模型可解释性分析

| 要求 | 状态 | 实现位置 | 备注 |
|------|------|----------|------|
| 集成SHAP分析 | ⚠️ | `src/explain/shap_analysis.py` | 代码存在，但**main.py中没有调用入口** |
| 可视化特征影响 | ⚠️ | `plot_global_importance()` `plot_temporal_heatmap()` | 函数存在，但不自动生成 |
| 梯度重要性分析 | ⚠️ | `src/explain/gradient.py` `integrated_gradients()` | 代码存在，主流程未调用 |
| 识别关键预测因子 | ⚠️ | `ig_global_importance()` | 同上 |
| 计算排列重要性 | ⚠️ | `permutation_importance()` | 代码完整，主流程未调用 |
| 评估特征稳定性 | ⚠️ | `feature_stability_analysis()` | 代码完整，主流程未调用 |

**可解释性分析：代码实现 ✅ 但集成不完整 ⚠️**

> **缺口说明**：`src/explain/` 下的所有函数均已实现，质量也很高。但 `main.py --mode full` 结束后**不会自动调用** SHAP / 梯度分析 / 排列重要性，也不会生成任何可视化图表。从考核角度看，没有实际运行输出等于没有做。

**修复方案**：在 `mode_full()` 末尾添加如下代码：

```python
# main.py mode_full() 末尾追加
from src.explain.shap_analysis import (
    compute_shap_values, global_importance, plot_global_importance,
    plot_temporal_heatmap, feature_stability_analysis, permutation_importance
)
from src.explain.gradient import (
    batch_integrated_gradients, ig_global_importance, extract_attention_weights
)

logger.info("Step 5: 可解释性分析...")
explain_dir = out_dir / "explain"
explain_dir.mkdir(exist_ok=True)

# 取OOS前500个样本做SHAP（SHAP计算慢，不用全量）
oos_ds = T0Dataset(feature_df[oos_mask], label[oos_mask],
                   limit_mask[oos_mask], selected_features=selected)
n_explain = min(500, len(oos_ds))
X_bg  = oos_ds.X[:100]     # SHAP背景集
X_exp = oos_ds.X[:n_explain]
y_exp = oos_ds.y[:n_explain]

# 用主模型做SHAP
main_model = predictor.models["transformer_lstm"]

# SHAP
shap_vals = compute_shap_values(main_model, X_bg, X_exp[:50], device=device)
imp_df    = global_importance(shap_vals, feature_names=selected)
plot_global_importance(imp_df, save_path=explain_dir / "shap_importance.png")
plot_temporal_heatmap(shap_vals, feature_names=selected,
                      save_path=explain_dir / "shap_heatmap.png")
imp_df.to_csv(explain_dir / "shap_importance.csv", index=False)
logger.info("SHAP分析完成 → %s", explain_dir)

# 集成梯度
ig_attrs = batch_integrated_gradients(main_model, X_exp[:100], device=device)
ig_imp   = ig_global_importance(ig_attrs, feature_names=selected)
ig_imp.to_csv(explain_dir / "ig_importance.csv", index=False)

# 排列重要性
perm_imp = permutation_importance(main_model, X_exp, y_exp, device=device,
                                  feature_names=selected)
perm_imp.to_csv(explain_dir / "perm_importance.csv", index=False)

# 特征稳定性
stability = feature_stability_analysis(main_model, X_exp, y_exp, device=device)
stability.to_csv(explain_dir / "feature_stability.csv", index=False)
logger.info("可解释性分析全部完成 → %s", explain_dir)
```

---

## ⑤ 回测和风险管理

| 要求 | 状态 | 实现位置 | 备注 |
|------|------|----------|------|
| 滚动窗口回测系统 | ✅ | `src/backtest/walk_forward.py` | 带断点续传，train 8月/test 1月 |
| 模拟真实交易环境 | ✅ | `src/backtest/engine.py` | T+0、日内交易次数限制、涨跌停过滤 |
| 交易成本模型 | ⚠️ | `src/backtest/costs.py` | 实现完整，但**成本参数与原始要求不符**（见下） |
| 夏普比率 | ✅ | `src/backtest/metrics.py` | 年化，252×8根/年 |
| 索提诺比率 | ✅ | `src/backtest/metrics.py` | 仅下行波动率 |
| 最大回撤 | ✅ | `src/backtest/metrics.py` | 基于净值曲线 |
| Calmar比率 | ✅ | `src/backtest/metrics.py` | 年化收益/最大回撤 |
| 动态仓位管理 | ✅ | `engine.py` `size_pct = min(0.20 + conf * 0.20, 0.40)` | 置信度映射到20%~40%仓位 |
| 止损机制 | ✅ | `engine.py` | 固定止损(1.2%) + 追踪止损(1.0%激活,0.6%间距) + 时间止损 |
| Omega比率 | ✅（额外） | `metrics.py` | 额外添加 |
| 信息比率 | ✅（额外） | `metrics.py` | 相对基准超额收益/跟踪误差 |

### ⚠️ 成本参数差异

| 参数 | 原始需求 | 当前实现 | 差异 |
|------|----------|----------|------|
| 佣金率 | **0.1%** (万10) | **0.025%** (万2.5) | 面试官给出的更新条件 |
| 滑点 | **0.05%** | **0.02%** | 同上 |
| 往返总成本 | ≈0.31% | ≈0.20% | -35% |

> **结论**：从技术实现角度成本模型完整正确，但若对照原始需求文字（0.1%佣金 / 0.05%滑点）则存在数值差异。这是**已知且有意为之**的调整（README明确说明），建议在答辩时主动解释。

---

## 总体差距汇总

| 编号 | 问题 | 严重程度 | 修复难度 |
|------|------|----------|----------|
| **G1** | **未使用VeighNa平台** | 🔴 严重 | 高（需要重构） |
| **G2** | **可解释性模块未集成到主流程** | 🟡 中 | 低（约30行代码）|
| **G3** | **通达信K线频率Bug（5min vs 30min）** | 🟡 中 | 低（改一行）|
| G4 | 佣金/滑点参数与原始需求不符 | 🟢 低 | 需要解释说明 |
| G5 | 无自动化可视化报告（已通过report_generator.py解决） | 🟢 低 | 已修复 |

---

### G1 详解：VeighNa平台缺失

**原始需求**：
> "使用VeighNa开源平台对中芯国际（688981）进行建模"

**当前状态**：项目是完全自研的Python框架，没有引入VeighNa。

**影响**：
- VeighNa 提供了标准化的行情/交易接口、回测引擎、风控框架
- 当前自研框架在功能上覆盖了需求的所有点，但不符合"使用VeighNa"的字面要求

**迁移难度**：
- VeighNa 的核心是 `Strategy` 基类，回测通过 `BacktestingEngine` 驱动
- 当前的 `EnsemblePredictor.predict()` 可以包装成 VeighNa 的 `on_bar()` 回调
- 大概需要 3-5天工作量进行迁移，不影响模型本身

**短期建议**：在答辩中说明"深度理解VeighNa框架设计，参考其架构理念自研了回测引擎，实现了等效功能"，并展示自研引擎与VeighNa的设计对照。

---

## 快速修复清单

### 立即可做（< 1小时）

```bash
# 1. 修复K线频率Bug
# 在 src/tdx/connector.py 中，将：
#   category=0   →   category=2
# 并重命名 TdxBar5Min → TdxBar30Min

# 2. 在 main.py 的 mode_full() 末尾添加可解释性分析调用
# （参见④章节的代码片段）

# 3. 运行报告生成器
python report_generator.py --run_id 20260603_165651
```

### 需要规划（> 1天）

- VeighNa 迁移（如果答辩有硬性要求）
- Optuna 超参数结果导出到报告
- Walk-Forward 各窗口 equity 曲线完整拼接（当前历史窗口的equity在断点续传时被丢弃）
