# 688981 T+0 策略调优日志

## 一、环境问题（已解决）

| 问题 | 原因 | 解决方案 |
|---|---|---|
| CPU训练极慢，半小时卡死 | `num_workers=6` 在Windows下多进程递归崩溃 | 改为 `num_workers=0` |
| RTX 5060 GPU不可用 | PyTorch 2.5.1不支持Blackwell架构(sm_120) | 换装nightly版 `cu128` |
| `torch.compile` 崩溃(Triton缺失) | Windows不支持Triton | 加平台判断，Windows自动跳过 |
| AMP API Warning | `torch.cuda.amp` 旧API被deprecated | 改为 `torch.amp.GradScaler("cuda",...)` |

---

## 二、数据问题（已解决）

| 问题 | 原因 | 解决方案 |
|---|---|---|
| 每次重跑都重新拉数据（3~5分钟） | parquet保存时带时区，读回后时区漂移，日期比较永远失败 | 保存时 `tz_localize(None)`，读回时还原时区 |
| 特征工程每次重算（10分钟） | 无缓存机制 | 加参数化缓存，文件名含ic/horizon/threshold，参数变了自动失效 |
| 2019年无数据 | Baostock对688981最早只到2020年7月 | start改为 `2020-07-16` |

**最终数据量：** 52023行（2020-07 ~ 2024-12），训练集截至2023-12，OOS为2024全年

---

## 三、模型/策略核心问题（进行中）

### 问题演进过程

**第1版（原始）**
- 佣金万10 → 改为**万2.5**
- 结果：交易6次，夏普1.36，但几乎不交易，无统计意义

**第2版：降低入场门槛**
- `buy_threshold` 0.60→0.53
- 结果：交易9次，夏普-0.16，胜率44%，交易次数还是太少

**第3版：改标签（单向→双向）**
- 标签从"收盘涨幅>阈值=1"改为"涨超阈值=1，跌超阈值=0，中间丢弃"
- horizon从3改为1（预测下一根5分钟K线）
- 结果：样本从28863降到11133（pos_rate=47%），AUC跌到0.50，几乎随机

**第4版：调低threshold救样本量**
- `cost_threshold` 0.0015→0.0008
- 结果：样本17088，AUC回到0.51~0.54，但胜率28%，亏损31%，94%靠时间止损

**第5版：改预测框架（最大涨幅标签）**
- 标签改为"未来H根K线窗口内最高价涨幅>阈值"（future_max_high）
- horizon从1改为6（预测未来30分钟窗口）
- 用numpy向量化替换逐行循环，IC计算也向量化
- 结果：样本37116，pos_rate=63.8%（太高），胜率34%，亏损30%

**第6版：扩充数据 + 提高门槛**
- 数据从2021延伸到2020-07，总行数52023
- `cost_threshold` 0.002→0.004，pos_rate从63%→55%（目标45%，还偏高）
- 加ATR+VWAP趋势过滤：ATR低于近期中位数80%或价格低于VWAP时不开仓
- 结果：交易489次，亏损15.9%，胜率34%，夏普-3.64

**第7版（最新）**
- 集成权重temperature从1→3，让最优模型权重更突出
- max_epochs 150→200，patience 30→40
- 结果：交易523次，亏损13.3%，夏普-2.29，盈亏比1.154→**1.367**（在改善）
- 胜率仍卡在34%

---

## 四、根本问题诊断

### AUC天花板

所有模型AUC稳定在 **0.54~0.58**，这是5分钟K线短期方向预测的现实上限。
四个模型AUC差距极小（约0.01），导致集成权重几乎均等，集成效果退化为简单平均。

### 止损逻辑不匹配

- 往返成本≈0.20%，止损1.2%，时间止损6根K线
- 胜率34%，盈亏比需要 **>1.94** 才能回本，实际只有1.37
- 79%的出场靠时间止损，说明持仓期间价格基本原地震荡或反向

### Kelly公式验证
```
期望值 = 34% × avg_win - 66% × avg_loss
       = 34% × 1.37x - 66% × 1x
       = 0.466 - 0.660 = -0.194  （负期望，必亏）
```

---

## 五、下一步：Optuna超参搜索

### 为什么用Optuna

手动调参已经接近极限。Optuna会系统性搜索：
- 学习率、dropout、模型层数、LSTM隐层维度
- 训练轮数、batch size
- 通过TPE采样+MedianPruner剪枝，30次trial找到最优组合

### 执行命令

```bash
python main.py --mode walk_forward --optuna
```

预计耗时：**约1小时**（RTX 5060，30 trials）

### 预期目标

| 指标 | 当前 | 目标 |
|---|---|---|
| 最优模型AUC | 0.5643 | 0.58+ |
| 胜率 | 34% | 42%+ |
| 盈亏比 | 1.37 | 1.80+ |
| 夏普比率 | -2.29 | >0 |

### Optuna之后的计划

1. 用最优超参重新训练四个模型（`--mode full`）
2. 若AUC能到0.58+，考虑把`buy_threshold`从0.53升回0.55，过滤低质信号
3. 若胜率到42%+，盈亏比1.8+，期望值转正，策略有实用价值
4. 最后接通达信模拟盘跑一周信号，验证实盘可行性

---

## 六、代码变更清单

| 文件 | 主要改动 |
|---|---|
| `config.yaml` | 佣金万10→万2.5，各项参数多次迭代 |
| `src/utils.py` | 新增（原项目缺失），路径/配置/日志 |
| `src/data/fetcher.py` | 时区修复，缓存命中逻辑修复 |
| `src/features/dataset.py` | 标签三次重设计，特征缓存，向量化 |
| `src/features/technical.py` | 新增ADX14/KAMA/DI差/枢轴点4个特征 |
| `src/models/transformer_lstm.py` | d_model/nhead扩容，Pre-LN，LayerScale |
| `src/models/baselines.py` | VanillaLSTM加注意力，CNN加SE模块，MLP加残差 |
| `src/models/ensemble.py` | Softmax权重，temperature scaling |
| `src/training/trainer.py` | AMP混合精度，pin_memory，新API |
| `src/training/optuna_search.py` | 新增（原项目缺失） |
| `src/backtest/engine.py` | 时间止损，追踪止损，ATR/VWAP趋势过滤 |
| `src/backtest/costs.py` | 万2.5佣金，cost_summary |
| `src/backtest/metrics.py` | 新增Omega/信息比率/盈亏比/连续亏损 |
| `src/tdx/connector.py` | 新增，通达信行情接入 |
| `src/tdx/live_trader.py` | 新增，实时信号循环 |
| `main.py` | GPU诊断，三模式入口，结果展示 |
| `check_env.py` | 新增，一键环境检测（含Blackwell检测） |
