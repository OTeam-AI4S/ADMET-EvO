# hERG Data-Route Decision（validation-only）

## 决策

当前证据不支持把 `leakage_safe` 作为 hERG 的性能增强数据策略。v4 finalist 搜索应
继续使用 official 数据；数据处理实验可以作为诊断证据保留，但不得因为检测到少量
train/validation 结构重叠就自动替换训练数据。

## 精确 matched-pair 证据

所有结果均使用相同模型、特征、超参数和 validation seed，仅改变 data policy。
正数表示 `leakage_safe` 优于 official。

| 模型配置 | Seeds | Mean Δ | Δ std | 正方向 | 判定 |
|---|---:|---:|---:|---:|---|
| balanced ExtraTrees, default | 5 | −0.01805 | 0.01427 | 0/5 | refuted |
| LightGBM, tuned | 5 | +0.00033 | — | 2/5 正、1 tie | inconclusive |
| LightGBM, default, production proposer | 5 | −0.00340 | 0.01499 | 3/5 | inconclusive |

生产 proposer 的逐 seed AUROC：

| Seed | official | leakage_safe | Δ |
|---:|---:|---:|---:|
| 1 | 0.90625 | 0.88750 | −0.01875 |
| 2 | 0.85106 | 0.85666 | +0.00560 |
| 3 | 0.93630 | 0.91304 | −0.02326 |
| 4 | 0.87995 | 0.89556 | +0.01561 |
| 5 | 0.87939 | 0.88319 | +0.00380 |

门槛为平均 oriented effect ≥ 0.005 且至少 80% seeds 同方向。三个配置均未支持
`leakage_safe`；balanced ExtraTrees 还满足反向 refutation 门槛。

## 单 Agent 与委员会的真实差异

使用 22 个已有 HardnessCard、ResearchPlan 和每轮前 4 个已完成 primary results
进行零训练回放：

- 第一轮 scientific configuration set：21/22 endpoints 完全相同，仅 hERG 不同。
- 第二轮 refinement set：22/22 完全相同。
- 两轮均相同：21/22。
- hERG 的差异只是用哪个模型承载 official/leakage-safe matched pair；两种策略都保留
  official LightGBM，而数据 intervention 本身没有得到支持。

因此完整运行第二套 22-task committee 训练不会提供与成本相称的新预测证据。委员会
保留为数据治理、仲裁审计和故障隔离组件；默认预测搜索继续采用较简单的单 Agent。

## 自我否证记录

`runs/v4-herg-data-pair-lightgbm-v1` 曾误用 `class_weight`，而生产 proposer 根据
hERG 阳性率 0.665 应使用 `imbalance=none`。该 run 已被明确判为不回答生产路由问题，
未与正式证据混合。正确结果位于：

`runs/v4-herg-data-pair-lightgbm-v2/matched_pair_summary.json`

## 安全边界

- `test_loaded: false`
- 未加载 TDC test labels。
- 未保存 discovery `model.joblib`。
- 本结论只约束 discovery 数据路由，不构成 frozen-test 性能声明。

