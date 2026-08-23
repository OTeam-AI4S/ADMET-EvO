# v5 Effect-first Streamlined Policy Spec

状态：已退出生产默认候选；保留为 8-slot 对照/消融协议。新的默认候选为
`reports/streamlined_category_size_policy_v1.md` 中的 category × data-size 三阶段 shadow
policy。仅使用 discovery/validation 证据，test 未加载。

## 目标

在固定 TDC ADMET 22 endpoints 上，同时降低验证集过度搜索、同质候选和无效多 seed
计算；任何精简都不能以降低 5-seed 预测效果或破坏 matched-pair 科学证据为代价。

## 执行拓扑

- 生产默认：单 Agent Router + 单一实验队列。
- 条件委员会：仅在外部数据、单位/物种/assay 对齐、标签冲突或高风险数据清洗时启用
  Data Steward 与 Skeptic；委员会只排序和否决，不复制训练。
- 同预算 committee 继续作为 ablation，不能用多套训练预算换分数。
- 全部发现过程只读取 train/validation；22-task 全局冻结后才允许一次性 test 认证。

## Primary search：8 个 action slots 的分层分配

### Round 1：诊断动作与强锚点

1. Router 选中轴的 treatment/control 占 2 个 slots；两者只允许一个科学变量不同。
2. 两个低成本、互补的强模型锚点占 2 个 slots，优先 LightGBM/XGBoost/ExtraTrees
   中的历史 Pareto 候选。
3. 非选中轴不得只凭 hardness severity 自动占用昂贵 slot；小数据或强跨任务先验可获
   一个 pilot 例外。

### Round 2：限制同质搜索

- 相同 model + representation family 的局部超参数候选最多 2 个；默认 1 个。
- 至少 2 个 slots 来自与当前 winner 正交的信息轴，例如 count/topological、path/torsion、
  radius 或路由动作的第二种实现。
- 每个候选记录 novelty signature：model family、feature family、data policy、action axis、
  hyperparameter neighborhood。重复或近重复候选不能占满一轮。
- 候选排序以预期效应为第一目标，信息新颖度和成本只用于同等效应候选的排序。

Bioavailability pilot 已证明固定 `3 Optuna + 1 feature` 会漏掉有效的 count/topological
候选，因此 v5 禁止一个局部调参邻域占用 3/4 slots。

## Robust racing：效果优先的预注册早停

所有 promoted candidates 先完成 seeds 1/2。一个非强制科学配对候选仅在同时满足以下
条件时停止 seeds 3--5：

1. 两个 seeds 均弱于当前 incumbent；
2. 两 seed oriented mean 至少落后 0.005；
3. 累计 fit time 至少为 incumbent 的 3 倍。

Router 选中动作的正式 treatment/control、最终 finalist 候选和需要出具科学结论的
matched pair 必须完成 5 seeds。停止记录进入 hash-chained ledger，包含比较对象、逐
seed 差值、成本比、阈值和 `test_loaded=false`。

旧 v5 的 22-task 结构化回放中，132 个 robust configs 有 15 个会停止，可避免 45
个 fits、约 2063 s；14 个满足门槛的正式 selected-action pair 成员被豁免。该规则未
提前淘汰任何最终 5-seed base winner，但会剪掉两个与 best 差距小于 0.005 的昂贵
实用等效候选，必须在报告中披露。这只是协议校准，v5 正式 run 仍需重新验证。

决策必须发生在 seed-2 全体候选 barrier 之后，并使用稳定 config hash 做顺序无关的
incumbent tie-break。selected-action pair 必须原子占两个 promotion slots；若
`robust_top_k < 2`，搜索在训练前失败。每个 stage 原子 checkpoint，resume 校验 task、
split、seeds、protocol/source hash 和 eligible signatures，已停止候选不得复活。

## 高成本模型与 UniMol2

- balanced ExtraTrees：Router 选择 imbalance 轴时完整运行；其他路由仅在小数据、强
  先验或低成本 pilot 支持时晋级。CYP2C9 seed-1/2 已显示约 49 倍成本且分数更低。
- UniMol2：不得因“domain shift 高”直接运行。先要求 2D matched feature action 有正向
  证据，再检查已有共享 embedding cache、预计增量成本和跨任务先验。
- 首次生成的 UniMol2 embedding 只能写入版本化共享 cache；task run 只保存引用和 hash，
  不复制 checkpoint/embedding。
- 未通过增益门槛的高成本表示进入冷却期，不在相邻 endpoints 自动重试。

## Router 与 action memory 校准

- `expected_effect` 必须表示可解释的 endpoint-normalized 增益，不能把 hardness severity
  直接当成 0--1 的指标提升概率或幅度。
- 分类 AUROC/AUPRC/Spearman 使用有界尺度；MAE 类任务使用 matched control 或目标尺度
  归一化，避免 LD50 等 endpoint 的原生数值主导跨任务记忆。
- memory 保存：normalized oriented effect、seed win rate、paired uncertainty、支持概率、
  任务类别和 assay/metric 兼容性。
- prior 采用任务去重和层级收缩；一个 endpoint 不能因重复运行造成伪证据。
- Router 同时输出 effect calibration error 和 retrospective routing regret，供自主科研
  能力评分，而不是只报告“选了哪个动作”。

## 自主科研能力 v2 评分

除 feature/model/data 三轴 validation→test gain 外，固定报告：

1. 预测效果：22-task normalized oriented delta、胜率和 bootstrap CI；
2. val→test transfer：normalized optimism、方向保持率、ranking correlation；
3. 故障发现：hidden-fault precision/recall/F1、critical recall；
4. 实验设计：single-axis matched-pair 合格率、falsification 完成率；
5. 路由质量：oracle-axis regret、effect calibration/Brier 或 ECE；
6. 效率：anytime score、time-to-best、fit 数、CPU/GPU 小时和峰值存储；
7. 稳定性：归一化 seed std、winner rank stability；
8. 数据科研：provenance、license、unit/species/assay 对齐、清洗有效率和 quarantine 率；
9. 停止与自纠错：无效假设停止准确率、错误诊断修正率、负结果保留率；
10. 安全与复现：test feedback=0、协议违规=0、source/data/config hash 完整率。

最终与 strong-fixed、AutoML、rules、random、no-memory、no-hardness、no-data 进行同预算
比较；多 Agent 只作为相同 action/compute budget 的额外 ablation。

## v5 接受门槛

- 四个代表任务中，5-seed 指标不得比各自旧最佳可复现实验差超过 0.005；
- Bioavailability 必须重新覆盖 count/topological 候选；
- CYP2C9 的非强制昂贵候选必须能由预注册 racing 正确停止或证明稳定性翻盘；
- primary action、replicate fit、derived ensemble 与 feature/checkpoint storage 分开核算；
- 全测试、budget preflight、source firewall、test freeze 与 resume 测试全部通过；
- 达标后才扩展剩余 18 tasks；否则继续在 validation-only 阶段修订，不触碰 test。
