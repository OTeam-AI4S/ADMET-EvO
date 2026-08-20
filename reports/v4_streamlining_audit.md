# v4 Streamlining Audit（仅 discovery/validation，未读取 test）

## 结论

v4 保留多角色规划，但不采用多套独立训练。三个逻辑角色的原始提议统一进入一个
确定性仲裁器，按科学配置指纹去重、保护单轴 matched pair，并由单一队列执行。
因此多 Agent 是候选排序机制，不是计算预算乘数。

## 22-task 第一轮规划回放

输入为 `runs/v3-agent-tdc22-v5` 已冻结的 22 个 `profile_seed_1.json`、
`hardness_seed_1.json` 和 `research_plan_seed_1.json`；未训练新模型，也未访问 test。

- 单 Agent 固定上限：22 × 4 = 88 个第一轮 action slots。
- 修复 matched-control 身份保留后，委员会同样接受 88 个唯一配置；它不降低正式
  action 数量，也不扩大 action 数量。
- 仲裁器在规划层拒绝 46 个重复或预算外提议；这些逻辑提议未进入训练队列，规划
  成本相对模型训练可忽略。
- 加入 feature/data/model matched-pair 保护后，委员会与单 Agent 在 21/22
  endpoints 的第一轮科学配置集合完全相同。
- 唯一不同的是 hERG：委员会让 Data Steward 使用低成本 LightGBM 构造
  official/leakage-safe pair；单 Agent 使用 diagnosis 排序首位的 balanced
  ExtraTrees 构造同一数据轴 pair。
- 只有 hERG 的 Router 明确选择 `data-leakage-safe`，因此只有 hERG 执行
  official/leakage-safe matched pair。
- 轻微的 incidental data-quality/leakage 信号不再自动触发数据清洗实验。
- 当前正式环境中 Uni-Mol2 representation 不可用，委员会计划中为 0 次；即使未来
  可用，也必须通过 domain-shift 严重度、置信度、历史先验和 matched-control 门槛。

这意味着当前多角色结构的主要价值是审计、故障隔离和在数据轴上的专业化，而不是
普遍提高候选分数。生产默认不应因为“多 Agent”标签而改变 21 个没有新增信息的
任务；委员会继续作为等预算 ablation，并优先在 hERG 数据动作上验证。

## Ensemble 稳定性回放

对旧 v5 的 17 个 validation ensemble finalists，使用各成员和 blend 的同一 5 个
validation seeds 重新计算证据。新门槛要求：相对均值更强的单成员，oriented mean
gain ≥ 0.005，且至少 80% seeds 的增益方向为正。

- 通过：4/17。
- 通过任务：AMES、Clearance_Microsome_AZ、PPBR_AZ、Solubility_AqSolDB。
- 拒绝：13/17；这些融合的平均效应过小或 seed 方向不稳定。
- 该门槛不会减少已经完成的基础模型 fit，但会阻止偶然 blend 成为 finalist，并且
  新运行只持久化最多 3 个有证据的 ensemble artifacts。

## Artifact 精简

- discovery 模型默认不再保存 `model.joblib`；冻结认证始终按 finalist recipe 重训。
- validation predictions、实验配置、结果和 hash-chained ledger 继续保留。
- 不同 policy arms 共用版本化、只含 SMILES 特征且原子写入的 feature cache。
- 旧 Uni-Mol2/checkpoint 文件未删除。只读保留计划见
  `reports/v4_artifact_retention_dry_run.json`。

## Clearance Hepatocyte 定向 pilot

该任务在旧 v5 的 8-action 截断回放中有 0.0686 的 validation regret，因此先进行
定向筛选。它不是多 Agent 优势证明：修复后委员会和单 Agent 的候选集合相同；收益
来自 4+4 的阶段调度更早进入 refinement，以及科学配对/集成门槛。

| 指标 | 旧 v5 | v4 定向 pilot |
|---|---:|---:|
| Primary actions | 12 | 8 |
| Total fits | 36 | 32 |
| 5-seed finalist Spearman | 0.44332 | 0.45080 |
| Finalist seed std | 0.10315 | 0.09822 |
| 记录的 fit time | 2190.7 s | 265.4 s |
| Discovery `model.joblib` | 保存 | 0 |

v4 对 multi-view ExtraTrees 与 Morgan-only ExtraTrees 的 matched feature effect 为
+0.1980，5/5 seeds 同方向；最终 rank ensemble 相对更强 XGBoost 成员平均提升
+0.00953，5/5 seeds 同方向。pilot 过程中还发现并修复了“未配对重复 baseline
覆盖 matched control 身份”的仲裁缺陷。

## 尚不能由回放证明的事项

规划回放只能证明预算、去重和门槛按设计工作，不能证明委员会分数优于单 Agent。
下一步不应直接运行完整 committee 22-task，因为 21/22 的候选集合并未改变。应先
对 hERG 的两种数据 matched-pair 路由做同预算验证，再决定委员会是否值得作为生产
默认；其他任务继续使用较简单的单 Agent 执行链。

## hERG 数据路由结论

生产 proposer 的 official/leakage-safe LightGBM matched pair 在 5 个 validation
seeds 上得到 mean oriented delta = -0.00340、std = 0.01499、正方向 3/5；证据为
inconclusive。旧 balanced ExtraTrees 配对的均值为 -0.01805，方向性明确不支持
leakage-safe。此前一个误把 `class_weight` 写死的定向实验已隔离，不用于生产决策。
因此 hERG 继续使用 TDC official discovery data；这也进一步说明委员会当前适合作为
数据治理审计器，而不是 22 个任务的默认执行器。

## 四任务冻结协议 pilot：已确认的两类过度操作

本节记录 `runs/v4-agent-tdc22-v1` 的阶段性 validation-only 证据。该 run 绑定 v4
protocol/source hash，且未加载 test。

### 1. 同质 Optuna 候选挤占正交特征探索

Bioavailability 的 primary actions 从旧 v5 的 12 降为 8，总 fit time 从约
300.7 s 降至 110.9 s；但是 5-seed 最佳单模型均值从 0.71813 降至 0.70013。
缺失的赢家是 `morgan_count + atom_pair + maccs + rdkit2d`。v4 第二轮的 4 个名额
被 3 个相同模型/相同表示的 Optuna 局部调参和 1 个特征候选占用，三个 Optuna
候选的 seed-1 分数均未超过未调优锚点。

因此不能把“减少 action 数”本身当作优化目标。下一版的默认第二轮应采用轴内上限：

- 最多 1 个局部超参数 refinement；
- 至少 2 个彼此不同的 feature families；
- 剩余名额用于 Router 选中轴的 treatment/control 或跨模型验证；
- 候选排序同时报告验证效应、seed 稳定性、信息新颖度和 fit cost，但效果门槛优先。

### 2. 非路由高成本模型进入基线

CYP2C9 的 Router 选择 `feature-domain-shift`，但旧规则仅因 imbalance severity
达到阈值，仍把 `balanced_extra_trees` 加入基线。seed-1 的阶段性结果为：

| 候选 | AUPRC | fit time |
|---|---:|---:|
| balanced ExtraTrees | 0.79286 | 501.5 s |
| default LightGBM | 0.79689 | 10.3 s |
| best observed tuned LightGBM | 0.81002 | 14.5 s |

balanced ExtraTrees 在该 seed 上比 default LightGBM 慢约 49 倍且分数更低。最终是否
淘汰仍须等待 5-seed 稳定性结果，但候选生成逻辑已经暴露出路由不一致：当 Router
选择 `model-imbalance-aware` 时，应完整执行 balanced treatment 及其 ExtraTrees
matched control；其他路由只有在小数据、强跨任务先验或低成本 pilot 通过晋级门槛时
才保留它，不能仅凭 imbalance severity 自动支付完整多 seed 成本。这个例外很重要：
旧 v5 的 `CYP2D6_Substrate_CarbonMangels` 中 balanced ExtraTrees 确实是最佳单模型并
进入有效 ensemble，因此不能把“非选中轴”机械地全部删除。

完整 5-seed 后结论没有翻转：balanced ExtraTrees 为 0.79366 ± 0.02373、累计
2461.4 s；最终 tuned LightGBM 为 0.81794 ± 0.01834、累计 74.3 s。balanced 一个
候选占 CYP2C9 全部 3027.1 s executor fit time 的约 81%，且落后 0.02428。正式 feature
matched pair 则得到 +0.02490、5/5 seeds 同方向，属于应该保留的科学证据。新 Goal
不再为刷分扩展后续任务；该 task 只作为“保护科学配对、停止非强制昂贵候选”的成本
案例。

## 拟采用的精简原则（待四任务 pilot 完成后冻结）

1. 单 Agent 是生产默认；委员会只承担数据治理、反方审查和同预算 ablation。
2. 每轮总预算不盲目缩小，改为限制同一模型/特征/超参数邻域的槽位占比。
3. Router 选中哪个轴，就为哪个轴保留 treatment/control；其他轴只保留低成本强锚点。
4. 先跑低成本锚点和 matched pilot，再决定昂贵表示或模型是否晋级多 seed。
5. 5-seed 仍是 finalist 与科学结论的硬要求；可在晋级前使用预注册的 successive
   halving，不能根据 test，也不能用临时主观判断停止。
6. 所有删减必须同时证明：不降低跨 seed 效果、减少重复动作或计算，并保持因果
   matched pair 的可解释性。

## 两阶段昂贵候选晋级：22-task 零训练回放

使用旧 v5 已完成的 22 个任务、每个 promoted candidate 的 5-seed validation 记录，
回放如下预注册式 futility gate：先运行 seeds 1/2；一个非强制科学配对候选只有同时
满足下列条件才停止 seeds 3--5：

- 在前两个 seeds 都弱于当前 incumbent；
- 两 seed oriented mean 至少落后 0.005；
- 累计 fit time 至少是 incumbent 的 3 倍。

初步按 hypothesis 文本宽泛豁免时只停止 9 个候选、节省约 853.7 s；进一步只读审计
证明这种豁免既过宽又可被文本标签误导。改用“训练前 selected action + 结构验证只差
声明轴”的 pair identity 后，132 个 robust configs 中可停止 15 个、避免 45 个
seeds 3--5 fits、约 2063 s；另有 14 个满足 futility 条件的正式 matched-pair 成员被
保留。没有一个被停止候选成为最终 5-seed base winner，但 HIA/Pgp 各有一个距最终
best 小于 0.005，必须透明报告为“实用等效但昂贵”的删减。

这个回放支持“seed 2 barrier 后只停止高成本、双 seed 同方向受支配的非科学配对
候选”，但仍只是历史校准，不是 test 证据。所有候选必须先完成 seed 2，再一次性、
顺序无关地作决定；节省的 fits 不动态投资给新候选。下一版协议必须固定门槛、结构化
豁免、tie-break、checkpoint/resume 语义和所有 policy arms 的一致适用方式后再重跑
discovery；不能在单个任务运行中临时改阈值。

审计还发现当前 robust promotion 可能用容量截断只保留 matched pair 的一侧。v5 必须
让 selected-action pair 原子占两个 promotion slots，并在 `robust_top_k < 2` 时预先
拒绝运行；否则所谓自主科研 action 不能形成可归因证据。
