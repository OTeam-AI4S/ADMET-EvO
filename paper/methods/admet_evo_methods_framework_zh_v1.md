# ADMET-EvO Methods 写作框架（v1）

> 用途：先固定 Methods 的逻辑骨架、信息边界和正文/补充材料分工，再逐节扩写英文正文。  
> 核心叙事：ADMET-EvO 不是单一预测模型，而是一个受协议约束、以证据门控的端点特异性研究系统。系统把 ADMET 问题形式化为可执行任务，进行仅限开发数据的诊断，提出并执行可证伪干预，记录正负结果，在冻结后进行隔离评测。

## Methods 总体逻辑

建议用以下链条贯穿整个 Methods：

\[
\text{EndpointContract}
\rightarrow \text{Discovery-only diagnosis}
\rightarrow \text{Registered action}
\rightarrow \text{Matched experiment}
\rightarrow \text{EvidencePacket}
\rightarrow \text{Promotion/Correction/Stop}
\rightarrow \text{Freeze lock}
\rightarrow \text{Independent certification}.
\]

Methods 需要分别回答四个问题：

1. **研究对象如何定义？**——endpoint、assay、label、unit、metric 和数据分区如何被固定。
2. **Agent 实际能做什么？**——它能读取什么状态、选择哪些注册动作、受到哪些硬约束。
3. **一个动作如何变成证据？**——如何做单轴配对实验、判定 supported/refuted/inconclusive，以及何时纠错或停止。
4. **结论如何被隔离验证？**——外部数据如何准入，recipe 如何冻结，预测与 Agent 行为如何分别评测。

---

## 1. Study design and scope

### 本节目标

用一段话定义研究对象和贡献边界，避免把 ADMET-EvO 写成一个新的通用分子基础模型。

### 必写内容

- 系统定位：protocol-constrained、endpoint-conditioned、evidence-gated research agent。
- 输入：endpoint specification、允许使用的数据与证据、计算预算。
- 输出：冻结的 endpoint-specific modelling recipe、预测产物和可审计 claim。
- 完整研究 episode 的阶段：contract、diagnosis、hypothesis、execution、evidence update、optional correction、freeze、certification。
- 明确区分：预测性能与 autonomous-research behavior 是两类评测对象。

### 建议主题句

> ADMET-EvO was designed as a protocol-constrained research agent that converts an ADMET endpoint specification and an admissible evidence set into a frozen, endpoint-specific modelling recipe and an auditable claim.

### 不应在本节声称

- Agent 普遍优于 expanded AutoML 或 matched-cost random search。
- 当前 router 是已经学习得到的最优策略。
- 所有外部数据扩增结果均已独立确认。

---

## 2. Endpoint contracts and data-partition ownership

### 本节目标

说明系统如何在建模前固定科学问题，并阻止 endpoint 漂移和 test leakage。

### 2.1 Executable EndpointContract

每个 contract 至少包含：

- stable endpoint identifier 与 ADMET category；
- classification/regression task type；
- assay system、species、label semantics；
- unit、threshold、target transformation；
- primary metric 及优化方向；
- molecular identity 和重复记录处理规则；
- discovery、confirmation、certification partition；
- 可接受的 external-data role；
- 缺失或冲突元数据的 abstention/repair 规则。

强调 contract 是可执行 schema，而不是自然语言描述。CYP substrate、reversible inhibition、time-dependent inhibition、induction 等不得仅因 endpoint 名称相似而合并。

### 2.2 Partition ownership and protocol guard

说明以下数据权限：

- planner、data collector、model selector 和 memory 只能访问 discovery environment；
- protected labels 不可用于诊断、动作选择、数据筛选或配方修改；
- 如污染检查必须读取 protected structures，只允许 evaluator 提供 structure-only view；
- certification 结果只能追加到 lineage，不能回写并修改同一 finalist。

### 建议图表

- 主文 Fig. 1 中展示 contract 和 protocol guard。
- Supplementary Table S1 给出全部 22 个 endpoint 的 contract 摘要。

---

## 3. Agent architecture and research-state construction

### 本节目标

精确定义“Agent”由哪些组件构成，以及生成式规划与确定性执行之间的边界。

### 3.1 System components

建议按五层描述：

1. **Task layer**：EndpointContract、source registry、split ownership；
2. **Diagnostic layer**：HardnessCard、fault detector、data audit；
3. **Decision layer**：constrained planner/action router；
4. **Execution layer**：matched executor、model/data tools、metric evaluator；
5. **Evidence layer**：EvidencePacket、ledger、memory、freeze lock、claim evaluator。

### 3.2 Research state

定义第 \(t\) 步状态：

\[
\mathcal{S}_t = \{C, H_t, F_t, D_t, M_t, B_t, P_t\},
\]

其中 \(C\) 为 EndpointContract，\(H_t\) 为 hardness profile，\(F_t\) 为已检测 fault，\(D_t\) 为获准数据，\(M_t\) 为历史 evidence/action memory，\(B_t\) 为剩余预算，\(P_t\) 为 provenance state。

闭环写为：

\[
\mathcal{S}_t \xrightarrow{\pi} a_t
\xrightarrow{\mathcal{E}} e_t
\xrightarrow{\mathcal{U}} \mathcal{S}_{t+1}.
\]

### 3.3 Generative and deterministic boundaries

必须明确：语言模型可以提出 hypothesis 或在已注册动作中选择，但不能：

- 创建任意未注册工具；
- 修改 metric、split 或 certification rule；
- 读取 sealed labels；
- 绕过 failed data/safety gate；
- 用文本判断替代实际完成的实验。

### 待补字段

- `[LLM/provider/version]`
- `[temperature/reasoning configuration]`
- `[prompt/version hash]`
- `[tool registry and schema version]`
- `[per-episode action/fit/time budget]`

---

## 4. Discovery-only task profiling and hardness diagnosis

### 本节目标

说明 Agent 在不接触测试结果的前提下，如何识别数据稀缺、类别不平衡、分布漂移、局部 SAR roughness、数据质量和泄漏风险。

### 4.1 HardnessCard

当前可使用六维定义：

\[
H=0.18H_{scarcity}+0.12H_{imbalance}+0.25H_{shift}
+0.23H_{roughness}+0.12H_{quality}+0.10H_{leakage}.
\]

逐项交代：

- 原始统计量；
- 归一化方法；
- 仅使用哪些分区；
- 分类与回归的不同计算；
- 缺失值与无效分子的处理；
- `test_loaded=false` 等审计字段。

### 4.2 Fault diagnosis

区分两类输出：

- 描述性 hardness signal：用于构造研究上下文；
- 可执行 fault：必须映射到 repair、data action、replicate 或 abstention。

不要把单一 scalar hardness 描写为已经验证的普适路由器；当前证据更适合表述为结构化诊断输入。

### 补充材料

- 每个 hardness component 的算法或伪代码；
- normalization reference；
- 全任务 HardnessCard；
- fault fixture 定义。

---

## 5. Registered hypothesis and intervention space

### 本节目标

把 Agent 的“自主性”写成受约束的实验选择问题，而不是自由生成任意建模代码。

### 5.1 Hypothesis schema

每个 proposal 必须注册：

- causal rationale；
- expected effect direction；
- control 与 treatment；
- 唯一允许变化的 action axis；
- primary endpoint/metric；
- seed schedule；
- compute estimate；
- promotion、rejection 和 stopping rule；
- parent hypothesis（若为 correction）。

### 5.2 Five action axes

- **Feature**：更换或组合分子表征；
- **Model**：更换 learner、loss、target transform 或注册 expert；
- **Data**：curation、weighting 或获准的 external evidence；
- **Replicate**：增加 paired seeds/重采样以降低决策不确定性；
- **Stop/Abstain**：冻结当前方案、停止低价值搜索或请求额外证据。

主文只给出动作类别与代表例子；完整 action registry 放 Supplementary Methods。

---

## 6. Molecular representations and predictive learners

### 本节目标

交代 Agent 实际搜索的建模原语，使读者能够复现，而不让模型清单淹没主线。

### 6.1 Representations

正文可概括：

- Morgan radius 2/3 bit fingerprints；
- count Morgan、Avalon、ErG、MACCS、RDKit path；
- atom-pair、topological-torsion；
- normalized RDKit2D descriptors；
- Morgan + Avalon + ErG + RDKit2D multiview；
- gated graph/3D experts（如 MapLight+GNN、Uni-Mol2）。

### 6.2 Learners

正文可概括：ExtraTrees、Random Forest、balanced ExtraTrees、CatBoost、XGBoost、LightGBM、RBF-SVM，以及满足依赖和 acquisition gate 时启用的图或三维 expert。

### 6.3 Reproducibility details

完整列出：

- fingerprint 参数与维度；
- descriptor cleaning/scaling；
- hyperparameter search space 或固定配置；
- SVM sample cap；
- class weighting 和 regression target transform；
- ensemble promotion rule；
- software/library versions；
- invalid molecule policy。

建议把详细参数放 Supplementary Table S2–S3，正文保留注册库及启用原则。

---

## 7. Matched execution, evidence grading and bounded correction

### 本节目标

这是 Methods 的核心：解释一次研究动作如何变成可审计、可证伪的证据。

### 7.1 Axis-isolated matched experiment

control 与 treatment 必须共享：endpoint contract、molecular identities、split、seed、预处理，以及除目标 action axis 外的全部配置。

对 higher-is-better metric：

\[
\Delta_{e,s}=m^{(T)}_{e,s}-m^{(C)}_{e,s};
\]

对 lower-is-better metric：

\[
\Delta_{e,s}=m^{(C)}_{e,s}-m^{(T)}_{e,s}.
\]

### 7.2 EvidencePacket and verdict

EvidencePacket 至少记录：paired predictions/metrics、oriented effect、seed consistency、uncertainty、runtime、failure、artifact hashes 和 protocol status。

若沿用现有默认阈值：

- mean \(\Delta\ge 0.005\)：supported；
- mean \(\Delta\le -0.005\)：refuted；
- 其余或证据不完整：inconclusive。

需在最终正文中注明是否还有 positive-seed、confidence 或 non-inferiority gate；不要只报告均值阈值。

### 7.3 Bounded correction and stopping

- correction 必须指出上一动作失败的具体原因并链接 parent hypothesis；
- scheduled correction 不算完成，必须实际产生 linked outcome；
- 默认最多一次 correction，除非另有预注册；
- 无合法动作、信息价值过低、预算不足或安全冲突时停止/abstain；
- failed、timed-out、refuted 和 inconclusive action 均保留。

---

## 8. External-data discovery, admission and adaptive expansion

### 本节目标

说明外部数据不是按关键词抓取后直接拼接，而是经过 endpoint compatibility、provenance、identity、contamination 和独立确认门控。

### 8.1 Source discovery and immutable acquisition

记录 query、source URL/API、retrieval time、licence、raw payload、checksum 和 parser version。先冻结原始证据，再进行变换。

### 8.2 Endpoint-aware curation

依次处理：molecular standardization、assay/target semantics、species、units、censoring、duplicate/conflict resolution 和 source lineage。

### 8.3 Score-blind admission

列出直接准入与 bridge/quarantine 的标准，包括：

- 最小有效样本数；
- invalid/conflict fraction；
- exact target overlap；
- classification disagreement 或 normalized regression difference；
- distribution shift 与 train relevance；
- provenance/licence completeness。

字段缺失或 gate 失败时应 abstain/quarantine，而不是根据 validation gain 反向放宽标准。

### 8.4 Fold-specific contamination firewall

说明 fingerprint、bit length、Tanimoto cutoff、exact identity removal，以及过滤操作只使用 structure 而不使用 protected label。

### 8.5 Core-set construction and external mass control

- 分类保持 class prevalence；
- 回归按 target quantile 分层；
- scaffold-aware rounds 限制冗余；
- external sample weight：

\[
w_{ext}=\rho\frac{N_{internal}}{N_{external}},
\qquad \rho\in\{0.5,1,2,4,8\}.
\]

- 选择规则：在最佳开发结果一个标准误以内选择最小 \(\rho\)；
- 必须用独立 seed、endpoint 或 source-held-out split 做 confirmation。

### 8.6 Claim boundary

数值提升但 source lineage 不完整的结果标记为 pending/development-only，不进入 certified claim。

---

## 9. Candidate ranking and compute-aware action selection

### 本节目标

将“模型选择”和“下一研究动作选择”分开定义，避免混用两个评分函数。

### 9.1 Predictive candidate ranking

若沿用当前公式：

\[
Q_i=d_i s_i-\lambda_u\sigma_i-
\lambda_c\frac{\log(1+t_i)}{\max_j\log(1+t_j)},
\]

其中 \(d_i\) 统一 metric 方向，\(s_i\) 为开发集估计，\(\sigma_i\) 为 seed 间不确定性，\(t_i\) 为实测训练时间。

需补：\(\lambda_u\)、\(\lambda_c\)、tie-break、候选全集，以及这些值何时冻结。

### 9.2 Action value or routing

若保留 EC–VOI 分析，应作为注册 policy variant 单独描述，并明确它只对合法动作排序，不能覆盖 protocol/safety gate。由于当前 richer router 未显示普适优势，主文避免写成已验证的最优学习策略。

---

## 10. Evidence ledger, memory, freeze lock and certification

### 本节目标

说明系统如何防止选择性遗忘、结果篡改以及 blind evaluation 后继续调参。

### 10.1 Evidence ledger and action memory

每个状态转换记录：parent hash、timestamp、endpoint、action、configuration、input/output hash、status、verdict 和 cost。对 canonical serialization 计算 SHA-256，形成可检测删除、重排和修改的 hash chain。

Memory 保留 supported、refuted、inconclusive、failed 和 stopped action；跨 endpoint 迁移必须满足 action/context 可比性，并向无信息先验收缩。

### 10.2 Freeze lock

冻结产物应完整包含：representation、learner、preprocessing、hyperparameters、ensemble weights、seeds、dependency versions、source snapshot 和 prediction schema。

### 10.3 Independent certification

certification executor 只能从 frozen recipe 重建模型；不得调用 planner、adaptive data module 或 action memory。外部结果只能改变 claim status，不能修改原 lineage finalist。

---

## 11. Evaluation design

建议把评测拆成四个互不替代的部分，不用一个 leaderboard 结果代替 Agent 行为评测。

### 11.1 TDC-22 predictive benchmark

交代：

- 22 个任务（13 classification、9 regression）；
- endpoint-native metric：AUROC、AUPRC、MAE、Spearman；
- split 和 seed protocol；
- metric direction normalization 只用于跨任务汇总；
- comparator、来源和 source-traceable auditing；
- source-reported cross-system comparison 与统一重训实验的区别；
- dated leaderboard snapshot 仅作为时间敏感的 sensitivity analysis。

### 11.2 Research-policy and ablation evaluation

比较对象可包括：

- ADMET-EvO；
- conventional AutoML；
- expanded AutoML（同 feature/model/data primitives）；
- matched-cost random policy；
- complete deterministic rules；
- no-hardness、no-memory、no-data 等 ablation。

过程指标至少覆盖：contract validity、fault diagnosis、unsafe action、matched-pair completeness、negative-result retention、correction completion、rollback、stop/abstention、cost 和 ledger integrity。

### 11.3 Failure and data-governance fixtures

分别描述 sealed fault cases 和 data dossiers 的构造、封存、评分单位及 comparator。定位为 control-execution verification；若 fixture 为本地构造且 complete rules 表现相当，不扩展为自然环境下的普适推理优势。

### 11.4 Prospective multi-endpoint CYP study

单列：

- 四个 reversible-inhibition regression endpoint；
- 两个 TDI classification endpoint；
- molecule canonicalization 与 chemistry grouping；
- outer folds、seed 和 sealed certification subset；
- regression/classification metric；
- task-type-specific external evidence；
- multi-objective selection 与 macro/worst-endpoint threshold；
- blind prediction schema、hash 和 freeze time；
- external score 的 pending/completed 状态。

---

## 12. Statistical analysis

### 必写内容

- endpoint 是 panel-level inference 的主要统计单位，seed 是配对 robustness replicate；
- effect 在 endpoint × seed 内先配对，再跨 endpoint 汇总；
- endpoint-level bootstrap confidence interval；
- 小样本 paired sign-flip test 及其 power 限制；
- metric 不直接跨尺度求平均，使用 oriented difference 或 within-task rank；
- practical-effect threshold、non-inferiority margin 和 multiple-comparison strategy；
- failure/timeout 如何进入估计；
- calibration：Brier score、adaptive ECE；
- applicability/uncertainty：nearest-neighbour similarity、ensemble dispersion、similarity-stratified error、高误差检测 AUROC；
- prospective regression interval 的 coverage target（若最终保留 conformal analysis）。

### 待确认字段

- `[bootstrap repetitions and CI type]`
- `[exact sign-flip procedure]`
- `[multiple-testing policy]`
- `[all preregistered thresholds and dates]`

---

## 13. Reproducibility, software and artifact retention

### 必写内容

- code commit、Python 和核心依赖版本；
- CPU/GPU/hardware route；
- random seeds 和 determinism setting；
- runtime/fit accounting boundary；
- source snapshot、raw/processed data hash；
- experiment config、prediction array、failure log；
- frozen manifest 和 reproduction command；
- data/code availability 与无法再分发数据的限制。

建议正文给出最小复现说明，完整 artifact map、目录结构和命令放 Supplementary Information 或 repository documentation。

---

## 正文与补充材料分工

| 内容 | 主文 Methods | Supplementary Methods/Tables |
|---|---|---|
| 系统目标与闭环 | 完整 | — |
| EndpointContract 与 partition firewall | 核心规则 | 22-task contract 表、schema |
| Agent 状态与权限边界 | 完整 | prompt、tool schema、配置 hash |
| HardnessCard | 定义与六维公式 | 各分量算法、全任务数值 |
| 动作空间与 matched experiment | 完整 | 全 action registry、伪代码 |
| 表征与 learner | 类别和启用原则 | 全参数与版本 |
| 外部数据准入 | 核心 gates 和 firewall | 每来源 dossier、排除清单 |
| 路由与候选评分 | 公式与决策规则 | 参数敏感性和 policy variants |
| Ledger 与 freeze | 完整 | schema、hash manifest 示例 |
| TDC/CYP/Agent 评测 | 设计与主指标 | endpoint-level protocol 与附加分析 |
| 统计分析 | 主要单位、检验和 CI | 详细实现与 robustness checks |

---

## 建议的 Methods 主标题顺序（英文稿）

1. **Study design and system objective**
2. **Endpoint contracts and partition ownership**
3. **Agent architecture and research-state construction**
4. **Discovery-only task profiling and hardness diagnosis**
5. **Registered hypotheses and intervention space**
6. **Molecular representations and predictive learners**
7. **Matched execution, evidence grading and bounded correction**
8. **External-data discovery, admission and adaptive expansion**
9. **Candidate ranking and compute-aware action selection**
10. **Evidence ledger, freeze lock and independent certification**
11. **Predictive and autonomous-research evaluation**
12. **Statistical analysis**
13. **Reproducibility and artifact retention**

这个顺序先定义对象和权限，再解释 Agent 的决策与实验，最后才进入评测。它可以直接支撑 Fig. 1 的系统闭环、Fig. 2 的多轴干预—证据反馈，以及 Results 中的 TDC-22、policy controls、external-data 和 prospective CYP 四条证据线。

## 扩写前必须锁定的关键信息

1. 目标期刊及 Methods 字数/格式限制。
2. 最终论文主张是“autonomous research architecture”还是较窄的“cost-aware evidence acquisition”。
3. 最终保留哪一版 Agent policy、预算和 LLM 配置。
4. 哪些 external-data 结果已达到 independent confirmation，哪些仍为 development-only。
5. OpenADMET CYP blind score 是否已返回；若未返回，保持 pending。
6. 所有阈值哪些是预注册，哪些是后验工程选择。
7. TDC 跨系统结果哪些为统一重训，哪些是 source-reported transcription。
8. statistics table 中 bootstrap、sign-flip、non-inferiority 和 multiple-testing 的最终实现。

