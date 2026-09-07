# ADMET-EvO Methods 架构 v4：从“模块清单”改写为“可审计研究协议”

> 状态：Methods 正文写作前的结构稿，不是可直接投稿的最终正文。  
> 依据：截至 2026-09-02 的 `/home/wyilin/admet` 论文与代码，以及 `/data/projects/admet_evo` 的协议、运行记录和认证产物。  
> 目标：用约 3,000–3,500 个英文词讲清系统如何提出、执行、否证和认证一次 ADMET 研究决策；模型枚举、全部阈值、prompt 和 schema 放入 Supplementary Methods。

## 1. 结论先行：Methods 应讲什么故事

Methods 不应写成“我们有 EndpointContract、HardnessCard、router、memory、committee、external-data module……”的功能清单。主文只讲一条可连续追踪的科学因果链：

\[
\text{EndpointContract}
\rightarrow \text{ResearchState}
\rightarrow \text{ActionSpec}
\rightarrow \text{matched execution}
\rightarrow \text{EvidencePacket}
\rightarrow \text{state update or stop}
\rightarrow \text{FrozenRecipe}
\rightarrow \text{independent certification}.
\]

建议使用一句统领公式，其他形式化细节按需放入各节或补充材料：

\[
S_t \xrightarrow{\pi} a_t
\xrightarrow{\mathcal{E}} e_t
\xrightarrow{\mathcal{U}} S_{t+1}.
\]

其中，策略 \(\pi\) 只能选择注册动作；执行器 \(\mathcal{E}\) 以相同 split、seed 和非干预设置运行 control–treatment pair；更新器 \(\mathcal{U}\) 只接受 evaluator 生成的完整 EvidencePacket。语言模型的文字判断、未完成运行和缺少配对的结果都不能改变已认证状态。

最合适的中心定位是：

> **ADMET-EvO is an evidence-gated research system in which autonomy is expressed as constrained experimental choice, while scientific validity is protected by evaluator-owned state transitions, fail-closed evidence admission, and planner-independent certification.**

这比“一个会调用很多模型和数据库的 Agent”更有区分度，也更符合当前实验证据：本项目最强的贡献是严谨的研究编排和证据治理，而不是已经证明某个学习型 router 普遍优于 AutoML 或完整规则。

## 2. 推荐的主文结构：8 节即可

| 主文小节 | 建议篇幅 | 核心问题 | 应凸显的优势 | 审稿人为什么会关心 |
|---|---:|---|---|---|
| 1. Study design and claim boundary | 250–300 词 | 一次研究 episode 是什么；什么结果允许形成 claim | 先定义分析单位、报告单位和主张边界 | 防止把最大 validation score 当成独立证据 |
| 2. Endpoint formalization and data ownership | 300–350 词 | 异质 ADMET 端点如何被转为可执行合同；谁能访问哪一分区 | endpoint specificity 与 protocol ownership | ADMET 的 assay、单位、指标和数据来源高度异质，错误默认值会直接破坏结论 |
| 3. Closed-loop architecture and typed research objects | 350–450 词 | planner、executor、evaluator、certifier 如何交互 | 生成式规划与确定性证据控制分离 | 证明系统不是一个自由修改代码并自行打分的黑箱 |
| 4. Discovery-only state construction and registered interventions | 450–500 词 | Agent 根据什么提出下一项可证伪实验 | task-conditioned diagnosis；动作空间可审计 | 回答“为何这是 research agent，而不仅是 AutoML” |
| 5. Budgeted matched execution and evidence-gated update | 500–600 词 | 如何归因、处理失败、更新记忆并停止 | 原子配对预算、负结果保留、有限纠错 | 控制多次试验造成的选择偏差和无限重试 |
| 6. Governed external-evidence acquisition | 450–550 词 | 外部数据如何搜索、准入、去污染、加权和回滚 | assay-aware、score-blind、fail-closed，而非“数据越多越好” | 数据泄漏和 assay mismatch 是最可能被重点质疑的部分 |
| 7. Freeze lock and independent certification | 300–350 词 | 被选择的方案怎样与搜索隔离并形成最终 claim | development–confirmation–certification 分层；不可变产物 | 这是对 adaptive overfitting 最直接的防线，也是与相近工作的核心差异之一 |
| 8. Evaluation, statistics and reproducibility | 500–600 词 | 如何分别评估预测、研究过程和协议安全 | 三层评价；公平预算；claim-state reporting | 让每项能力主张都有相应 control，而不是用一个 leaderboard 分数证明全部能力 |

正文约 3,100–3,800 词。如果目标期刊要求更短，可把第 1 节压缩为第 2 节开头，把第 8 节中的实现细节移到 Supplement；不建议删除第 5–7 节，它们是方法学贡献的中心。

## 3. 每一节具体写什么、为何这样写

### 3.1 Study design and claim boundary

建议由 3 个短段落组成。

第一段定义研究目标：不是寻找全局最优模型，而是在固定 endpoint contract、预算与认证协议下，为每个端点选择一个可追溯的 FrozenRecipe，并给出其证据状态。

第二段定义分析单位和报告单位：

- 基本分析单位是一个 **endpoint research episode**；
- 干预估计单位是同 seed、同 split 的 control–treatment pair；
- panel-level 统计单位是 endpoint，而不是 seed 或单个分子；
- 报告单位是经预先规定选择规则冻结的 recipe，不是所有 trial 中的逐端点最大值。

第三段定义 claim 边界：结果分为 `certified`、`independently confirmed`、`development-only` 和 `pending`。没有通过 certification 的局部正信号可以用于生成后续假设，但不能进入最终性能主张。

这样写的优势是先固定“什么算证据”，读者之后看到 router、模型和外部数据时，不会误解为系统可以一边适应 validation 一边直接宣布成功。

### 3.2 Endpoint formalization and data ownership

先说明每个端点在任何建模前被实例化为 EndpointContract。主文列字段类别，不必逐一抄完整 JSON：

- task identity：端点 ID、ADMET 类别、分类/回归类型；
- scientific semantics：property、assay、species、label semantics、unit/transform；
- evaluation：primary metric、metric direction、split ownership；
- provenance and data policy：source family、molecular identity rule、external-evidence policy。

随后写权限边界：

| 主体 | 可读取 | 可修改 | 明确禁止 |
|---|---|---|---|
| Planner | contract、discovery data 摘要、历史 EvidencePacket | 提交注册的 ActionSpec | 读取 certification labels、修改 metric/split、直接写 evidence |
| Executor | 已批准 ActionSpec、对应训练分区 | 产生 RunArtifact | 更换未注册动作或越过预算 |
| Evaluator | control/treatment 产物与评价标签 | 生成 metric、effect、verdict | 接受纯文本“成功”声明 |
| External-data firewall | 分子结构、来源与 assay metadata；必要时可见受保护结构 | accept/quarantine/reject 记录 | 向 planner 暴露受保护结构或任何受保护标签 |
| Certifier | FrozenRecipe、冻结 manifest、certification partition | 产生一次认证读出与 ClaimRecord | 回调 planner 或重新选择 recipe |

需要使用明确否定句，例如 `The planner could not access certification labels or alter endpoint metrics after contract validation.` 这类句子比“我们避免了 leakage”更有操作性。

### 3.3 Closed-loop architecture and typed research objects

本节只介绍 6 个对象，避免把内部 Python 类全部搬进论文：

1. **EndpointContract**：固定科学任务与协议；
2. **ResearchState**：contract、HardnessCard、预算和历史证据的只读视图；
3. **ActionSpec**：hypothesis、唯一干预轴、control、treatment、seed、成本上限和判定规则；
4. **RunArtifact**：执行状态、预测、时间、配置与 hash；
5. **EvidencePacket**：方向统一的 effect、uncertainty、cost、provenance 和 verdict；
6. **FrozenRecipe / ClaimRecord**：搜索后不可修改的方案与认证状态。

每个对象统一用 `input → operation → output → invariant → failure behavior` 的句法写一段。这样既精炼，也让读者看到每一步的失败是怎样显式化的。

Figure 1 应画控制权和信息流，而不是软件包清单。必须可视化四条边界：contract 在 planning 前冻结；planner 不能写 evidence；protected labels 不进入 discovery loop；certifier 不能重新调用 planner。

### 3.4 Discovery-only state construction and registered interventions

本节先写 HardnessCard，再写动作空间和预测工具组合。

HardnessCard 使用 discovery-side 数据描述六类困难：scarcity、imbalance、chemical-domain shift、local structure–activity roughness、data quality 和 leakage risk。正文解释每一维如何影响候选动作，精确权重、采样上限和阈值放 Supplement。需要明确它是诊断摘要，不是已被证明最优的因果模型。

然后区分两类动作，避免当前草稿把它们混称为“五个优化轴”：

- **科学干预轴**：Feature、Model、Data；这些轴改变预测管线，可进入单轴归因或 factorial ablation；
- **控制动作**：Replicate、Stop；它们获取方差证据或终止 episode，不应被表述为与前三者同等的模型优化轴。

每个 ActionSpec 必须包含 causal rationale、expected sign、control、treatment、single active axis、primary metric、seed schedule、cost ceiling 和 falsification/promotion rule。自由文本 hypothesis 只有编译为合法 ActionSpec 后才可执行。

表征和 learner 只用一个短段落概括类别：fixed 2D fingerprints/descriptors、learned graph representations、pretrained 3D experts，以及 tree/boosting/kernel/ensemble learners。完整 fingerprint 维度、模型超参数和依赖版本放 Supplement。正文重点不是“模型库很大”，而是同一实验中非干预轴保持不变。

特别注意：仓库中存在 legacy effect-first、EC–VOI v1/v2、three-axis factorial 和 Evidence Packet controller 等不同策略。主文不能同时给出两个 utility 公式并暗示它们共同驱动同一次实验。应增加一张 **Operational configurations** 表，逐项标注每个实验使用的 planner、动作空间、预算、seed、promotion gate 和 certification layer。

### 3.5 Budgeted matched execution and evidence-gated update

这是 Methods 的核心节，建议按执行顺序写 5 段。

1. **Atomic reservation**：选择动作后，一次性预留完整 control–treatment pair 所需的 candidate-fit 与 fit-time 预算；预算不足以完成整对时直接停止，不能只运行 treatment。
2. **Axis isolation**：control 与 treatment 使用相同 endpoint、molecular identities、split、seed 和非干预配置。只有一个已注册科学轴可以改变。
3. **Evidence construction**：高者优指标用 \(m_T-m_C\)，低者优指标用 \(m_C-m_T\) 形成 oriented effect；同时记录不确定性、运行成本、失败、timeout 与 provenance。
4. **Evidence-gated update**：只有完整、same-seed、single-axis-verified 的 EvidencePacket 可更新状态。supported、inconclusive、refuted 和 failed 全部进入 append-only ledger；不能只保留赢家。
5. **Correction and stopping**：最多允许预注册次数的 correction；纠错必须提出 named deficiency 和 child hypothesis。无正的 robust utility、预算耗尽、无合法动作或 correction 不过 gate 时停止或 abstain。

本节可保留一个 oriented-effect 公式和一个预算约束；具体 EC–VOI 分数、所有 verdict 阈值和 edge cases 移到 Supplement。审稿人真正需要知道的是选择何时发生、哪些信息可进入后验、失败是否收费、以及何时必须停止。

EvidenceLedger 的 hash chain 用一句话说明即可：每条状态转移包含 parent hash、输入/输出 hash、状态和 verdict，修改祖先会使后续链失效。不要花大段篇幅讲序列化实现。

### 3.6 Governed external-evidence acquisition

外部数据是最值得单列的一节，因为它既是项目特色，也是最大风险源。建议写成 6 阶段：

1. endpoint-specific source discovery；
2. immutable acquisition and source lineage；
3. molecular and assay curation；
4. score-blind admission；
5. fold/test firewall, coreset and mass control；
6. independent confirmation and rollback。

这里要先固定层级：**Evidence Packet v5/v5.4、adaptive-data rerun 和单端点 best-seed 都是 Data axis 的不同实验实例，不是独立于 Data axis 的额外系统或新轴。** 建议在正文中先定义一个共享的 Data-axis workflow，再用 Operational configurations 表列出各实例的选择与证据规则：

| Data-axis 实例 | 目的 | 选择粒度 | 证据层级 | 可支持的主张 |
|---|---|---|---|---|
| broad/adaptive data screen | 发现可能有数据增益的端点、来源和质量区间 | 多来源或质量/权重网格 | development / exploratory | hypothesis generation，不形成最终因果效应 |
| Evidence Packet v5 | 检查来源可追溯的小型 packet 能否跨 seed 确认 | endpoint × packet/configuration | development + confirmation | independently confirmed 或 rollback |
| Evidence Packet v5.4 | 在 task-specific frozen recipe 下增加独立 certification gate | endpoint × frozen recipe × packet/configuration | development + confirmation + certification | strict certified data-axis effect 或 baseline fallback |
| single-endpoint best-seed | 探索特定端点在某个 seed/configuration 下的可达上限 | endpoint × configuration × development seed | exploratory held-out readout | best-observed case，不能替代多 seed data-axis certification |

因此 Methods 的抽象层只写一次 Data action：改变训练证据，同时冻结 Feature、Model、split 与 metric；各实例的差别放在 selection rule、seed schedule、promotion gate 和 claim state，而不是重复描述成多个模块。

准入不是只看结构相似度。必须同时覆盖：分子身份、endpoint/assay 定义、species、unit/transform、censoring、重复与冲突、来源独立性、exact/near-analogue contamination、chemical coverage 与 external sample mass。

将 Evidence Packet 写成最小证据单元：一个 packet 绑定一个来源、一个端点、明确 assay 语义、record-level provenance 和可靠性权重。新增、移除或替换 packet 是一次可审计数据动作；`legacy bulk pool` 与 packet-based evidence 必须在论文中分开报告。

需要明确版本化阈值。当前项目不同协议曾使用不同的近邻阈值、外部质量网格和 seed 设计。例如 Evidence Packet v5.4 使用 test-neighbour Tanimoto 0.90 的 firewall，而早期 fold-specific adaptive workflow 出现过 0.80；二者不能写成一个“系统默认阈值”。主文只解释原则，Supplement 的 operational table 给出每个实验的精确值。

这一节最能凸显的不是“Agent 自动找到更多数据”，而是：**系统能够拒绝看似有用但 assay 不匹配、来源不独立或污染风险过高的数据，并把拒绝/回滚保留为科学结果。** `/data/projects/admet_evo` 的最新结果正支持这种保守叙事：许多 Data-axis development 正信号在 confirmation/certification 消失，严格 v5.4 最终只保留极少数 data intervention；后续 best-seed 结果应作为同一 Data axis 的探索层单列，而不是覆盖 v5.4 的认证结论。

### 3.7 Freeze lock and independent certification

用 3 个阶段写清，不要只说“held-out test”：

- **Development**：允许在固定 discovery validation 上选择配置；
- **Confirmation**：使用独立 seeds、endpoint 或 cohort 检查选择稳定性；
- **Certification**：冻结 recipe、source packets、preprocessing、implementation hashes 和 selection rule，从头重训并在 planner 不可访问的分区上读取一次结果。

Certification 失败时回退到 baseline 或上一已认证 incumbent，而不是保留局部正结果。最终 ClaimRecord 同时报告选择时的开发效应、独立认证效应、状态和失败原因。

与 *Closed-loop Auto Research for Molecular Property Prediction* 的共同点是 discovery 与 held-out certification 分离；本项目应强调的差异是：认证不仅是一次 test rescoring，还包括 endpoint contract、evidence admission、分阶段 gate、失败回滚和 claim-state ledger。

### 3.8 Evaluation, statistics and reproducibility

评价必须分为三层，否则一个 leaderboard 结果无法证明 autonomy、safety 和 scientific value。

**Predictive utility**：TDC-22 的 endpoint-native metrics、oriented paired effects、within-task rank、calibration、applicability/domain shift；模型比较必须使用相同 split 和信息条件，来源转录型 leaderboard 只作为 source-traceable comparison，不等同于统一重训。

**Research-process quality**：合法 ActionSpec 比例、complete matched-pair rate、负结果保留、从 refutation 到 correction 的完成率、预算遵守、停止/abstention、单位成本收益。Router 的价值应与 matched-budget random、score-only、expanded AutoML 和 deterministic policy 对照，而不是只与单一弱 baseline 对照。

**Protocol safety**：sealed-label access、contract mutation、single-axis violation、unsafe data merge、hash/freeze integrity、certification reconstruction。人工构造 fault fixtures 只能支持“控制机制按设计执行”，不能单独证明对自然发生未知故障的广泛推理能力。

统计部分应明确：endpoint 是 panel-level inference unit；seed 是配对重复而非独立 endpoint；不同量纲的 raw metrics 不直接平均；使用 oriented effect 或 within-task rank；panel CI 用 endpoint bootstrap；小样本 sign-flip test 明确功效限制；所有 timeout、失败 gate 和 refuted hypothesis 都保留。若使用 one-standard-error rule，必须说明它发生在 development，且 selection 后的估计不能被当成无偏效应。

最后用一段 reproducibility 收口：列出环境版本、随机种子、硬件/时间核算边界、数据与代码 snapshot、prediction-level artifact、source dossier、protocol hash、frozen manifest 和公开限制。主文说“保存了什么”，完整路径与 SHA 表进入 Supplement/Repository Map。

## 4. 与重点参考论文的借鉴与区分

重点论文的 Methods 顺序是：可靠性问题 → 干预轴与报告单位 → benchmark/split/metric → held-out protocol → submitted-trial loop → budget → strong baseline → leakage-safe augmentation。这个顺序非常值得借鉴，因为它先固定评价对象，再讲系统。

但不建议直接沿用其 7 个标题。本项目真正不同的对象是 endpoint-conditioned state、typed evidence、staged gates 和 fail-closed data governance。建议形成以下对照：

| 维度 | Closed-loop Auto Research | ADMET-EvO 应如何写 |
|---|---|---|
| 基本研究单位 | suite-level submitted pipeline snapshot | endpoint research episode 与 matched action pair |
| 干预空间 | feature/model/data 三轴代码或数据编辑 | Feature/Model/Data 科学轴 + Replicate/Stop 控制动作 |
| 状态 | shared lineage 与 validation reward | EndpointContract + HardnessCard + budget + EvidencePacket history |
| 归因 | file-level ablation lock | ActionSpec single-axis invariant + matched executor + factorial controls |
| 外部数据 | contamination filter 后 merge | Evidence Packet、assay/lineage compatibility、score-blind admission、mass cap、rollback |
| 选择后验证 | frozen configuration held-out rescore | development → confirmation → certification + claim state |
| 失败处理 | lineage 保留失败 trial | failed/inconclusive/refuted evidence、bounded correction、abstention、baseline fallback |
| 最适合的贡献措辞 | auto-research gain 是否 transfer | 自主选择如何被证据门控与协议约束，从而形成可审计 claim |

这样既能吸收其清晰的实验设计写法，又不会让审稿人觉得只是增加更多模块的复刻。

## 5. 文献学习集：17 篇期刊论文 + 4 篇直接相关旗舰论文

下面不是为了在 Methods 中逐篇引用，而是提炼写作范式。最终正文只在相应技术点引用最直接来源。

### 5.1 自主研究、adaptive selection 与认证

1. [Closed-loop Auto Research for Molecular Property Prediction (2026)](https://arxiv.org/abs/2606.22731)：先定义 reporting unit 和 held-out reliability question，再写 agent loop；是本文最直接的结构参考。
2. [Autonomous chemical research with large language models, *Nature* (2023)](https://www.nature.com/articles/s41586-023-06792-0)：把工具接口、planner、实验执行和 schema-valid outputs 写成可操作系统，而非抽象“推理能力”。
3. [Large language models for scientific discovery in molecular property prediction, *Nature Machine Intelligence* (2025)](https://www.nature.com/articles/s42256-025-00994-z)：用清楚的 knowledge synthesis → feature construction → predictive evaluation 链条组织方法，并把可解释知识与预测模型分开。
4. [The reusable holdout, *Science* (2015)](https://www.science.org/doi/10.1126/science.aaa9375)：说明 adaptive reuse of validation 会损害统计有效性；支持 planner-isolated certification 的必要性。
5. [On over-fitting in model selection and subsequent selection bias in performance evaluation, *JMLR* (2010)](https://www.jmlr.org/papers/v11/cawley10a.html)：支持把 selection 与 unbiased performance estimation 分离，并先定义 reporting unit。

### 5.2 数据、split 与 benchmark 可靠性

6. [Trust, But Verify, *Journal of Chemical Information and Modeling* (2010)](https://pubs.acs.org/doi/10.1021/ci100176x)：化学结构标准化、重复与冲突处理必须成为方法正文，不是清洗脚注。
7. [Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction, *JCIM* (2013)](https://pubs.acs.org/doi/10.1021/ci400084k)：评价 split 必须匹配未来应用场景。
8. [MoleculeNet, *Chemical Science* (2018)](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a)：先定义 dataset、split 和 metric，再比较 featurizer/learner；对异质任务使用 endpoint-native metrics。
9. [Therapeutics Data Commons, NeurIPS Datasets and Benchmarks (2021)](https://arxiv.org/abs/2102.09548)：benchmark task object、数据来源、split 与 evaluator 的标准化写法。
10. [Large-scale comparison of machine learning methods for drug target prediction on ChEMBL, *Chemical Science* (2018)](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c8sc00148k)：nested cluster CV 把化学系列偏差与超参数选择偏差同时纳入设计。
11. [Exposing the Limitations of Molecular Machine Learning with Activity Cliffs, *JCIM* (2023)](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01073)：整体平均性能不能替代 activity-cliff/局部 roughness 分析，支持 HardnessCard 的局部难度维度。

### 5.3 表征、强 baseline 与 endpoint heterogeneity

12. [Analyzing Learned Molecular Representations for Property Prediction, *JCIM* (2019)](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237)：强 fingerprint/descriptor baseline、scaffold overlap 与 hyperparameter selection 都要公平控制。
13. [Pushing the Boundaries of Molecular Representation with the Graph Attention Mechanism, *Journal of Medicinal Chemistry* (2020)](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959)：模型结构、输入特征、任务与解释分析分层写，避免用模型名代替可复现细节。
14. [ADMET-AI, *Bioinformatics* (2024)](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030)：多端点 ADMET 工具需同时交代模型、单位、reference context、速度与部署边界。
15. [ADMETlab 2.0, *Nucleic Acids Research* (2021)](https://doi.org/10.1093/nar/gkab255)：大规模 ADMET 平台用数据来源、endpoint 定义、模型与应用接口四层组织方法。
16. [Unified and explainable molecular representation learning for imperfectly annotated data, *Nature Communications* (2025)](https://www.nature.com/articles/s41467-025-63730-6)：强调 imperfect annotation 与多任务异质性，支持按 endpoint/annotation regime 描述方法边界。
17. [Uni-Mol, ICLR (2023)](https://openreview.net/forum?id=6K2RM6wVqKu) 与 18. [Uni-Mol2, NeurIPS (2024)](https://arxiv.org/abs/2406.14969)：预训练表征的 pretraining、finetuning 和 downstream evaluation 必须分开；在本文只作为 gated expert，不应写成默认组件。

### 5.4 Applicability、uncertainty 与可拒绝预测

19. [Introducing Conformal Prediction in Predictive Modeling, *JCIM* (2014)](https://pubs.acs.org/doi/10.1021/ci5001168)：用预先定义的 coverage 解释预测区间，并明确 calibration/evaluation 分区。
20. [Conformal Regression for QSAR Modeling, *JCIM* (2018)](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00054)：报告 coverage 与 interval efficiency，不能只报平均精度。
21. [Molecular Similarity-Based Domain Applicability Metric, *JCIM* (2019)](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00597)：ensemble dispersion 与 chemical distance 捕捉的风险不同，支持将 applicability 和 model uncertainty 分开评价。
22. [Conformalized Graph Learning for Molecular ADMET Prediction, *JCIM* (2024)](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01139)：ADMET accuracy 与 uncertainty validity 应并列描述，且特别考察 out-of-domain compounds。

严格按出版类型计，上述集合含 17 篇期刊论文和 4 篇旗舰会议/直接相关预印本，超过“至少 15 篇”的要求。另有现成的 24 篇 Agent 顶会 Methods 复盘，可继续作为 agent interface、budget、tool execution 和 process evaluation 的补充学习集，但不应替代化学信息学与统计有效性文献。

## 6. 主文与 Supplement 的分工

| 内容 | 主文保留 | Supplement 展开 |
|---|---|---|
| EndpointContract | 字段类别、验证与失败行为 | 完整 schema、22 个实例、所有单位/transform |
| HardnessCard | 六维含义、discovery-only、如何影响候选动作 | 公式、权重、采样、阈值、敏感性 |
| Research loop | 一条状态转移公式、权限边界、停止 | 伪代码、完整 state schema、policy variants |
| Action space | 三个科学轴和两个控制动作 | registry、prompt、全部候选配置 |
| Matched evidence | 配对不变量、oriented effect、verdict 类别 | endpoint thresholds、共享 control 规则、edge cases |
| External evidence | 六阶段、score-blind、firewall、mass control、rollback | source dossiers、每版阈值、查询式、清洗与 licence 表 |
| Predictive portfolio | 表征/learner 类别与 gated expert 原则 | 所有维度、超参数、依赖、硬件 |
| Freeze/certification | 三阶段、权限隔离、claim states | manifest、hash chain、重建脚本和 artifact map |
| Evaluation/statistics | 三层问题、controls、analysis/reporting unit | 全 endpoint 表、bootstrap/sign-flip 细节、额外分析 |

## 7. 当前稿件在扩写前必须修正的 8 个问题

1. **Data axis 内部的协议版本混写**：早期 adaptive-data、Evidence Packet v5/v5.4、three-axis v2 中的 D1–D3 与 DILI best-seed 都属于 Data axis，但选择粒度、阈值、seed 和 gate 不同；必须在一个 Data-axis hierarchy 下用 Operational configurations 表分开，而不能误写成多个新轴或把结果直接合并。
2. **“五轴”表述不严谨**：Replicate 和 Stop 是控制动作，不是与 Feature/Model/Data 等价的预测干预轴。
3. **两个 routing utility 不能并列冒充同一 deployed policy**：effect-first、EC–VOI 与后续 controllers 要逐实验标注。
4. **外部数据阈值版本不一致**：0.80 与 0.90 Tanimoto firewall、不同 ratio/weight grid 需对应具体协议。
5. **同一 Data axis 的开发正信号与认证结果必须分层**：v5 retained-confirmation、v5.4 strict certification、exploratory best-seed 都是 Data-axis evidence，但不能合并成一个数；应分别报告发现、确认、认证和 best-observed envelope。
6. **DILI 2026-09-01 best-seed 是探索性选择**：其 freeze 文件明确按单一最佳 development seed 选择，且该配置全 seed 均值为负；即使 official test 为正，也不能写成预注册多 seed certification 的证据。
7. **Agent 优于 AutoML/规则的措辞需受限**：现有稿件自己记录 router 未胜 expanded AutoML/matched-cost random，complete rules 在若干 sealed fixtures 持平；Methods 应主动给出这些 comparator 的公平设置。
8. **“代码中存在”不等于“端到端被使用”**：Methods 的每个核心对象都必须指向实际 run artifact、protocol hash 或结果表；仅被单元测试覆盖的模块放在 implementation/Supplement，不作为已验证贡献。

## 8. 推荐 Figure 1 与 Algorithm 1

Figure 1 只画 3 个 panel：

- **a, Protocol ownership**：EndpointContract 和 discovery/certification partitions；标明可见性边界；
- **b, One endpoint episode**：state → registered action → matched execution → EvidencePacket → update/stop；
- **c, Certification and claim states**：FrozenRecipe → independent retraining/readout → certified/rollback/pending。

Algorithm 1 保持 15–20 行：验证 contract；构建 discovery state；在预算内选择合法 action；原子预留 pair；执行；仅在 evidence 完整时更新；若满足 promotion 则冻结；独立认证；输出 ClaimRecord。外部数据六阶段单独作为 Algorithm S1，不塞进主算法。

## 9. 下一步正文写作的推荐顺序

先写第 1、2、3、5、7 节，锁住对象、权限、证据和认证；再把第 4 节的具体 policy 与第 6 节的版本化数据协议填入；最后根据 Results 的最终数字写第 8 节实验设置。这个顺序能避免 Methods 先写死一个策略，随后又被尚在变化的实验结果推翻。
