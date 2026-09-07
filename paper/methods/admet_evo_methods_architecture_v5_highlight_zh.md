# ADMET-EvO Methods 架构 v5：先呈现系统创新，再展开闭环实现

> 本版根据讨论重构：整体架构前置；原 v4 的状态构建、matched execution、外部数据与 freeze/certification 不再各占一个主章节，而是作为闭环内部的技术环节。详细 protocol audit、文献矩阵和 Supplement 分工继续保留在 `admet_evo_methods_architecture_v4_zh.md`。

## 1. Methods 首先应回答什么

Methods 开头不应先解释数据划分或 EvidencePacket schema，而应先回答：

> **ADMET-EvO 与一个普通的多模型 ADMET pipeline 或 AutoML 系统本质上有什么不同？**

建议给出以下定义：

> **ADMET-EvO is an endpoint-conditioned autonomous research framework that diagnoses the limiting factors of a molecular-property task, selects among model-, representation-, and evidence-level interventions, tests each intervention through executable comparisons, and uses the resulting evidence—including negative evidence—to determine the next research action.**

这一定义突出三个核心创新：

1. **Endpoint-conditioned research**：系统不向所有 ADMET 端点套用一套全局 recipe，而是先理解 assay、metric、data regime 和 chemical difficulty，再决定该端点需要什么研究动作。
2. **Joint evolution of models and chemical evidence**：系统不仅搜索模型或超参数，还能改变表征、模型结构与训练证据；Data axis 中的 Evidence Packet、adaptive-data 和 best-seed 都是“化学证据演化”的不同实验实例。
3. **Evidence-driven cumulative research**：Agent 的输出不是一次性的最佳模型，而是一条 hypothesis → experiment → evidence → updated decision 的研究 lineage；失败、无效和有害干预同样改变后续选择。

这三点应该在 Methods 第一页和 Figure 1 中直接可见。协议安全、hash、freeze 和 certification 是支撑这些贡献可信的机制，不应抢占贡献本身的叙事中心。

## 2. 推荐主结构：5 个部分

### 2.1 Overall architecture of ADMET-EvO

这是第一节，也是最重要的一节，建议 500–650 词。

#### 第一段：研究目标

ADMET-EvO 接收一个分子性质端点及其可用数据，输出两类结果：

- 一个 endpoint-specific predictive recipe；
- 一条说明该 recipe 为什么被保留、哪些方向失败、证据强度如何的 research lineage。

强调系统优化的不是单一 validation score，而是“在有限预算内，下一项最值得执行的科学干预是什么”。

#### 第二段：三层系统架构

**Layer I — Scientific task layer**

把异质端点转化为 EndpointContract，固定 assay semantics、task type、metric、metric direction、split 和 external-evidence policy。该层回答“我们究竟在预测什么”。

**Layer II — Autonomous research layer**

根据 endpoint contract、数据性质和历史证据构建 ResearchState；形成可证伪 hypothesis；在 Feature、Model、Data 三类科学干预中选择动作；执行实验并读取 evaluator 返回的结果。该层回答“下一步应该研究什么”。

**Layer III — Evidence and model-evolution layer**

把执行结果转化为 EvidencePacket，更新 action memory，保留 supported/refuted/failed actions，并输出更新后的 model/evidence configuration。必要的 freeze、rollback 和 claim status 在这一层完成。该层回答“实验结果如何改变系统”。

#### 第三段：一条统领闭环

正文只保留一个闭环：

\[
\text{endpoint} \rightarrow
\text{research state} \rightarrow
\text{hypothesis/action} \rightarrow
\text{experiment} \rightarrow
\text{evidence} \rightarrow
\text{updated state or stop}.
\]

这里应明确语言模型负责 hypothesis formation 和 constrained action choice；数据处理、训练、metric computation 和 evidence verdict 由确定性组件执行。这样既能体现 Agent 的研究决策作用，也不会给人“LLM 自己出题、自己打分”的印象。

#### 本节应凸显的优势

- ADMET 被处理为一组 assay-specific research problems，而非同质 benchmark tasks；
- Agent 位于模型训练之上的 scientific decision layer；
- 系统输出可积累的 endpoint knowledge，而不仅是一个 prediction file；
- 模型、表征和外部证据在同一研究语言中被统一为可测试干预。

### 2.2 Endpoint-conditioned research-state construction

建议 450–550 词。本节回答：系统凭什么决定一个端点下一步需要什么。

先简要定义 EndpointContract，然后描述由三类信息组成的 ResearchState：

1. **Scientific context**：endpoint、assay、species、label semantics、metric 和 data policy；
2. **Empirical context**：sample size、imbalance、chemical-domain shift、local roughness/activity cliffs、data quality 和 leakage risk；
3. **Research context**：已经执行的动作、正负结果、剩余预算和可用 expert。

HardnessCard 是 ResearchState 的诊断摘要，不应被写成独立贡献或占用大量公式。正文只说明它如何把观察到的困难映射为研究假设：

- domain shift → representation or applicability hypothesis；
- local roughness → model/ensemble hypothesis；
- scarcity → external-evidence or transfer hypothesis；
- imbalance → loss/model hypothesis；
- data-quality/leakage risk → curation or abstention hypothesis。

这里最值得强调的是：**诊断不会直接决定最终 recipe，而是缩小下一步需要被实验检验的 hypothesis space。** 因此 ADMET-EvO 不是基于静态规则一次性路由，而是用诊断产生候选、用实验结果修正候选。

### 2.3 Multi-axis closed-loop model and evidence evolution

这是技术主体，建议 900–1,100 词。原 v4 的第 4、5、6、7 节全部吸收到这里，作为一个完整闭环的子环节，而不是四个并列主章节。

#### 2.3.1 Falsifiable hypothesis and action selection

每项行动包含 rationale、预期方向、唯一变化轴、control、treatment、seed、成本和判定规则。Agent 可以提出或选择行动，但只能执行注册后的 ActionSpec。

科学干预只有三轴：

- **Feature axis**：改变固定描述符、分子视图、图或三维表征；
- **Model axis**：改变 learner、训练策略、ensemble 或 endpoint expert；
- **Data axis**：改变训练证据的质量、权重、数量或来源。

Replicate 和 Stop 是闭环控制动作，而不是第四、第五个科学轴。

#### 2.3.2 Executable intervention and evidence feedback

control 与 treatment 尽量共享 endpoint、split、seed 和所有非干预设置。执行结果统一转为方向一致的 effect、uncertainty、cost、status 和 provenance。

本段不必展开 atomic reservation、hash schema 和所有 gate。只需说明三个关键原则：

1. incomplete or unmatched runs do not support an intervention；
2. negative and failed outcomes remain in memory；
3. later actions are conditioned on measured evidence rather than the original textual hypothesis。

随后说明 bounded correction 与 stopping：证据不支持时，Agent 可以提出有限次数、带明确 deficiency 的 child hypothesis；预算耗尽、没有正的可辩护行动或安全条件不满足时停止/abstain。

#### 2.3.3 Data-axis evidence evolution

Data axis 在本节中使用一个较长段落或一个三级小标题即可，不再是主章节。

先说明共同逻辑：来源发现 → assay/identity curation → contamination and lineage checks → relevance/diversity selection → mass/weight control → independent evaluation。

然后说明各实验是同一 Data axis 的不同证据层：

- broad/adaptive screen 用于发现可能受益的端点和数据 regime；
- Evidence Packet v5 检查小型、来源可追溯证据能否跨 seed 确认；
- Evidence Packet v5.4 增加 development–confirmation–certification 三阶段 gate；
- endpoint best-seed study 描述该 Data action 在特定配置下的 best-observed potential。

重点不是枚举阈值，而是突出：**ADMET-EvO evolves the empirical evidence available to each endpoint while preserving assay compatibility and the option to reject or roll back harmful evidence.**

#### 2.3.4 Research memory and endpoint-specific output

闭环输出不是全局 winner，而是 endpoint-specific FrozenRecipe、相应 evidence state，以及支持/反驳该选择的 lineage。Freeze/certification 在这里用一段交代即可：最终配置在读取独立结果前冻结；失败则回滚或保留 baseline。无需单列章节。

#### 本节应凸显的优势

- Feature、Model、Data 被统一成可证伪的 scientific interventions，而不是三个彼此孤立的工程模块；
- Agent 决定“在哪个轴上继续获取证据”，这是与固定搜索空间 AutoML 的核心差别；
- Data axis 使系统能够共同演化模型与经验数据，而不是默认训练集永远正确且固定；
- negative evidence 与失败会影响下一轮研究，体现 cumulative research 而非 winner-only optimization；
- endpoint-specific output 允许不同 assay 获得不同 recipe，不强求一个模型统治所有任务。

### 2.4 Predictive realization and implementation

建议 350–450 词。本节是实现说明，不承担核心创新叙事。

只概括系统可调用的表示与学习器类别：

- 2D fingerprints and physicochemical descriptors；
- graph-based and pretrained 3D representations；
- tree ensembles、boosting、kernel models 和 ensembles；
- classification/regression-specific preprocessing and metrics。

说明 expert 只有在依赖可用、数据规模与 endpoint diagnosis 满足条件时才被激活。完整模型表、fingerprint 维度、超参数和软件版本放 Supplement。

本节还应增加一张简短的 Operational configurations 表，说明不同结果实验究竟使用了哪个 planner/controller、seed、预算与 evidence gate。其作用是防止 protocol 版本混写，不是扩展系统架构。

### 2.5 Evaluation design and statistical analysis

建议 600–750 词。

评价围绕三项核心主张组织，而不是围绕代码模块组织：

1. **Does endpoint-conditioned evolution improve predictive performance?**  
   使用 TDC-22 原生 metric、跨端点 rank、multi-seed effects 和独立评估。
2. **Does the research loop choose useful interventions efficiently?**  
   与 score-only、matched-budget random、AutoML 和固定规则比较；报告 performance–cost trade-off 和 negative decisions。
3. **Does evidence governance prevent unsafe or non-transferable promotion?**  
   报告外部来源拒绝、confirmation/certification rollback、leakage fixtures 和 claim states。

统计上以 endpoint 为 panel-level unit，seed 为 paired repeats。不同 raw metrics 不直接平均，而使用方向统一的 effect 或 within-task rank。所有失败、timeout 和未过 gate 的结果必须保留。

Freeze、held-out readout、hash 和 artifact reconstruction 在本节最后作为 reproducibility safeguards 简述即可。它们重要，但属于可信度支撑，不需要成为 Methods 的视觉中心。

## 3. Figure 1 应怎样表现“出彩”

Figure 1 不应画成十几个软件模块。建议使用一个从左到右、带反馈环的三层架构：

```text
Endpoint + available evidence
            │
            ▼
┌─────────────────────────────────────────────┐
│ Scientific task layer                       │
│ Endpoint contract + assay semantics         │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ Autonomous research layer                   │
│ Diagnose → hypothesize → choose axis        │
│              ┌────────┼────────┐             │
│           Feature    Model    Data           │
│              └────────┼────────┘             │
│                    Execute                   │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ Evidence and model-evolution layer          │
│ Effect + uncertainty + cost + provenance    │
│ Update memory → revise / retain / stop       │
└──────────────────────┬──────────────────────┘
                       │ feedback
                       └───────────────► Research state

Output: endpoint-specific recipe + evidence lineage
```

图上使用一句醒目的中心标注：

> **The agent chooses what evidence to acquire next; the evaluator determines what the evidence supports.**

这句话同时呈现自主性和可信度，比突出 freeze hash 或工具数量更容易让审稿人记住。

## 4. 推荐的 Methods 标题列表

1. **Overall architecture of ADMET-EvO**
2. **Endpoint-conditioned research-state construction**
3. **Multi-axis closed-loop model and evidence evolution**
   - Falsifiable hypothesis and action selection
   - Executable interventions and evidence feedback
   - Governed Data-axis evidence evolution
   - Research memory and endpoint-specific output
4. **Predictive realization and implementation**
5. **Evaluation design and statistical analysis**

如果目标期刊要求更短，可以把第 4 节并入第 3 节末尾，形成四个主标题。不要再把 HardnessCard、matched execution、external-data、ledger、freeze 分别列成同一级标题。

## 5. 一段可放在 Methods 开头的英文定位草稿

> ADMET-EvO was designed as an endpoint-conditioned research framework rather than a single predictive architecture. Given an endpoint specification and its available molecular evidence, the system first constructs a research state that integrates assay semantics, data regime, chemical-space difficulty, prior interventions and remaining budget. It then formulates falsifiable hypotheses and selects among representation-, model- and data-level actions. Each action is converted into an executable experiment, evaluated under a fixed endpoint contract and returned as structured evidence containing its effect, uncertainty, cost and provenance. This evidence—not the textual hypothesis itself—determines whether the action is retained, revised or rejected and conditions the next research decision. The resulting output is therefore both an endpoint-specific predictive recipe and an auditable lineage of the evidence that supports or refutes it. In contrast to fixed-dataset AutoML, ADMET-EvO can decide whether further progress requires a different representation, a different learner or different empirical evidence, allowing predictive models and their supporting chemical evidence to evolve within the same closed research loop.

这段应当成为后续 Methods 全文的“总纲”。

