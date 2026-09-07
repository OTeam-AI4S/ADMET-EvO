# ADMET-EvO Methods 双语写作骨架：参考 Closed-loop Auto Research 的组织方式

> 参考对象：Ning et al., *Closed-loop Auto Research for Molecular Property Prediction: Discovering and Certifying Generalizable Improvements* (2026).  
> 参考范围：章节组织、定义顺序和实验叙事，不复用其原句，也不在 Methods 中进行优劣比较。  
> ADMET-EvO 的写作目标：把复杂系统写成一组清楚、可执行、可复现的实验规则。

## 一、参考论文的 Methods 是怎样组织的

该论文将 Methods 命名为 **Experimental design**，主体顺序为：

1. **Axis-isolated attribution and the reliability question**  
   先定义研究问题、干预轴、选择规则、validation return、held-out return 和最终报告单位。
2. **Benchmark suites, splits, and metrics**  
   再说明任务、划分、指标方向和跨任务归一化。
3. **Held-out-test protocol**  
   在介绍 Agent 细节前，先固定 freeze–retrain–score-once 的认证规则。
4. **Submitted-trial loop, specialists, and lineage**  
   然后才写 Agent 如何读取历史、提出假设、改动管线、提交实验和接收反馈。
5. **Search budgets and reporting units**  
   明确每个 axis 的 trial budget，以及哪些 maxima 允许进入主结果。
6. **Strong baseline**  
   单独解释起始模型为何足够强，避免把弱基线提升写成研究发现。
7. **Leakage-safe data augmentation**  
   最后将身份去重、same-source rejection 和近邻过滤写成不可绕过的 evaluator rule。

这个结构的核心写作技巧是：**先定义要测量的科学对象，再说明系统如何产生它。** Methods 的主角不是 Agent 的“智能”，而是一个被清楚定义的实验设计。

## 二、我们可以借鉴的七个写作原则

### 1. 先提出可靠性问题，再介绍架构

不要以 “Our system contains five modules” 开头。先说本研究究竟要区分什么，例如：开发阶段看似有用的动作，是否在独立证据上仍成立；Agent 的研究过程是否满足预设协议。

### 2. 尽早定义分析单位和报告单位

在 Methods 开头就说明：

- endpoint 是跨任务统计的主要单位；
- seed 是配对重复，而不是独立任务；
- action outcome 是科研过程评测单位；
- frozen endpoint-specific recipe 是预测认证单位；
- main-text result 使用预先定义的 finalist，而不是所有 trial 中的事后最大值。

### 3. 把 discovery 与 certification 写成两个隔离环境

这比笼统说 “we prevented leakage” 更有说服力。需要分别交代：谁能看 validation、谁能看 protected structures、谁能看 protected labels、认证结果能否回流。

### 4. 每节都使用“对象—操作—输出—约束”的句法

例如：EndpointContract 是对象；schema validation 是操作；validated contract 是输出；字段冲突则 abstain 是约束。

### 5. 将预算和停止规则视为方法，而不是实现细节

Agent 可以尝试多少动作、何时纠错、何时停止，决定了结果的统计含义，必须写在正文。

### 6. 安全机制必须由 harness/evaluator 强制执行

不要只写 Agent “was instructed not to use test labels”。应写 protected labels 在接口层不可访问，external-data filter 和 metric evaluator 位于 Agent 不可编辑的执行边界之外。

### 7. 正文只保留决定 claim 的规则

Prompt、完整 action registry、所有超参数、source dossier 和 hash manifest 放 Supplementary Methods；正文保留读者判断有效性所必需的规则。

---

# 三、建议采用的 ADMET-EvO Methods 双语结构

## 3. Experimental design / 实验设计

### 3.1 Study objective, analysis units, and claim boundary  
### 研究目标、分析单位与主张边界

**English starter**

ADMET-EvO was evaluated as a protocol-constrained research system rather than as a single universal predictor. For each endpoint, the system converted an executable endpoint specification and an admissible evidence set into a sequence of registered research actions, a frozen modelling recipe, and an auditable claim. We evaluated two distinct outcomes: predictive utility, measured from frozen molecule-level predictions, and research-process validity, measured from the Agent's diagnoses, interventions, evidence updates, stopping decisions, and protocol compliance. The endpoint was the primary unit for cross-task inference, paired seeds were robustness replicates, and the frozen endpoint-specific recipe was the unit of certification.

**中文对应**

我们将 ADMET-EvO 作为一个受协议约束的研究系统进行评估，而不是将其视为单一的通用预测器。对于每个端点，系统把可执行的端点定义和获准使用的证据集合转化为一系列已注册的研究动作、一个冻结的建模配方以及一项可审计的结论。我们分别评估两类结果：其一是由冻结的分子级预测衡量的预测效用；其二是由 Agent 的诊断、干预、证据更新、停止决策和协议合规性衡量的科研过程有效性。端点是跨任务推断的主要统计单位，配对 seed 是稳健性重复，冻结的端点特异性配方是认证单位。

**本节还需补充**

- 最终 claim tier：certified、independently confirmed、development-only、pending；
- 哪些实验属于 predictive benchmark，哪些属于 Agent/process benchmark；
- 明确不以单次最佳 validation trial 作为最终报告单位。

---

### 3.2 Endpoint suite, data partitions, and metrics  
### 端点集合、数据分区与评价指标

**English starter**

The principal retrospective benchmark comprised 22 ADMET endpoints from the Therapeutics Data Commons, including 13 classification and 9 regression tasks. Each endpoint retained its native prediction target and registered metric: AUROC or AUPRC for classification, and MAE or Spearman correlation for regression, as specified by the benchmark contract. Metric direction was normalized only when comparing intervention effects across endpoints; all endpoint-level tables retained the native metric and scale. Discovery, confirmation, and certification partitions were assigned before Agent execution, and protected labels were unavailable to the planner, data collector, candidate selector, and action memory.

**中文对应**

主要回顾性基准由 Therapeutics Data Commons 的 22 个 ADMET 端点组成，包括 13 个分类任务和 9 个回归任务。每个端点均保留其原生预测目标和已注册指标：分类任务根据 benchmark contract 使用 AUROC 或 AUPRC，回归任务使用 MAE 或 Spearman 相关系数。指标方向仅在跨端点比较干预效应时进行统一；所有端点级表格均保留原始指标及其尺度。在 Agent 执行前，我们预先分配 discovery、confirmation 和 certification 分区，protected labels 对 planner、data collector、candidate selector 和 action memory 均不可见。

**推荐紧接一个方向统一公式**

\[
\Delta_{e,s}=
\begin{cases}
m^{(T)}_{e,s}-m^{(C)}_{e,s}, & \text{higher is better},\\
m^{(C)}_{e,s}-m^{(T)}_{e,s}, & \text{lower is better}.
\end{cases}
\]

这里建议使用 **oriented paired effect**，不要照搬参考论文的 baseline-relative ratio，除非我们最终所有跨端点分析确实使用相对归一化。

---

### 3.3 Discovery–certification separation  
### 发现阶段与认证阶段的隔离

**English starter**

Model discovery and certification were executed in separate permission domains. During discovery, the Agent could access the endpoint contract, training data, permitted development feedback, prior evidence, and the remaining research budget, but could not access protected labels. After candidate selection, the complete endpoint-specific recipe—including molecular representation, preprocessing, learner, hyperparameters, seeds, dependency versions, and admitted data snapshot—was serialized and content-addressed. A separate certification executor reconstructed the model exclusively from this frozen recipe. Certification outcomes were appended to the evidence lineage but could not modify the finalist within the same research episode.

**中文对应**

模型发现与最终认证在相互隔离的权限域中执行。在发现阶段，Agent 可以读取端点契约、训练数据、允许使用的开发集反馈、既往证据以及剩余研究预算，但不能访问 protected labels。完成候选选择后，系统将完整的端点特异性配方序列化并计算内容哈希，其中包括分子表征、预处理、学习器、超参数、seed、依赖版本以及获准数据的快照。独立的 certification executor 仅根据该冻结配方重建模型。认证结果可以追加到证据 lineage，但不能在同一研究 episode 中修改 finalist。

**为什么这一节应放在 Agent 架构之前**

读者先知道最终结果如何被保护，之后再阅读 Agent 可以做哪些探索，会更容易相信整个闭环。

---

### 3.4 Protocol-safe closed-loop research process  
### 协议安全的闭环研究过程

**English starter**

An ADMET-EvO episode followed a registered state–action–evidence loop. The Agent first validated an EndpointContract that fixed the assay semantics, species, label definition, units, transformation, metric, molecular identity rule, data ownership, and external-evidence policy. It then constructed a research state from discovery-only task diagnostics, detected faults, admissible data, prior action outcomes, provenance records, and the remaining budget. At step \(t\), the constrained policy selected a registered action \(a_t\), the matched executor returned a standardized EvidencePacket \(e_t\), and an evidence-gated update produced the next state:

\[
\mathcal{S}_{t}\xrightarrow{\pi}a_t
\xrightarrow{\mathcal{E}}e_t
\xrightarrow{\mathcal{U}}\mathcal{S}_{t+1}.
\]

The loop terminated when a registered stopping condition was met, the expected value of additional evidence was insufficient, no protocol-valid action remained, or the research budget was exhausted.

**中文对应**

每个 ADMET-EvO episode 均遵循已注册的“状态—动作—证据”闭环。Agent 首先验证 EndpointContract，以固定 assay 语义、物种、标签定义、单位、变换、指标、分子身份规则、数据权限以及外部证据政策。随后，系统根据仅使用 discovery 数据的任务诊断、已检测故障、获准数据、既往动作结果、provenance 记录和剩余预算构建研究状态。在第 \(t\) 步，受约束策略选择一个已注册动作 \(a_t\)，matched executor 返回标准化的 EvidencePacket \(e_t\)，证据门控的更新函数据此生成下一状态。达到预设停止条件、额外证据的预期价值不足、不再存在符合协议的动作或研究预算耗尽时，闭环终止。

**随后用一段话写权限边界**

The language-model planner could formulate a hypothesis or choose among registered actions, but it could not create arbitrary tools, alter the endpoint contract or evaluation protocol, read sealed labels, edit the metric evaluator, or bypass a failed safety gate.

语言模型 planner 可以提出假设或从已注册动作中进行选择，但不能创建任意工具、修改端点契约或评测协议、读取密封标签、编辑 metric evaluator，或绕过未通过的安全门槛。

---

### 3.5 Discovery-only diagnosis and registered action space  
### 仅限发现数据的诊断与注册动作空间

**English starter**

Before proposing an intervention, the Agent generated a multidimensional HardnessCard using training and development data only. The card summarized sample scarcity, class imbalance where applicable, chemical-domain shift, local structure–activity roughness, data quality, and leakage risk, while retaining the underlying statistics and data-access audit. These signals were treated as structured diagnostic context rather than as a universally validated scalar router. Each proposed intervention was required to specify a causal rationale, an expected effect direction, matched control and treatment configurations, one editable action axis, a seed schedule, an estimated cost, and a decision rule.

**中文对应**

在提出干预前，Agent 仅使用训练和开发数据生成多维 HardnessCard。该卡片概括样本稀缺性、适用时的类别不平衡、化学域偏移、局部构效关系粗糙度、数据质量和泄漏风险，同时保留底层统计量及数据访问审计。我们将这些信号作为结构化诊断上下文，而不将其表述为已经得到普适验证的单一标量路由器。每项候选干预必须预先说明因果依据、预期效应方向、相互匹配的 control 与 treatment、唯一可编辑的动作轴、seed 计划、预计成本和决策规则。

**动作轴建议写法**

Actions were grouped into Feature, Model, Data, Replicate, and Stop/Abstain axes. Feature and Model actions changed registered representations or learners; Data actions changed curation, weighting, or an admissible external source; Replicate actions acquired additional paired evidence; and Stop/Abstain actions froze the current state or requested additional evidence without further model search.

动作分为 Feature、Model、Data、Replicate 和 Stop/Abstain 五个轴。Feature 与 Model 动作改变已注册的表征或学习器；Data 动作改变数据整理、权重或获准外部来源；Replicate 动作用于获得额外的配对证据；Stop/Abstain 动作则冻结当前状态，或在不继续模型搜索的情况下请求额外证据。

---

### 3.6 Matched intervention, evidence grading, and correction  
### 配对干预、证据分级与纠错

**English starter**

Every research hypothesis was tested as an axis-isolated matched intervention. Control and treatment runs shared the endpoint contract, molecular identities, data split, seed, preprocessing, and all non-intervened settings. The executor retained molecule-level predictions, native metrics, oriented paired effects, uncertainty estimates, runtime, failures, and artifact hashes in an EvidencePacket. Unless otherwise preregistered, an action was classified as supported when its mean oriented effect was at least 0.005, refuted when the effect was at most −0.005, and inconclusive otherwise or when the required evidence was incomplete. Failed fits and timeouts remained part of the lineage and could not be removed when assigning the verdict.

**中文对应**

每个研究假设均通过单轴隔离的配对干预进行检验。Control 与 treatment 共享端点契约、分子身份、数据划分、seed、预处理以及所有未被干预的设置。Executor 将分子级预测、原生指标、方向统一后的配对效应、不确定性估计、运行时间、失败记录和 artifact hash 保存到 EvidencePacket 中。除非另有预注册，当平均方向统一效应不低于 0.005 时，该动作被判为 supported；不高于 −0.005 时被判为 refuted；处于两者之间或必要证据不完整时判为 inconclusive。失败拟合和 timeout 均保留在 lineage 中，不能在判定 verdict 时被删除。

**纠错段落**

A refuted or inconclusive action could trigger at most one bounded correction unless a separate protocol had been preregistered. The correction had to name the diagnosed deficiency, reference its parent hypothesis, change an untried action axis, and complete a new matched experiment. Scheduling a correction without an executed outcome was not counted as successful self-correction.

除非另有预注册，refuted 或 inconclusive 动作最多触发一次有限纠错。纠错必须指出已诊断的具体缺陷、引用 parent hypothesis、切换到尚未尝试的动作轴，并完成新的 matched experiment。只生成纠错计划而没有实际实验结果，不计为成功的 self-correction。

---

### 3.7 Search budgets, candidate ranking, and reporting rules  
### 搜索预算、候选排序与报告规则

**English starter**

Research budgets were fixed before each evaluation phase and applied to the complete endpoint episode rather than independently to each planner role. Candidate recipes were ranked using a prespecified score-only, uncertainty-aware, or compute-aware policy. Under compute-aware selection, candidate \(i\) received

\[
Q_i=d_i s_i-\lambda_u\sigma_i-
\lambda_c\frac{\log(1+t_i)}{\max_j\log(1+t_j)},
\]

where \(s_i\) was the development score, \(d_i\) encoded metric direction, \(\sigma_i\) was between-seed uncertainty, and \(t_i\) was measured fit time. Policy parameters and tie-breaking rules were fixed before certification. Main-text results used the frozen recipe selected by the registered policy; post hoc per-endpoint maxima across all trials were treated as exploratory and did not enter certification.

**中文对应**

每个评测阶段开始前均固定研究预算，预算约束作用于完整的端点 episode，而不是分别授予每个 planner role。候选配方按照预先规定的 score-only、uncertainty-aware 或 compute-aware policy 排序。在 compute-aware selection 中，\(s_i\) 表示开发集得分，\(d_i\) 表示指标方向，\(\sigma_i\) 表示 seed 间不确定性，\(t_i\) 表示实测拟合时间。所有 policy 参数和 tie-breaking 规则均在认证前固定。主文结果使用注册策略选择并冻结的配方；跨全部 trial 事后计算的每端点最大值仅作为探索性结果，不进入认证。

**必须填入的具体量**

- `[episode wall-time limit]`
- `[maximum actions/fits]`
- `[seed allocation]`
- `[lambda_u and lambda_c]`
- `[stopping thresholds]`
- `[what cost includes: fit time, feature generation, LLM calls, data acquisition]`

---

### 3.8 Predictive portfolio and strong baseline  
### 预测模型库与强基线

**English starter**

All model-building actions operated within a registered predictive portfolio. Two-dimensional representations included Morgan, Avalon, ErG, MACCS, RDKit path, atom-pair, topological-torsion, and normalized RDKit2D features, together with a registered multiview combination. Tabular learners included ExtraTrees, Random Forest, CatBoost, XGBoost, LightGBM, and a size-capped RBF support-vector machine. Graph and three-dimensional experts were available only when their dependencies and endpoint-specific acquisition gates were satisfied. The unmodified reference pipeline used [FINAL BASELINE CONFIGURATION], providing a strong compact baseline against which each intervention was evaluated.

**中文对应**

所有建模动作均限制在已注册的预测模型库内。二维表征包括 Morgan、Avalon、ErG、MACCS、RDKit path、atom-pair、topological-torsion 和归一化 RDKit2D 特征，以及一个已注册的多视图组合。表格学习器包括 ExtraTrees、Random Forest、CatBoost、XGBoost、LightGBM，以及受样本量上限约束的 RBF 支持向量机。图模型和三维 expert 仅在依赖可用且满足端点特异性 acquisition gate 时启用。未经修改的参考管线使用 `[最终强基线配置]`，所有干预均相对于这一紧凑且有竞争力的基线进行评估。

**写作注意**

这一节只说明“Agent 能调用什么”。完整 fingerprint 参数、hyperparameter grid 和依赖版本放 Supplementary Tables。基线为何足够强可用一两项预实验或已发表结果支撑，但不要在 Methods 展开结果讨论。

---

### 3.9 Score-blind external-evidence admission  
### 与得分隔离的外部证据准入

**English starter**

External evidence was acquired through a controlled workflow comprising source discovery, immutable capture, endpoint-aware curation, score-blind admission, fold-specific contamination filtering, and independent confirmation. Raw payloads, retrieval metadata, licence information where available, parser versions, and cryptographic hashes were stored before transformation. Candidate sources were evaluated against the EndpointContract for molecular identity, assay semantics, species, units, label transformation, conflicts, source lineage, and distributional compatibility. Missing mandatory fields or a failed gate resulted in rejection, quarantine, or abstention rather than validation-guided admission.

**中文对应**

外部证据通过受控流程获得，该流程包括来源发现、不可变原始捕获、端点感知的数据整理、与得分隔离的准入、fold-specific 污染过滤和独立确认。在任何变换前，系统保存原始 payload、检索元数据、可获得的许可信息、parser 版本和加密哈希。候选来源依据 EndpointContract 对分子身份、assay 语义、物种、单位、标签变换、冲突、来源 lineage 和分布兼容性进行审核。强制字段缺失或未通过 gate 时，系统执行 reject、quarantine 或 abstain，而不是根据 validation gain 反向决定准入。

**污染防火墙段落**

The fold-specific firewall removed exact standardized identities and excluded external molecules whose Morgan fingerprint similarity to the protected fold exceeded the registered threshold. This operation could access evaluator-authorized structures but not protected labels. Every admission decision recorded the source, exclusion reason, overlap statistics, retained sample count, and resulting dataset hash.

Fold-specific firewall 首先删除标准化身份完全相同的记录，并排除 Morgan fingerprint 与 protected fold 相似度达到注册阈值的外部分子。该操作只能读取 evaluator 授权的结构信息，不能访问 protected labels。每项准入决策均记录来源、排除原因、overlap statistics、保留样本数以及最终数据集 hash。

**注意**

我们的 endpoint-aware admission、conflict/unit/assay checks、fold-specific firewall、adaptive core set 和 external-mass control 比单纯三层去污染更复杂，但正文仍应先写成一条清楚的控制链；详细阈值再放表格。

---

### 3.10 Evaluation of research behavior and safety controls  
### 科研行为与安全控制评测

**English starter**

Predictive performance alone did not define successful autonomous research. We separately evaluated contract completion, fault diagnosis, unsafe-action avoidance, matched-pair completeness, retention of negative evidence, correction completion, rollback, stopping or abstention, review routing, compute use, and ledger integrity. Comparators included conventional and expanded AutoML, matched-cost random selection, complete deterministic rules, and component ablations. Sealed fault cases and data-governance dossiers were used to verify control execution; because locally constructed fixtures may favor complete rule systems, these experiments were interpreted as protocol-conformance tests rather than as unrestricted evidence of general scientific reasoning.

**中文对应**

预测性能并不单独定义成功的自主研究。我们另外评估 contract 完整性、fault diagnosis、unsafe-action avoidance、matched-pair completeness、负证据保留、纠错完成、rollback、停止或 abstention、review routing、计算使用以及 ledger integrity。比较对象包括 conventional AutoML、expanded AutoML、matched-cost random selection、完整 deterministic rules 以及组件消融。我们使用密封的 fault cases 和 data-governance dossiers 验证控制机制的执行；由于本地构造的 fixture 可能有利于完整规则系统，这些实验被解释为协议符合性测试，而不是对开放式科学推理能力的无限外推。

---

### 3.11 Statistical analysis and reproducibility  
### 统计分析与可复现性

**English starter**

Intervention effects were first calculated within endpoint and paired seed. Panel-level confidence intervals were obtained by endpoint-level bootstrap resampling, and paired sign-flip tests were used for prespecified small paired studies with their limited power reported explicitly. Metrics on different scales were not averaged in native units; cross-task summaries used oriented effects or within-task ranks. Failed gates, timeouts, refuted hypotheses, and inconclusive outcomes remained in the denominator defined by each analysis. Reproducibility records included source snapshots, endpoint contracts, execution configurations, dependency versions, random seeds, molecule-level predictions where distributable, runtimes, failures, ledger events, and frozen SHA-256 manifests.

**中文对应**

干预效应首先在端点和配对 seed 内计算。Panel-level 置信区间通过端点级 bootstrap resampling 获得；对预先规定的小样本配对研究使用 paired sign-flip test，并明确报告其有限统计功效。不同尺度的指标不在原始单位上直接求平均；跨任务汇总使用方向统一后的效应或任务内排名。未通过的 gate、timeout、refuted hypothesis 和 inconclusive outcome 均保留在相应分析预先定义的分母中。复现记录包括 source snapshot、endpoint contract、execution configuration、依赖版本、random seed、可再分发的分子级预测、运行时间、失败日志、ledger event 以及冻结的 SHA-256 manifest。

---

# 四、推荐的主文标题列表

如果目标期刊允许较长 Methods，推荐最终使用以下标题：

1. **Study objective, analysis units, and claim boundary**  
   **研究目标、分析单位与主张边界**
2. **Endpoint suite, data partitions, and metrics**  
   **端点集合、数据分区与评价指标**
3. **Discovery–certification separation**  
   **发现与认证隔离**
4. **Protocol-safe closed-loop research process**  
   **协议安全的闭环研究过程**
5. **Discovery-only diagnosis and registered action space**  
   **仅限发现数据的诊断与注册动作空间**
6. **Matched intervention, evidence grading, and correction**  
   **配对干预、证据分级与纠错**
7. **Search budgets, candidate ranking, and reporting rules**  
   **搜索预算、候选排序与报告规则**
8. **Predictive portfolio and strong baseline**  
   **预测模型库与强基线**
9. **Score-blind external-evidence admission**  
   **与得分隔离的外部证据准入**
10. **Evaluation of research behavior and safety controls**  
    **科研行为与安全控制评测**
11. **Statistical analysis and reproducibility**  
    **统计分析与可复现性**

如果期刊 Methods 篇幅较紧，可以合并为七节：

1. Study design and evaluation units
2. Endpoints, splits, and metrics
3. Protocol-safe closed-loop research agent
4. Registered interventions and evidence-gated updates
5. External evidence and contamination control
6. Freeze-locked certification and controls
7. Statistical analysis and reproducibility

---

# 五、写作语气模板

## 推荐使用

- “was defined as”, “was fixed before”, “could access”, “could not access”
- “was classified as supported/refuted/inconclusive according to”
- “was frozen and reconstructed by a separate executor”
- “was treated as development-only/pending until”
- “was evaluated separately from predictive performance”

这些表达让 Methods 呈现为可核查的操作规则。

## 尽量避免

- “The Agent intelligently decided …”
- “The Agent deeply understood the endpoint …”
- “Our comprehensive system greatly outperformed …”
- “To ensure no leakage, we carefully …”

前两类无法操作化，后两类分别属于 Results 和缺乏机制细节的保证。应改成具体输入、动作、权限、阈值和输出。

# 六、与参考论文保持距离的方式

1. 沿用它的**问题优先顺序**，不沿用其句子。
2. 不把我们的五轴系统强行写成对方的三轴系统。
3. 不采用对方的 normalized improvement，除非它与我们的统计分析一致。
4. 不采用 per-suite single best aggregate snapshot，除非这确实是我们的 selection unit。
5. 我们自己的重点是 EndpointContract、HardnessCard、EvidencePacket、bounded correction、data stewardship、ledger 和 claim tiers；这些应成为 Methods 的辨识度来源。
6. 任何“更完整”或“更严格”的特征只通过我们自己的实验和 artifact 支撑，不在 Methods 中进行修辞性比较。

