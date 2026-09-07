# ADMET-EvO Methods v3：一个端点究竟如何被自主研究

> 本稿用于替换上一版中过于抽象的 system overview、planning 和 evidence 部分。重点是说明系统实际计算什么、执行什么，以及一次实验怎样改变下一步。英文为主文初稿，中文用于核对逻辑。

## 1. Protocol kernel and autonomous research policy  
## 协议内核与自主研究策略

### English

ADMET-EvO comprised two coupled but separately testable subsystems. The **protocol kernel** deterministically owned endpoint semantics, molecular identity processing, split and metric definitions, experiment validation, model execution, paired-effect calculation, artifact hashing, and freeze/certification. The **research policy** observed the discovery-side state and decided which registered experiment should be performed next, whether evidence should be replicated or repaired, and when the episode should stop. The policy could not train a model or assign an evidential verdict directly; it had to emit an executable action that passed the protocol kernel.

The validated primary implementation used a deterministic state-conditioned planner. A language-model planner was defined behind the same typed interface but was not required for the principal evidence in this study. We therefore define autonomy by closed-loop experimental decision-making—state observation, action selection, physical execution, evidence-dependent update, and stopping—rather than by the use of an LLM.

### 中文

系统分成两个部分。**协议内核**控制端点、数据划分、指标、实验校验、训练执行、证据计算和冻结认证；**研究策略**决定下一项实验、是否补充证据以及何时停止。Planner 不能直接训练模型或宣布 hypothesis 成立，只能提交经过 Protocol Guard 校验的可执行动作。

当前主要证据来自确定性的 state-conditioned planner。LLM adapter 使用相同接口，但不是核心结果成立的必要条件。因此，本文的 autonomy 来自“观察—决策—执行—证据更新—停止”的闭环，而不是“调用了 LLM”。

## 2. Operational configurations  
## 实际运行配置

Methods 必须区分四种配置，不能把它们写成每个端点都会完整运行的一条大流水线。

| Configuration | Purpose | Autonomous operation |
|---|---|---|
| **Recipe-construction search** | 构建 TDC endpoint-specific recipe | 多轮候选生成、validation、robust racing、可选 ensemble 和 finalist selection |
| **Matched-action episode** | 检验 hardness-driven intervention | 生成一个单轴 hypothesis，执行 treatment/control，多 seed 分级，决定 correction/stop |
| **Sequential EC–VOI episode** | 在严格预算下决定下一实验 | 根据历史与当前配对效应逐步选择下一 pair，或在 utility 非正时停止 |
| **Lean/prospective operation** | 复用合格任务或研究新任务 | 健康任务 reuse-and-stop；新 endpoint、异常或数据缺口触发有界 research ticket |

Broad recipe search、research-policy evaluation 和 lean deployment 共享协议内核，但不是同一次运行。论文每个实验小节应明确使用了哪一种 configuration。

## 3. Episode input and initialization  
## Episode 输入与初始化

### English

An endpoint episode received a task identifier, discovery train/validation data, a registered metric and direction, ordered seeds, candidate and fit-time budgets, an optional frozen action-memory snapshot, and a source-code snapshot. The task was resolved to an EndpointContract containing the endpoint identifier, ADMET category, task type, label semantics, assay system, species where applicable, source family, metric, discovery and certification ownership, and external-data policy.

Discovery loaded data only through the train/validation split interface. The API that materialized the TDC test frame was reserved for finalization. Before planning, the Agent fixed the contract, source and protocol hashes, seed schedule, and budget. A contract mismatch, invalid seed schedule, unavailable mandatory artifact, or forbidden data path caused fail-closed termination before fitting.

For an already qualified endpoint, the operational policy first verified the FrozenRecipe and checked for a new contract fault, data drift, uncertainty, or performance-degradation signal. A valid artifact without a new trigger was reused with zero additional research fits. A new or flagged endpoint entered the active procedure below.

### 中文

一次 episode 的输入包括 endpoint、discovery train/validation、metric、ordered seeds、candidate/fit-time budget、冻结的 memory snapshot 和 source snapshot。系统首先建立 EndpointContract 并冻结协议。Discovery 只能访问 train/validation；物化 test 的接口只属于 finalization。

若一个已经合格的 endpoint 没有新的 contract fault、drift、uncertainty 或性能退化信号，Agent 直接复用 FrozenRecipe 并停止。新任务或异常任务才启动主动研究。

## 4. Discovery-side state construction  
## 构建 discovery-side state

The Agent canonicalized molecules and audited invalid structures, missing or non-finite targets, repeated identities, conflicting labels, and replicate dispersion. It computed sample size, class prevalence or target dispersion, unique Bemis–Murcko scaffolds, validation-scaffold novelty, and nearest-neighbour chemical similarity. These quantities formed TaskProfile and a six-dimensional HardnessCard. Morgan radius-2, 2,048-bit fingerprints were used for diagnostic similarity.

Let \(n_{\mathrm{scaf}}\) be the number of training scaffolds and \(p\) the positive prevalence. Scarcity and classification imbalance were

\[
h_{\mathrm{scarcity}}
=\operatorname{clip}_{[0,1]}
\left[
1-\frac{\log(1+n_{\mathrm{scaf}})}{\log(3001)}
\right],
\qquad
h_{\mathrm{imbalance}}
=\operatorname{clip}_{[0,1]}[1-2\min(p,1-p)].
\]

Imbalance was zero for regression. For every validation molecule, the nearest training molecule was identified by Tanimoto similarity. With unseen-scaffold fraction \(r_{\mathrm{unseen}}\), fraction below nearest-neighbour similarity 0.40 \(r_{\mathrm{OOD}}\), and mean nearest similarity \(\bar s_{NN}\),

\[
h_{\mathrm{shift}}
=0.45r_{\mathrm{unseen}}
+0.35r_{\mathrm{OOD}}
+0.20(1-\bar s_{NN}).
\]

Local roughness was

\[
h_{\mathrm{rough}}
=0.60r_{\mathrm{NN-disagree}}
+0.40r_{\mathrm{cliff}}.
\]

For classification, disagreement meant different labels. For regression, target difference was scaled by twice the training-target standard deviation. An activity-cliff candidate required Tanimoto similarity at least 0.75; it was counted as a cliff when labels differed or a regression target differed by at least one training-target standard deviation.

Data-quality severity combined invalid structures/targets with identity-level conflicts for classification or replicate prevalence and dispersion for regression. Leakage severity was exact canonical train–validation overlap divided by the smaller split size. The reported summary was

\[
H=
\frac{
0.18h_{\mathrm{scarcity}}
+w_Ih_{\mathrm{imbalance}}
+0.25h_{\mathrm{shift}}
+0.23h_{\mathrm{rough}}
+0.12h_{\mathrm{quality}}
+0.10h_{\mathrm{leakage}}
}{
0.18+w_I+0.25+0.23+0.12+0.10
},
\]

where \(w_I=0.12\) for classification and \(0\) for regression. The planner received the complete dimension vector, supporting counts, and confidence, not only \(H\). The full diagnostic payload was content-hashed.

### 中文

这里不是让 Agent 泛泛地“理解数据”，而是计算六组可复现信号：scaffold scarcity、class imbalance、chemical shift、local roughness/activity cliff、data quality 和 exact-overlap leakage。真正进入 planner 的是六维向量、原始统计量和置信度，而不只是一个 hardness 总分。

## 5. From diagnosis to an executable experiment  
## 从诊断到可执行实验

The planner mapped diagnostic conditions to registered action families with fixed falsification tests and treatment–control templates.

| Trigger in the implemented planner | Treatment | Matched control | Axis |
|---|---|---|---|
| exact split overlap \(>0\) | leakage-safe data | official/identity-safe data | Data |
| data-quality severity \(\geq0.03\) | quality-filtered or leakage-safe data | identity-safe data | Data |
| imbalance severity \(\geq0.25\) | class-weighted or balanced ExtraTrees | unweighted ExtraTrees | Model |
| domain-shift severity \(\geq0.25\) | Morgan + Avalon + ErG + RDKit2D | Morgan radius-2 only; same ExtraTrees | Feature |
| local-roughness severity \(\geq0.20\) | Random Forest | ExtraTrees; same rich 2D features | Model |
| scarcity severity \(\geq0.35\) | quarantined external-data ticket | native data | Data |
| every active episode | additional paired seeds | current runner-up | Replicate |
| every active episode | Stop/Abstain | — | Stop |

An external-data proposal did not authorize a merge. It created a quarantined ticket requiring endpoint, assay, species, unit, provenance and licence checks, score-blind admission, contamination filtering, and independent confirmation. Only admitted evidence could become a supervised Data treatment.

Each ResearchAction contained its identifier, axis, causal hypothesis, falsification test, rationale, expected effect, diagnostic confidence, normalized compute cost, scientific risk, executable payload, and optional parent action. It was compiled to an ActionPair with explicit treatment, control, ordered seeds, practical-effect threshold, predicted observation error, candidate charge, time ceiling, and task-context hash. The guard rejected a pair if treatment and control were identical, changed a protected field, or changed any field outside the declared axis.

### 中文

HardnessCard 的关键作用是把诊断编译成唯一变化轴的实验。例如，domain shift 不是触发“试更强模型”，而是固定 endpoint、data、ExtraTrees、hyperparameters 和 seeds，只比较 rich 2D 与 Morgan。Local roughness 触发 Random Forest 对 ExtraTrees；imbalance 触发 weighted/balanced 对 unweighted；data quality 触发过滤数据对 identity-safe data。

## 6. Representations, learners, and candidate generation  
## 表征、模型与候选生成

The registered representation library contained Morgan radius-2 and radius-3 bit fingerprints (2,048 dimensions), count Morgan fingerprints, Avalon (1,024), ErG (315), MACCS (167), RDKit path, hashed atom-pair and topological-torsion fingerprints (2,048 each), finite RDKit2D descriptors, and an optional frozen Uni-Mol2 representation. The default rich 2D view concatenated Morgan radius-2, Avalon, ErG, and RDKit2D.

The learner registry contained ExtraTrees, Random Forest, balanced ExtraTrees for classification, CatBoost, XGBoost, LightGBM, and RBF-SVM for at most 5,000 training examples. During broad recipe construction, the first round paired the rich 2D view with available learner families and prioritized balanced ExtraTrees for diagnosed imbalance. Later rounds could refine the current winner using Optuna TPE and compare three registered alternatives: radius-3 circular features, count/topological features, and path/torsion features. Cross-endpoint recipe transfer required at least two supporting tasks and a mean within-task percentile of at least 0.75.

Broad search was used to construct endpoint recipes. In matched-action and lean modes, only the selected treatment and control were fitted; the full model zoo was not reopened.

### 中文

我们实际使用的表征包括 Morgan、Avalon、ErG、MACCS、RDKit path、atom-pair、torsion、RDKit2D 和受 gate 控制的 Uni-Mol2；模型包括 ExtraTrees、Random Forest、balanced ExtraTrees、CatBoost、XGBoost、LightGBM 和小样本 RBF-SVM。需要明确：宽搜索用于 recipe construction，而研究一个具体 hypothesis 时只跑一对配置。

## 7. How the Agent selects the next experiment  
## Agent 如何选择下一项实验

### 7.1 Initial effect-first policy

The initial auditable policy ranked eligible actions using

\[
U_{\mathrm{EF}}(a)
=p\widetilde E_aC_a
-(1-p)(0.55K_a+0.45R_a),
\]

where \(p=0.80\), \(\widetilde E_a\) was memory-calibrated expected effect, \(C_a\) diagnostic confidence, \(K_a\) compute cost, and \(R_a\) scientific risk. Because the expected-effect scales were heuristic, this policy was treated as an initial implementation rather than a proven optimal router.

### 7.2 Sequential metric-valued EC–VOI policy

The sequential policy operated on resolved ActionPairs. Only complete, single-axis, discovery-side matched effects were admitted. Test- or leaderboard-derived observations failed at object construction. Historical evidence from another endpoint was weighted by context similarity.

For each action family, standardized matched effects were combined in a Gaussian posterior with prior mean \(0\), prior standard deviation \(1.5\), and between-endpoint standard deviation \(1.0\). Multiple pairs from the same endpoint were first aggregated so they did not count as independent endpoint replicates. The policy computed

\[
I_{\mathrm{robust}}(a)
=\max[0,\mu_a-z\sigma_a-\delta_a]
\]

and expected information gain

\[
\operatorname{IG}(a)
=\frac12\log
\left[
1+\left(
\frac{\sigma_a}{\sigma_{\mathrm{obs},a}}
\right)^2
\right].
\]

The robust execution utility was

\[
U_{\mathrm{ECVOI}}(a)
=I_{\mathrm{robust}}(a)
+0.04\operatorname{IG}(a)
-0.08C_a
-0.08R_a,
\]

where \(C_a\) combined the fractions of the task fit-time and candidate budgets required by the pair. The pair with the largest positive robust utility was selected; if every admissible utility was non-positive, the policy emitted Stop. A preregistered exploration variant allowed one endpoint-local experiment before local evidence existed only when its acquisition utility was positive and risk was at most 0.10.

### 中文

这一步才是“自主研究”的决策核心。Agent 对每个已编译 pair 同时计算：可能的实际增益、考虑不确定性后的下界、实验带来的信息增益、预算占用和科学风险。最大 robust utility 为正才执行，否则直接 Stop。Stop 因而是与 Feature、Model、Data 正式竞争的动作。

## 8. Atomic matched execution and evidence grading  
## 原子化配对执行与证据分级

The principal EC–VOI pair templates were:

- Feature: rich 2D + ExtraTrees versus Morgan + the same ExtraTrees;
- Model family: Random Forest versus ExtraTrees on rich 2D;
- Imbalance: class-weighted versus unweighted ExtraTrees;
- Data quality: leakage-safe/quality-filtered versus identity-safe data under the same LightGBM or ExtraTrees.

For three ordered seeds, the Agent reserved six fits atomically—treatment and control on every seed—before executing any fit. Each run independently reconstructed the discovery split, applied the identity-safety floor, computed or loaded content-addressed features, fitted the registered learner, stored molecule-level validation predictions, and recorded score, runtime, status, and artifact paths.

A PairExecution was complete only when all treatment and control run identifiers were present and unique, returned seeds matched the reservation, no fit failed, and the single-axis and complete-pair flags were true. A pair exceeding the reserved fit-time ceiling caused a protocol error. Incomplete or failed pairs remained in the ledger but did not update the EC–VOI posterior.

For treatment \(T\), control \(C\), seed \(s\), metric \(m\), and direction \(d\),

\[
\Delta_s=d[m_s(T)-m_s(C)],
\]

where \(d=+1\) for maximized metrics and \(d=-1\) for minimized metrics. The manuscript term EvidencePacket corresponds to MatchedActionEvidence in the standard loop and MatchedEffectObservation in sequential EC–VOI.

In the standard loop, at least three completed paired seeds were required. Unless an endpoint-specific protocol overrode the generic rule:

- supported: mean \(\Delta\geq0.005\) and at least 80% positive seeds;
- refuted: mean \(\Delta\leq-0.005\) and at least 80% negative seeds;
- inconclusive: all other cases, including missing pairs or insufficient seeds.

The Replicate action required two candidates over at least five completed seeds. Its winner was supported only when mean effect was non-negative and at least 80% of paired seeds favoured it. EC–VOI observations stored the mean native effect and

\[
\operatorname{SE}(\Delta)
=\max
\left[
0.0005,
\frac{\operatorname{SD}(\Delta_s)}{\sqrt{n_{\mathrm{seed}}}}
\right].
\]

The floor prevented a zero empirical variance from creating arbitrary certainty.

During broad recipe construction, a two-seed futility gate stopped an unprotected candidate when it was worse than the incumbent on both seeds, the mean oriented gap was at least 0.005, and its time was at least threefold greater. Treatment/control candidates required for causal evidence were protected from this pruning.

### 中文

一个三-seed pair 等于六次预留训练。不能先运行 treatment、看到好分数后再补 control。六次运行、seeds 和单轴校验必须全部完整，结果才能更新 posterior。

训练完成也不等于 hypothesis 成立。系统按 seed 计算 paired delta；至少三个 paired seeds、平均效应达到 0.005 且 80% 同向才 supported，对称负效应才 refuted，其他均为 inconclusive。

## 9. Evidence-dependent state update  
## 证据如何改变下一步

The update was deterministic:

1. **Supported:** promote the candidate, store the outcome, and stop redundant validation search.
2. **Refuted:** do not repeat the action family; if bounded budget remained, select the highest-ranked unattempted executable action on another axis.
3. **Incomplete evidence:** if fewer than three paired seeds completed or the axis check failed, repair the same pair or missing seed coverage.
4. **Complete but inconclusive:** switch once to an unattempted axis when budget allowed; otherwise stop to limit validation overfitting.
5. **Failed or unsafe:** retain the failure, quarantine the artifact/source, and forbid score-based promotion.

A child action received a new attempt identifier and explicit parent link. Scheduling did not count as self-correction; the child experiment had to complete and produce linked evidence. The default bounded policy allowed at most two actions per endpoint unless another limit was preregistered.

In sequential EC–VOI, a complete local observation was appended immediately, all remaining pairs were reranked, and the Agent selected the next pair or Stop in the same episode. In the standard loop, a hashed correction plan was emitted to a registered follow-up queue; process metrics counted only completed children.

Ordinary cross-endpoint outcomes were stored but could not route a new task. Transport required a prospective, same-budget randomized assignment or complete action grid, a shared protocol hash, at least five eligible endpoints, and variation in outcomes. Contextual routing further required at least three effective context-weighted endpoints and maximum context similarity of at least 0.35.

### 中文

Agent 的“学习”不是更新语言模型参数，而是更新实验策略：supported 后 promotion/stop；refuted 后禁止重复同一 action family 并有限换轴；证据结构不完整则补 pair/seed；证据完整但效应太小则换轴或停止；unsafe 则 quarantine。只有 child experiment 真正完成才算 self-correction。

## 10. Finalist selection, freeze, and certification  
## Finalist 选择、冻结与认证

Only completed multi-seed candidates entered finalist selection. Broad-search selection retained the control and treatment needed for causal evaluation, score–runtime Pareto candidates, and limited model diversity. An ensemble was eligible only after every member was independently evaluated and the blend exceeded its stronger member by mean oriented effect at least 0.005 with at least 80% positive paired seeds.

Score-only selection used the best oriented mean. LCB selection used

\[
Q_i=ds_i-\lambda_u\sigma_i,
\]

and compute-aware selection used

\[
Q_i=ds_i-\lambda_u\sigma_i
-\lambda_c
\frac{\log(1+t_i)}
{\max_j\log(1+t_j)}.
\]

Ties were resolved by lower elapsed time and deterministic run identifier. The finalist serialized representations, preprocessing, model and hyperparameters, data policy, optional ensemble weights, seeds, dependencies, and source snapshot.

The freeze service calculated SHA-256 over normalized finalists and task identities before the test interface was opened. A different finalist could not replace an existing freeze in the same finalization directory. Certification required at least five unique seeds, refitted the frozen recipe from discovery train and validation data, and generated predictions in an executor that did not invoke the planner or action memory. Certification could update ClaimRecord but not the finalist.

### 中文

Action evidence 回答“某项干预是否有效”，finalist selection 回答“最终冻结哪个完整 recipe”，两者不能混在一起。只有完成 multi-seed 验证的 candidate 才能进入 selection。冻结在 test 打开之前完成；独立 executor 从 FrozenRecipe 重建，认证结果不能返回 planner。

## 11. Algorithm 1: autonomous research of one endpoint  
## 算法 1：自主研究一个端点

~~~text
Input:
    endpoint q; discovery data Dtrain and Dvalid;
    metric m and direction d; ordered seeds;
    action/candidate/fit-time budget B;
    frozen protocol, code, context, and optional memory snapshots

1.  Resolve q to EndpointContract E and validate ownership and hashes.
2.  If a qualified FrozenRecipe exists and no new trigger is present:
        return REUSE_AND_STOP with zero research fits.
3.  Canonicalize and audit Dtrain and Dvalid.
4.  Build TaskProfile and six-dimensional HardnessCard H.
5.  Detect executable faults; repair, quarantine, or abstain on contract faults.
6.  Map H to registered Feature, Model, Data, Replicate, and Stop actions.
7.  Compile eligible actions to single-axis treatment/control ActionPairs.
8.  Reject protected-field, off-axis, unavailable, or over-budget pairs.
9.  Load only transport-eligible discovery evidence from frozen memory.
10. Rank pairs by effect-first or EC–VOI utility.
11. If no pair has positive admissible utility: emit STOP.
12. Otherwise atomically reserve all treatment/control fits, seeds, and time.
13. Execute both arms under identical non-intervened conditions.
14. If incomplete/unsafe: record failure and do not update effect posterior.
15. If complete: calculate seed-wise oriented effects and evidence status.
16. Append decision, reservation, runs, evidence, cost, and hashes to ledger.
17. Update state:
        supported  -> promote and stop/reselect;
        refuted    -> bounded switch to an untried axis;
        incomplete -> repair pair or seed coverage;
        low effect -> switch once or stop;
        unsafe     -> quarantine and stop.
18. Repeat from Step 10 only if the registered episode permits another action.
19. Select a finalist only from complete, protocol-valid candidates.
20. Serialize and hash FrozenRecipe before certification access.
21. Reconstruct in a planner-independent executor and emit ClaimRecord.
~~~

## 12. Drop-in manuscript text  
## 可直接替换主文的核心正文

### English

ADMET-EvO implemented autonomous molecular-property research as a sequence of budgeted, executable treatment–control decisions. For each endpoint, the system constructed an EndpointContract and profiled the discovery split into a six-dimensional HardnessCard describing scaffold scarcity, class imbalance, chemical-domain shift, local structure–activity roughness, data quality, and identity leakage. These quantities did not directly select a model. Instead, they activated registered hypothesis families that compiled to concrete single-axis ActionPairs, such as rich two-dimensional features versus Morgan fingerprints under the same ExtraTrees learner, Random Forest versus ExtraTrees under the same features, or quality-filtered versus identity-safe data under the same learner.

At each active step, the research policy ranked executable pairs using discovery-only evidence. The sequential EC–VOI policy combined a posterior lower bound on practically relevant paired improvement, expected information gain, normalized fit/candidate cost, and scientific risk. A pair was executed only when its robust utility was positive and the complete treatment–control budget could be reserved in advance. Treatment and control were then run on the same ordered validation seeds. Failed or incomplete pairs remained in the lineage but could not update the effect posterior.

Completed runs were converted to evidence from seed-wise oriented effects. An intervention was supported only when its mean paired effect was at least 0.005 and at least 80% of paired seeds agreed in direction; the symmetric negative rule defined refutation, and all other outcomes were inconclusive. Depending on the verdict, the Agent promoted the candidate, repaired missing evidence, switched once to an untried axis, quarantined an unsafe input, or stopped. Thus, research state changed only through completed matched evidence, not through the planner's textual rationale.

Only protocol-valid multi-seed candidates entered finalist selection. The selected recipe was serialized with its representation, preprocessing, learner, hyperparameters, data policy, seeds, dependency versions, and source hashes. Its SHA-256 freeze was written before certification data were opened. A separate executor reconstructed the frozen recipe without invoking the planner or action memory; certification could update only the claim record, not the finalist.

### 中文

ADMET-EvO 将自主分子性质研究实现为一系列受预算约束、可以真实执行的 treatment–control 决策。系统首先建立 EndpointContract，并用 discovery 数据形成六维 HardnessCard。诊断量不会直接指定模型，而是激活注册 hypothesis，并编译成唯一变化轴的 ActionPair。

研究策略利用 discovery-only evidence 对 pairs 排序。EC–VOI 同时考虑实际提升下界、信息增益、计算成本和科学风险；只有 robust utility 为正且完整 pair 预算能够预留时才执行。Treatment/control 使用相同 seeds，失败或不完整 pair 不能更新 posterior。

完整实验通过 seed-wise paired effects 分为 supported、refuted 或 inconclusive。Agent 据此 promotion、补齐证据、有限换轴、quarantine 或 stop。最终只有合规的 multi-seed candidates 进入 selection；recipe 在 certification data 打开前冻结，并由独立 executor 重建。

## 13. Central methodological contribution  
## 最需要突出的贡献

ADMET-EvO 的重点不是“使用了很多模型和 Agent 模块”，而是四项可检验设计：

1. **Diagnosis-to-experiment compilation**：把化学和数据诊断编译成真实单轴实验；
2. **Metric-valued experiment acquisition**：用 effect、uncertainty、information gain、cost 和 risk 决定下一实验或 Stop；
3. **Evidence-owned state transition**：只有完整 treatment/control evidence 能改变 research state；
4. **Planner-independent finalization**：发现产物在认证前冻结，认证不能重新开启搜索。

普通 AutoML 在固定的 \(X/y/metric\) 上选择配置；ADMET-EvO 还决定 **应该检验哪一种科学干预、需要多少证据、外部数据是否有资格作为训练证据，以及何时不应继续实验**。
