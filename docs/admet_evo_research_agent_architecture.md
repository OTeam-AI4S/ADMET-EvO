# ADMET-EvO Research Agent：改进版总体架构与逻辑链条

> 状态：v5 AI Scientist 评测实施中（仅限 TDC ADMET 22 endpoints，既有 test 结果只读封存）  
> 目标：预测指标以“应用合格”为约束，不再刷榜；主目标改为量化 Agent 相比传统 AutoML 的诊断、证伪、自我纠错、数据治理、成本感知停止和可审计科研闭环优势。

## 0. v5 当前主链：性能合格后，评价“会不会研究”

```mermaid
flowchart TD
    P["既有 TDC-22 冻结结果<br/>18/22 Top-5；21/22 Top-10"] --> Q{"Application Gate<br/>15 Top-5；21 Top-10"}
    Q -->|"通过；禁止回流选模"| V["Validation-only Research Benchmark"]
    Q -->|"未通过"| B["阻断 AI Scientist claim"]

    V --> H["Hardness + Fault Diagnosis"]
    H --> Y["可证伪假设 + 单轴 matched experiment"]
    Y --> R["seed-2 公平 racing<br/>gap≥0.005 且 cost≥3x"]
    R --> E{"supported / refuted / inconclusive"}
    E -->|"supported"| M["性能已合格：记忆并停止冗余搜索"]
    E -->|"refuted"| C["Correction Scheduler<br/>换未尝试实验轴"]
    E -->|"inconclusive"| A["修复 matched pair/seed 覆盖<br/>或低信息停止"]
    C --> CE["执行纠正实验并链接 parent action"]
    A --> CE
    CE --> CC{"completed outcome?"}
    CC -->|"是"| H
    CC -->|"仅 scheduled"| NC["不计 self-correction 成功"]
    M --> H

    D["外部数据 dossier"] --> S["来源、许可、assay、species、unit、transform、checksum"]
    S --> G{"Data Admission Gate"}
    G -->|"安全且对齐"| SH["仅 source-held-out 验证"]
    G -->|"需桥接"| K["quarantine"]
    G -->|"危险/无许可"| X["reject"]

    V --> C1["Agent"]
    V --> C2["Expanded AutoML<br/>同模型/特征/数据原语"]
    V --> C3["Conventional AutoML / Rules / Random"]
    V --> C4["no-hardness / no-memory / no-data"]

    C1 --> Z["Research Capability Scorecard v3"]
    C2 --> Z
    C3 --> Z
    C4 --> Z
    SH --> Z
    K --> Z
    X --> Z

    Z --> Z1["Foundation：应用性能、critical faults、零 test 反馈"]
    Z --> Z2["九维不可补偿过程指标"]
    Z --> Z3["分项 Agent-control delta + CI"]
    Z --> Z4["独立 flags：process advantage / frozen application / AutoML predictive noninferiority"]
    Z3 --> CL["Claim Ladder<br/>证据到哪，只讲到哪"]
```

v5 的正式预注册是 `runs/v5-policy-benchmark-90m/protocol.json`，哈希为
`ff01a79f6db0bb6d51a2abff3bcec1becde54827265ad720f525514d0858f4a7`。每个策略每个
endpoint 的整体计算硬上限为 90 分钟；多角色只共享一个执行队列，不增加 action/fit
预算。既有 TDC test 不再承担新策略选择或迁移性证明。

当前 self-correction 机制已实现“证据 → 下一纠正计划”的确定性、validation-only、哈希
可审计转换，并接入主 runner。它不会把 scheduled 计划当成已完成能力；只有后续 matched
experiment 实际执行并产生链接 outcome，过程评分才会增加。这样防止用更多文字步骤替代
真实实验，也防止在 refuted 后无意义重复同一动作族。

第一轮前瞻 smoke 已按该链执行：22 个历史 endpoint 中 18 个 supported 直接停，另有 2 个
因下一步超过 compute-cost gate 而停；仅 CYP2C9_Substrate 和 hERG 进入执行队列。2/2
纠正动作完成，共 12 fits/67.6 fit-s，无 Uni-Mol2、调参或 ensemble。两个 child hypotheses
均 inconclusive，最终再次停止。因此当前证据支持“能把失败变成受控 follow-up 并接受
无结论”，不支持“纠正后一定涨分”或“优于 complete rules”。

fresh-validation 成本证书给出的当前边界是：相对 score-only selection，mean oriented
delta = −0.000348，95% CI [−0.001482, 0.000585]，20/22 endpoints 在 −0.005 实用界限
内，aggregate selected-recipe cost ratio=0.278。它证明有界 Pareto efficiency，不证明
路由优于随机或 Agent 优于 AutoML。

Scorecard v3 不再把“预测非劣于 expanded AutoML”错误地放在所有 Agent claim 的必经路径
上。应用性能合格与安全构成 foundation；科研过程优势、冻结应用认证、以及预测非劣分别
出具独立 flags。这样旧 Agent-vs-AutoML 的负预测证据会阻断 predictive-noninferiority
flag，但不会抹掉已经验证的诊断、治理或自纠错过程能力。

## 1. 核心目标

改进后的 ADMET-EvO 不只回答：

> 哪个模型在当前验证集上分数最高？

而要完整回答：

1. 这个 endpoint 为什么难？
2. 当前主要问题来自数据、表示、模型、评价协议，还是验证方差？
3. 下一步最有价值的研究动作是什么？
4. 该动作是否有可证伪的实验设计？
5. 验证集上的提升是否能迁移到独立 held-out 数据？
6. Agent 的决策是否优于同成本的随机、规则和传统 AutoML？
7. 当模型超出适用域时，系统是否知道应该拒绝预测或请求新数据？

## 2. 总体架构

```mermaid
flowchart TD
    A["原始 endpoint 数据<br/>结构、标签、来源、assay 元数据"] --> B["Endpoint Contract<br/>定义性质、物种、assay、单位、指标"]
    B --> C["Protocol Guard<br/>冻结 split、metric、test 权限和计算预算"]
    C --> D["Hardness Profiler<br/>生成多维 HardnessCard"]
    D --> E["Problem Diagnostician<br/>提出根因、证据和可证伪测试"]
    E --> F["Action Router<br/>选择下一步研究动作"]

    F --> G1["Feature Action<br/>描述符、SMARTS、文献规则"]
    F --> G2["Model Action<br/>模型族、loss、target transform、集成"]
    F --> G3["Data Action<br/>外部证据、assay 对齐、来源审核"]
    F --> G4["Replicate Action<br/>多 seed、重采样、稳定性复核"]
    F --> G5["Abstain / Stop<br/>停止搜索、拒绝预测或请求新实验"]

    G1 --> H["Axis-Isolated Executor<br/>一次只允许一个研究轴变化"]
    G2 --> H
    G3 --> H
    G4 --> H
    G5 --> H

    H --> I["Validation Evaluator<br/>多 seed、OOF、scaffold/OOD 划分"]
    I --> J["Evidence Ledger<br/>记录成功、失败、成本、预测和代码哈希"]
    J --> K{"假设是否被支持？"}

    K -->|"否"| E
    K -->|"证据不足"| F
    K -->|"是"| L["Candidate Promotion<br/>LCB、风险和成本联合筛选"]

    L --> M["Freeze Lock<br/>序列化完整 recipe 并计算 SHA-256"]
    M --> N["Held-out Certification<br/>冻结后从头训练并只读盲测"]
    N --> O["Claim Evaluator<br/>预测、迁移、Agent 增量、自主诊断"]
    O --> P["只读审计与论文证据包"]

    N -.->|"不得回灌同一 test"| Q["下一版本使用新的 future audit set"]
```

### 2.0 当前默认候选：生产 / 科研两线极简链（v2）

预测已经通过应用资格门后，不再把 22 个任务全部当成待优化问题。A/D/M/E/T 只决定
endpoint、assay、单位、物种和数据治理 contract；建模问题只分成四类：普通 AUROC
二分类、不平衡 AUPRC 二分类、MAE 连续值回归和 Spearman 排序型 PK 回归。数据量只保留
两个有不同动作的档位：low-data（`N<1000`，13 tasks）和 higher-data（`N>=1000`，
9 tasks）。旧 medium 桶没有独立动作，已从默认调度删除。

```mermaid
flowchart LR
    T["Task"] --> K{"Endpoint/data contract"}
    K -->|"失败"| Q["Quarantine / repair<br/>0 model fits"]
    K -->|"通过"| F{"Frozen artifact available?"}
    F -->|"否"| A["Restore artifact<br/>不重新搜索"]
    F -->|"是"| P{"明确的 validation-side problem?"}
    P -->|"否"| X["Reuse qualified recipe and stop<br/>0 research fits"]
    P -->|"数据/漂移"| D["One data-stewardship ticket<br/>先不训练"]
    P -->|"性能/不确定性"| E["One hypothesis + one matched pair"]
    E --> R["Retain supported/refuted/inconclusive<br/>stop"]
```

默认生产线只有 contract → frozen artifact → reuse/stop。当前 22/22 资格通过且没有新异常，
因此零训练回放的默认新增科研 fits 为 0。这里的 0 不表示模型推理免费，而是不再为已经合格
的任务追加搜索训练。

只有四类 validation-side 信号可以启动异常线：contract 失败、冻结 artifact 缺失、资格
退化/高不确定性、数据漂移/来源缺口。被触发的预测问题也只允许一个假设和一个单轴 matched
pair：low-data 使用 5 paired seeds（最多 10 fits），higher-data 使用 3 paired seeds（最多
6 fits）；supported、refuted 或 inconclusive 都写入 memory 后停止，不自动切第二个轴。

因此从健康任务的默认调度继续删除：每任务 matched pair、每任务 strong anchor、screen 后再补
5 seeds、自动跨轴 correction、scalar hardness routing、committee、HPO、ensemble 和 UniMol2。
外部数据搜索也不会在 22 个健康任务上机械全开；只有 endpoint contract 明确发现数据缺口时，
才创建独立的数据研究票据，并走下面的准入闭环。旧 category × small/medium/large 三阶段 policy
及其 252-fit ceiling 保留为历史 shadow evidence，不再是默认候选。实现与 22-task 表见
`reports/lean_category_size_policy_v2.md`。

### 2.1 条件多角色：默认单 Agent，按风险激活

默认始终由一个 Agent 决策和执行。多 Agent 不再参与普通预测、候选生成、模型投票或
finalist selection；只有外部数据准入、contract 冲突或破坏性 policy 变更，才增加一名
targeted reviewer。reviewer 只审计证据与安全边界，不能增加候选或训练预算。

```mermaid
flowchart LR
    S["Single Agent"] --> R{"Data/safety risk?"}
    R -->|"否"| E["Execute or stop"]
    R -->|"是"| V["One targeted reviewer"]
    V --> G["Approve / quarantine / request evidence"]
    G --> E
```

旧条件式委员会的 22-task 回放（20 single、1 targeted、1 full）只保留为历史机制证据；
它没有证明预测增益，因此不进入 v2 默认调度。

### 2.1.1 复合研究决策验证（历史过程证据）

本地 commit/reveal composite challenge 的 10 个 episode 将这些约束同时呈现。Agent
直接调用生产 planner，取得 10/10 正确、0 unsafe、30 estimated fits；locked static
rules 为 6/10、3 unsafe、48 fits，expanded AutoML 为 4/10、1 unsafe、48 fits。该证据
表明闭环不是把多个独立 checklist 简单相加，但由于 challenge 仍由同一团队编写，只能
视为 policy-conformance 测试；它不要求生产任务执行这些步骤。下一步仍需外部作者先提交
commitment 后再盲测。

### 2.2 防止“操作越多、效果越差”的五道门

| 门槛 | v2 规则 | 防止的问题 |
|---|---|---|
| 资格停止 | 已合格且无新问题就复用并停止 | 为刷分重复搜索 |
| 研究票据 | 每次只有 1 hypothesis × 2 configs | 多轮 action 扩张 |
| 数据优先 | contract/drift 问题先治理，0 model fits | 用模型掩盖数据问题 |
| 昂贵组件 | HPO、ensemble、UniMol2、外部数据均不在默认调度 | 低收益高成本分支 |
| Artifact | 先恢复/复用冻结 recipe 与共享特征缓存 | 重复 checkpoint 和表示文件 |

### 2.3 新主线：自动找数据、判 endpoint、密封去污染

这条链是 Agent 相比传统 AutoML 最重要的新差异。传统 AutoML 对输入的 `X/y/metric` 进行
搜索；这里 Agent 必须先判断外部记录是不是在测同一个科学问题，随后才能训练。

```mermaid
flowchart LR
    C["Endpoint Contract"] --> S["Source Scout<br/>同一 source/query/API budget"]
    S --> J{"Endpoint Judge"}
    J -->|"identity/species/unit/license 错"| R["Reject + evidence"]
    J -->|"assay/threshold 尚不等价"| B["Bridge-required quarantine"]
    J -->|"可继续"| A["Frozen acquisition"]
    B --> A
    A --> F["Sealed structure firewall"]
    F --> F1["脱盐/去电荷/互变异构规范化<br/>标准 InChIKey 去 exact overlap"]
    F1 --> F2{"InChIKey skeleton 命中 >5% test molecules?"}
    F2 -->|"是"| R
    F2 -->|"否"| F3["删除 ECFP4 Tanimoto >=0.90 test neighbors"]
    F3 --> H["Source-held-out matched data experiment"]
    H --> O{"supported / refuted / inconclusive"}
    O --> L["Ledger + stop/next hypothesis"]
```

Test 结构是这条链唯一的例外权限：密封 executor 只读取 `Drug` 列用于排除，不读取标签，
不把原始 test 结构或相似度反馈给 planner，也不允许它据此选择模型或 threshold。审计字段固定为
`test_structures_used_for_decontamination=true`、`test_labels_loaded=false` 和
`test_structures_exposed_to_planner=false`。Agent 与 AutoML 的 data arm 使用完全相同的来源
registry、查询/API 调用预算和三重防火墙；比较的是 endpoint 判断、准入理由、实验设计和停止，
不是谁拿到更多数据。

当前 hERG scout 能发现 ChEMBL 与两个 TDC-related 候选，但 3/3 均判为 `bridge_required`，
0 个允许直接合并。真实 ChEMBL 获取已经两次在官方 API HTTP 500 后 failed closed：没有接收
半截 raw data、没有训练、没有 unsafe merge。因此机制已实现，真实 source-held-out 增益仍未
证明。详见 `reports/external_data_agent_capability_v1.md` 与
`reports/chembl_herg_external_ticket_v1.md`。

### 2.4 有界图表征融合：2D + GIN + MPNN + OOF gate

图模型不是重新开启 22-task model zoo。它只是一张不超过 20 fits 的单任务、两阶段科研票据：

1. Stage A 只改变 representation：`2D + ExtraTrees` 对
   `2D+GIN+MPNN static concat + same ExtraTrees`，5 paired seeds、10 fits。
2. 只有 Stage A 的 validation paired mean delta ≥0.005 且 calibration 不退化，才运行 Stage B。
3. Stage B 只改变 fusion：static concat 对 OOF-safe shallow decision-tree gate，最多 10 fits。
4. 未过门即保留负结果并停止；禁止 HPO、UniMol2、committee、扩张 ensemble 与 test feedback。

当前已有 pinned MapLight GIN、共享内容寻址 cache，以及可执行的 OOF decision-tree gate。
Gate 只用训练集 out-of-fold expert predictions 学习，再把各 expert 在完整 train 上重拟合，避免
用 in-sample 置信度训练虚假路由。独立 MPNN provider 尚不存在，因此票据在训练前明确阻断；
GIN 不能改名冒充 MPNN，系统也不会自动安装另一个数 GB 环境。完整协议见
`reports/bounded_graph_fusion_ticket_v1.md`。

## 3. 三条并行逻辑链

系统同时维护三条链，不能只保留最终最佳分数。

```mermaid
flowchart LR
    subgraph Scientific["科学推理链"]
        S1["问题"] --> S2["假设"] --> S3["可证伪实验"] --> S4["结果"] --> S5["结论或修正"]
    end

    subgraph Execution["可执行谱系链"]
        E1["ResearchAction"] --> E2["受控配置或补丁"] --> E3["训练与预测"] --> E4["产物哈希"] --> E5["可重现 snapshot"]
    end

    subgraph Evidence["评价证据链"]
        V1["Validation"] --> V2["多 seed 稳定性"] --> V3["外层 held-out"] --> V4["迁移差距"] --> V5["支持的 claim"]
    end

    S3 --> E1
    E3 --> V1
    V5 --> S5
```

## 4. Endpoint Contract

每个 endpoint 必须先被转换成结构化合同，避免只凭一个任务名称推断标签含义。

```yaml
endpoint_id: cyp2c9_substrate
property: CYP2C9 substrate status
task_type: binary_classification
species: human
assay_system: curated substrate evidence
label_definition:
  positive: substrate
  negative: non_substrate
unit: null
metric:
  primary: pr_auc
  direction: maximize
source:
  dataset: TDC ADMET
  provenance: Carbon-Mangels et al.
split_contract:
  discovery: scaffold_multi_seed
  certification: evaluator_owned_holdout
```

Endpoint Contract 应至少包含：

| 字段 | 含义 |
|---|---|
| `property` | 预测的生物学或理化性质 |
| `species` | 人、大鼠、小鼠或其他体系 |
| `assay_system` | 实验体系和读出方式 |
| `label_definition` | 分类阈值或回归标签定义 |
| `unit` | 原始单位、对数尺度和变换方式 |
| `metric` | 主指标、辅助指标及方向 |
| `source` | 数据来源、年份、许可和引用 |
| `split_contract` | discovery、certification 和 future audit 的所有权 |

## 5. Hardness Profiler

### 5.1 Hardness 不是单一分数

系统首先输出多维 `HardnessCard`，保留每一种困难的原始证据。

```mermaid
flowchart TD
    A["Endpoint 数据"] --> H1["Scarcity<br/>样本量、有效 scaffold、学习曲线"]
    A --> H2["Imbalance<br/>阳性率、少数类数量、标签熵"]
    A --> H3["OOD / Shift<br/>相似度、scaffold novelty、domain AUC"]
    A --> H4["Landscape Roughness<br/>MODI、SARI、ROGI"]
    A --> H5["Activity Cliffs<br/>cliff 密度、cliff pair recall"]
    A --> H6["Assay / Label Noise<br/>重复测量、来源效应、冲突标签"]
    A --> H7["Selection Variance<br/>跨 seed 方差、rank instability"]
    A --> H8["Baseline Saturation<br/>当前分数、headroom、提升难度"]

    H1 --> C["HardnessCard"]
    H2 --> C
    H3 --> C
    H4 --> C
    H5 --> C
    H6 --> C
    H7 --> C
    H8 --> C
```

### 5.2 建议的 HardnessCard

```json
{
  "endpoint": "CYP2C9_Substrate_CarbonMangels",
  "scarcity": {
    "n_train": 428,
    "effective_scaffolds": 117,
    "learning_curve_slope": 0.31
  },
  "imbalance": {
    "positive_rate": 0.18,
    "minority_count": 77,
    "severity": "high"
  },
  "distribution_shift": {
    "median_nearest_train_similarity": 0.39,
    "novel_scaffold_fraction": 0.72,
    "domain_classifier_auc": 0.81
  },
  "roughness": {
    "modi": 0.42,
    "sari": 0.37,
    "rogi": 0.61
  },
  "activity_cliffs": {
    "pair_fraction": 0.08,
    "baseline_pair_recall": 0.11
  },
  "selection_variance": {
    "validation_std": 0.024,
    "top_candidate_rank_stability": 0.43
  },
  "baseline_saturation": {
    "baseline_score": 0.373,
    "headroom": "high"
  }
}
```

### 5.3 Hardness 到动作的初始映射

| 主要难度 | 优先动作 | 不应默认采取的动作 |
|---|---|---|
| 小数据 | 简单模型、预训练表示、外部证据 | 从头训练大模型 |
| 极端不平衡 | balanced ensemble、PR-AUC、阈值/校准分析 | 只看 ROC-AUC |
| scaffold/OOD | 2D/3D 对照、适用域、拒绝预测 | 无条件融合所有模型 |
| roughness/activity cliff | cliff-aware loss、局部模型、matched-pair 特征 | 只做普通超参数调优 |
| assay 噪声 | 来源分层、leave-one-source-out、噪声模型 | 直接合并更多数据 |
| selection variance | 多 seed 复核、LCB、seed ensemble | 增加同一验证集 trial 数 |
| baseline 饱和 | 停止、转向新证据或新评价维度 | 追逐极小验证增益 |

## 6. Problem Diagnostician

Diagnostician 不直接修改模型。它先产生一个可审核的 `ProblemHypothesis`。

```json
{
  "problem_id": "ph_0042",
  "endpoint": "Clearance_Microsome_AZ",
  "suspected_root_cause": "selection_variance",
  "confidence": 0.82,
  "evidence": [
    "top-5 candidate ranks change across validation seeds",
    "candidate differences are smaller than bootstrap uncertainty",
    "validation gain increases with trial count but LCB does not"
  ],
  "falsification_test": {
    "action": "repeat_top_candidates",
    "seeds": [6, 7, 8, 9, 10],
    "expected_signature": "winner rank instability"
  },
  "recommended_next_action": "replicate",
  "alternative_hypothesis": "dataset-specific model mismatch"
}
```

诊断状态机：

```mermaid
stateDiagram-v2
    [*] --> Proposed
    Proposed --> Testing: "提交可证伪实验"
    Testing --> Supported: "观察到预期签名"
    Testing --> Rejected: "证据与假设冲突"
    Testing --> Inconclusive: "不确定性过大"
    Inconclusive --> Testing: "增加复核证据"
    Rejected --> Revised: "生成替代假设"
    Revised --> Testing
    Supported --> Actionable: "允许进入研究动作选择"
    Actionable --> [*]
```

## 7. ResearchAction 统一协议

所有特征、模型、数据、复核和停止行为都使用同一种结构。

```yaml
action_id: action_0107
parent_action_id: action_0099
endpoint: half_life_obach
axis: data
hypothesis: >
  Independent FDA clinical-pharmacology records improve coverage of
  long-half-life compounds without importing the benchmark source.
expected_effect:
  primary_metric_delta: 0.03
  subgroup: low_similarity
falsification_condition: >
  Multi-seed validation LCB is non-positive or leave-one-source-out
  performance degrades.
editable_assets:
  - external_evidence/half_life_obach.csv
budget:
  max_rows: 500
  max_trials: 1
  max_minutes: 60
provenance:
  source: FDA clinical pharmacology records
  citation: required
  license: required
```

每个 action 必须回答：

1. 改什么？
2. 为什么改？
3. 哪些证据支持它？
4. 什么结果会否定它？
5. 允许修改哪些资源？
6. 成本上限是多少？
7. 如何判断成功、失败或证据不足？

## 8. 五类研究动作

### 8.1 Feature Action

优先使用声明式特征，而不是开放任意 Python 编辑。

```yaml
axis: feature
endpoint: herg
feature_type: smarts_alert_set
features:
  - name: lipophilic_basic_amine
    smarts: "..."
  - name: benzhydryl
    smarts: "..."
evidence:
  citation: required
```

受控编译器负责：

- SMARTS 语法检查；
- 计算复杂度限制；
- 常量特征检测；
- NaN/Inf 检查；
- 标签和 split 信息隔离；
- 特征产物哈希。

### 8.2 Model Action

允许的模型动作包括：

- 模型族选择；
- 受限超参数；
- target transform；
- metric-aligned loss；
- activity-cliff-aware loss；
- calibration；
- multi-seed ensemble；
- 2D/3D/多模态组合。

模型动作不能修改：

- 数据划分；
- 主评价指标；
- held-out test loader；
- 最终 evaluator；
- freeze manifest。

### 8.3 Data Action

```mermaid
flowchart LR
    A["外部来源"] --> B["Assay Contract 检查"]
    B --> C["标准化身份去重"]
    C --> D{"同源重叠是否超过阈值？"}
    D -->|"是"| X["拒绝整个来源并记录"]
    D -->|"否"| E["近邻 / Tanimoto 过滤"]
    E --> F["单位、标签方向和 censoring 检查"]
    F --> G["化学空间与 domain classifier 审计"]
    G --> H["source-only 和 leave-one-source-out 验证"]
    H --> I{"是否有可迁移证据？"}
    I -->|"否"| J["拒绝、降权或隔离来源"]
    I -->|"是"| K["受控合并或加权训练"]
```

数据轴的 `AssayContract` 至少记录：

- source、publication、license；
- species；
- assay protocol；
- measurement unit；
- label direction；
- classification threshold；
- censoring；
- experimental year/laboratory；
- 与目标 endpoint 的兼容性判断。

### 8.4 Replicate Action

当主要风险是验证方差时，最有价值的下一步不是提出新模型，而是复核：

- 新 validation seeds；
- bootstrap；
- OOF；
- 重复训练；
- candidate rank stability；
- ensemble variance decomposition。

### 8.5 Stop / Abstain Action

系统必须允许以下结论：

- 没有研究动作超过最小效应阈值；
- endpoint 已基线饱和；
- 当前样本不足以区分候选；
- 分子超出适用域；
- 外部数据 assay 不匹配；
- 应请求新实验，而不是继续搜索模型。

## 9. Axis Isolation 和安全边界

```mermaid
flowchart TB
    subgraph AgentEditable["Agent 可编辑区域"]
        F["Feature manifest"]
        M["Model recipe"]
        D["External evidence submission"]
    end

    subgraph HarnessOwned["Evaluator / Harness 所有区域"]
        S["Split registry"]
        P["Protocol guard"]
        E["Metric evaluator"]
        T["Held-out test labels"]
        L["Leakage and assay audit"]
        Z["Freeze lock"]
    end

    F --> X["Axis-Isolated Executor"]
    M --> X
    D --> X
    S --> X
    P --> X
    L --> X
    X --> E
    T --> E
    E --> Z
```

执行规则：

1. 每个 trial 只开放一个 axis；
2. trial 前恢复所有非目标文件；
3. trial 前后比较冻结资源哈希；
4. 发生修改、删除或旁路即判为 tampered；
5. evaluator、metric、split 和 test 永远不进入可写区；
6. 外部数据只能通过受控工具进入训练。

## 10. Action Router

### 10.1 目标函数

Router 不只最大化当前验证分数。

\[
U(a)=
\mathbb{E}[\Delta_a]
-\lambda_{\sigma}\sigma_a
-\lambda_c C_a
-\lambda_d D_a
-\lambda_h P(\text{harm}\mid a)
\]

其中：

- \(\mathbb{E}[\Delta_a]\)：动作的预期收益；
- \(\sigma_a\)：收益不确定性；
- \(C_a\)：训练、API、GPU 和数据获取成本；
- \(D_a\)：数据或化学空间偏移风险；
- \(P(\text{harm}\mid a)\)：动作导致 endpoint 退化的概率。

### 10.2 Router 决策顺序

```mermaid
flowchart TD
    A["读取 HardnessCard 和 Evidence Ledger"] --> B{"当前证据是否稳定？"}
    B -->|"否"| R["Replicate"]
    B -->|"是"| C{"是否存在明显数据问题？"}
    C -->|"是"| D["Audit / Data Action"]
    C -->|"否"| E{"是否是 representation / roughness 问题？"}
    E -->|"是"| F["Feature 或 2D/3D Model Action"]
    E -->|"否"| G{"是否存在 calibration / imbalance 问题？"}
    G -->|"是"| H["Loss、sampling 或 calibration Action"]
    G -->|"否"| I{"预期收益是否超过风险和成本？"}
    I -->|"是"| J["执行最高效用动作"]
    I -->|"否"| K["Stop / Abstain / Request experiment"]
```

## 11. Evidence Ledger

Evidence Ledger 是研究过程的单一事实来源，不只保存赢家。

建议的数据实体：

```text
endpoints
hardness_profiles
problem_hypotheses
research_actions
runs
predictions
external_sources
data_audits
snapshots
freeze_manifests
certification_results
claims
```

每次运行至少记录：

- parent action 和谱系深度；
- hypothesis；
- 完整配置；
- validation seed；
- raw metric 和 normalized improvement；
- validation predictions；
- subgroup metrics；
- wall time、GPU time 和 API 成本；
- failure/crash/tamper 原因；
- artifact SHA-256；
- 是否被保留以及为什么。

失败实验不能删除，因为它们是后续 Agent 避免重复无效研究的证据。

## 12. 三层评价隔离

```mermaid
flowchart LR
    A["原始数据"] --> B["Discovery partition"]
    A --> C["Certification holdout"]
    A --> D["Future audit / external blind set"]

    B --> B1["内部 train"]
    B --> B2["多 seed validation / OOF"]
    B1 --> S["Agent 搜索"]
    B2 --> S
    S --> F["冻结 finalist"]
    F --> C1["Certification 只评价冻结配置"]
    C --> C1
    C1 --> R["发布 claim 和只读审计"]
    R -.->|"若据此继续开发"| D1["下一版本必须使用新的盲测数据"]
    D --> D1
```

### 12.1 Discovery

允许重复访问，但必须使用：

- 多 validation seeds；
- OOF 或 nested CV；
- candidate rank stability；
- LCB；
- 最小效应阈值；
- trial budget。

### 12.2 Certification

- Agent 永远不读取标签；
- 配置 freeze 后才可运行；
- 不能根据结果替换 finalist；
- 不能回灌进 Agent memory；
- 报告 raw metric、增益、置信区间和 subgroup 结果。

### 12.3 Future audit

一旦人类根据 certification 结果修改系统，同一个 certification set 就不再是下一版的独立证据。下一版本需要：

- 新 outer split；
- temporal/source split；
- 外部独立数据；
- 社区 blind challenge；
- 或新生成的实验数据。

## 13. 如何证明 Agent 本身有效

不能只用最终 leaderboard 分数证明 Agent 有效。需要四类独立证据。

| Claim | 必要对照 | 主要指标 |
|---|---|---|
| 预测效果好 | 强固定基线、公开 SOTA | held-out raw metric、win/tie/loss |
| 改进可靠 | validation 与独立 test 配对 | transfer ratio、generalization gap、sign reversal |
| Agent 有增量贡献 | 随机、规则、Optuna/FLAML、matched cost | regret、gain/trial、gain/GPU-hour |
| 能自主发现问题 | Hidden Fault Benchmark | diagnosis precision/recall、修复成功率、harm rate |

### 13.1 建议的对照组

1. Strong fixed MapLight；
2. FLAML/Optuna；
3. Maximum-entropy random action policy；
4. Fixed action schedule；
5. Rule-based ADMET-EvO；
6. 当前 LLM ADMET-EvO；
7. Auto-Research-Molprop；
8. Hardness-aware Research Agent。

所有比较需要匹配：

- split；
- baseline；
- trial 数；
- wall-clock 或 compute；
- LLM/API 预算；
- 可用动作空间；
- 外部数据权限；
- seed。

## 14. Hidden Fault Benchmark

### 14.1 隐藏故障类型

```mermaid
mindmap
  root(("Hidden Faults"))
    Data
      Identity leakage
      Near-analogue leakage
      Same-source reaggregation
      Unit mismatch
      Log/linear mismatch
      Label reversal
      Source confounding
      Conflicting duplicates
    Evaluation
      Wrong metric
      Wrong direction
      Random-split optimism
      Validation reuse
      Single-seed winner's curse
    Model
      Underfitting
      Overfitting
      Calibration failure
      OOD overconfidence
      Activity-cliff failure
      3D conformer failure
      Harmful fusion
```

### 14.2 Agent 输出

Agent 不知道故障标签，只能看到 train/validation 证据，并需要输出：

1. 是否存在异常；
2. 推测的根因；
3. 支持证据；
4. 置信度；
5. 可证伪实验；
6. 修复动作；
7. 预期结果；
8. 替代假设。

### 14.3 评分

| 能力 | 指标 |
|---|---|
| 异常发现 | precision、recall、F1 |
| 根因定位 | exact/partial accuracy |
| 置信度质量 | calibration error、Brier score |
| 假设质量 | falsifiability 和证据充分性 |
| 修复效果 | 独立 held-out improvement |
| 安全性 | harmful intervention rate |
| 效率 | 诊断所需 trial、时间和成本 |

## 15. 实施阶段

```mermaid
gantt
    title ADMET-EvO Research Agent 实施路线
    dateFormat  YYYY-MM-DD
    axisFormat  %m-%d

    section P0 评价合同
    Endpoint Contract 与 claim 预注册       :p0a, 2026-08-19, 10d
    三层 split registry 与 certification gate :p0b, after p0a, 12d

    section P1 Hardness 与诊断
    HardnessCard 基础指标                    :p1a, after p0a, 18d
    Roughness、cliff、OOD 指标                :p1b, after p1a, 20d
    ProblemHypothesis schema                 :p1c, after p1a, 12d

    section P2 研究动作
    ResearchAction 与 axis lock              :p2a, after p1c, 18d
    声明式 feature action                    :p2b, after p2a, 18d
    model 与 replicate action                :p2c, after p2a, 20d

    section P3 外部证据
    AssayContract 与来源审计                 :p3a, after p2a, 20d
    data shift 与 source validation          :p3b, after p3a, 22d

    section P4 路由与证明
    跨轴 Action Router                       :p4a, after p2c, 24d
    Hidden Fault Benchmark                   :p4b, after p1b, 30d
    matched-budget 对照实验                  :p4c, after p4a, 30d

    section P5 外部认证
    36-endpoint 论文协议复现                 :p5a, after p4c, 25d
    新 blind / external certification        :p5b, after p5a, 30d
```

时间只是建议顺序，不是未经资源评估的承诺。

## 16. 现有模块到新模块的映射

| 现有模块 | 保留能力 | 建议新增或扩展 |
|---|---|---|
| `profiling.py` | 数据画像 | `HardnessCard`、roughness、cliff、OOD、学习曲线 |
| `schemas.py` | ExperimentSpec、RunResult | EndpointContract、ProblemHypothesis、ResearchAction |
| `guard.py` | 注册模型和参数防火墙 | axis 权限、feature/data schema、assay contract |
| `brain.py` | 规则候选生成 | 诊断优先、可证伪假设和动作建议 |
| `openai_brain.py` | 结构化 LLM planner | 分离 Diagnostician 和 Action Planner |
| `evidence_router.py` | LCB 与成本惩罚 | 跨轴效用、shift risk、harm risk、replicate/stop |
| `experiment_db.py` / `action_memory.py` | 运行和跨任务记忆 | transport eligibility 防火墙；只有前瞻随机/complete-grid、同预算同协议且覆盖充分的证据可影响路由 |
| `transport_protocol.py` | 5-endpoint × 3-action 前瞻运输协议 | shared-control 去重、global/category 分层 prior、全 inconclusive 保持 neutral；破坏性剪枝另需独立 confirmation certificate |
| `runner.py` | 搜索、multi-seed、freeze、finalize | axis-isolated executor、action lifecycle |
| `frozen_executor.py` | 冻结 recipe 执行 | certification-only 权限和 subgroup audit |
| `features.py` | 注册表示和缓存 | 声明式特征编译器和 feature provenance |

## 17. 最小可交付版本

第一版不需要一次实现所有能力。最小闭环应包含：

1. `EndpointContract`；
2. 包含 scarcity、imbalance、shift、seed variance 和 saturation 的 `HardnessCard`；
3. `ProblemHypothesis`；
4. feature、model、replicate、stop 四类 `ResearchAction`；
5. axis lock；
6. Evidence Ledger；
7. matched-cost random policy 对照；
8. 5–8 种隐藏故障；
9. frozen held-out certification。

完成这九项后，系统才形成最小意义上的研究闭环：

```text
识别困难
→ 提出根因
→ 设计可证伪实验
→ 选择受控动作
→ 记录完整证据
→ 冻结候选
→ 独立认证
→ 判断 Agent 是否真正增加价值
```

## 18. 最终验收标准

建议只有同时满足以下条件，才宣称系统升级成功：

- 官方 test 在搜索和路由阶段零访问；
- 非目标研究轴零未授权修改；
- 所有 finalist 均有 freeze hash；
- 所有外部数据均有 provenance 和 assay audit；
- 所有主结果提供 raw metric、normalized gain 和不确定性；
- validation-positive/test-negative 的比例显著降低；
- Agent 在 matched-cost 下优于随机或固定策略；
- Hidden Fault Benchmark 的问题发现和根因定位显著优于规则基线；
- Agent 能正确选择 `replicate`、`stop` 或 `abstain`，而不是永远提出更复杂模型；
- 失败实验和阴性结果完整保留在 Evidence Ledger；
- 人工查看 certification 结果后，下一版本使用新的独立盲测证据。

## 19. 一句话定位

> ADMET-EvO Research Agent 是一个 hardness-aware、problem-diagnosing、protocol-safe 的闭环研究系统：它先判断 endpoint 为什么难，再选择特征、模型、数据、复核或停止动作，并通过独立盲测区分真实发现与验证集幻觉。
## 20. 2026-08-23：EC-VOI v5 的可检验算法与前瞻证据

### 20.1 当前冻结主链

```mermaid
flowchart LR
    A[EndpointContract<br/>任务语义/许可/来源] --> B[Identity + chemistry-group split]
    B --> C[State encoder<br/>规模/不平衡/结构支持/来源]
    C --> D[跨 endpoint/source<br/>action-effect posterior]
    D --> E{双门控}
    E -->|E gain >= 0.005<br/>P harm <= 0.20| F[成本约束 EC-VOI]
    E -->|否则| G[Stop / 留空预算]
    F --> H[Feature / nested imbalance<br/>Data-if-admitted / Replicate]
    H --> I[Evidence Ledger<br/>fits/wall-time/hash/failure]
    I --> J{matched ablation<br/>claim gate}
    J -->|PASS| K[一次性 confirmation / blind]
    J -->|FAIL| L[保持封存并缩小论文主张]
```

对 endpoint 状态 `s` 和研究动作 `a`，核心效用为：

`U(a|s) = E[delta|s,a] - lambda_h P(delta < -epsilon_h) - lambda_c cost(a) + lambda_i IG(a)`。

冻结常数为 practical gain `0.005`、harm margin `0.005`、最大 harm probability `0.20`。`Stop` 是显式零成本动作，策略允许不花完预算。这使论文贡献成为一个可被 no-diagnosis、shuffled-diagnosis、rules、no-harm 和 no-stop 消融直接证伪的算法，而不是不可检验的 LLM 工作流描述。

### 20.2 新 public cohort：独立于 TDC-22 的机制开发证据

公共 cohort v4 已通过治理 gate：25,642 条 endpoint-records、17 个候选端点，其中 16 个达到至少 100 个保留身份的科研规模门槛；覆盖 B3DB、MoleculeNet ClinTox、NCATS PAMPA、Tox21 四个科研来源，并同时包含分类和回归。FreeSolv 在冻结 firewall 后仅保留 68 个身份，因此只作为 small-sample stress artifact，不进入 primary gate。

结构防火墙对 14,391 个候选身份执行冻结的 exact/near-neighbor 排除：删除 6,667 个 development exact overlaps、701 个 protected-test exact overlaps 和 784 个 Morgan Tanimoto >= 0.90 的 protected-test neighbours，最终允许 6,239 个身份。测试标签未加载；来源是否准入不依赖观察到的删除数量。五折划分的 identity overlap 和 chemistry-group overlap 均为 0。

matched action screen 在任何结果出现前冻结为 415 个 endpoint-fold/action 单元，包含：

- Morgan ExtraTrees baseline；
- rich-2D ExtraTrees feature action；
- group-disjoint inner-OOF MCC threshold + class-balanced nested imbalance action；
- 固定 rich-2D LightGBM 强锚点；
- 四候选、五 inner folds、21 fits 的 expanded AutoML 强对照。

严格 policy cross-fit 会为每个 held endpoint 删除其整个来源。Full、no-diagnosis 和 shuffled-diagnosis 共用同一 action-effect registry、outer folds、数据权限和最大 21 fits/endpoint 的预算合同。TDC-19 在全部 promotion checks 通过前继续封存。

### 20.3 OpenADMET CYP：前瞻冻结而非榜单调参

OpenADMET Stage-B 在官方盲标签和榜单反馈为零时选择单一 B1 rich-2D LightGBM。冻结 recipe SHA 为 `2182b275b35c097f5739a3daa677964d2936576b6b893ceafc221016a5f03d1e`，并绑定数据、split、long-form、EndpointContract、blind firewall、执行源码及官方 Challenge Space commit `13c5057b37d1e72b3f036dd0d59718b1823f8fdd`。

唯一一次内部 certification 在协议冻结后打开，四项 gate 全部通过：macro ST-RAE `0.684642`、macro MCC `0.276628`、最差回归 endpoint ST-RAE `0.947342`、最差分类 endpoint MCC `0.115551`。随后同一 recipe 在全部官方训练标签上重拟合，生成两个 750 行 content-addressed 文件：四个 direct-inhibition pIC50 回归端点和两个 TDI 二分类端点。

这条证据目前只支持“在外部 readout 前完成不可变预测冻结”。正式 blind performance/rank 仍需上传后的官方结果；任何 readout 都不得重开本版本的模型、外部数据、阈值或 UQ 选择。

### 20.4 截至本节冻结时的证据边界

| 主张 | 当前状态 | 可写结论 |
|---|---|---|
| TDC-22 应用能力 | 冻结通过 | 18/22 Top-5，21/22 Top-10；不得因新实验重开 |
| public cohort 治理 | PASS | 多来源、许可审计、结构 firewall、group-safe folds 已完成 |
| v5 诊断条件化优势 | public cross-source gate FAIL | Full gain/regret 均劣于 no-diagnosis/shuffled；不能声称诊断优势 |
| v5 harm calibration | public cohort 0 harmful interventions | 安全 gate 通过，但未带来性能/选择优势 |
| OpenADMET 内部 certification | PASS | 冻结 B1 在独立内部 holdout 达到预注册门槛 |
| OpenADMET 官方 blind 优势 | 待外部上传/readout | 只能声称 prospective prediction freeze |
| 独立专家科研任务 | 外部 packet 未收到 | 只能声称基础设施已就绪，不能声称独立 reasoning superiority |
