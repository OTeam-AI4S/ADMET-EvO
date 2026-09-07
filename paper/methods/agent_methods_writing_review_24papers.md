# Agent 顶会论文 Methods 写作复盘：24 篇文献及其对 ADMET-EvO 的启示

> 目的：学习这些工作的 **Methods 组织方式和论证顺序**，而不是与其中任何一篇工作做性能或贡献比较。  
> 范围：24 篇发表于 ICLR、ICML、NeurIPS、ACL、COLM、EMNLP Findings 的 Agent 论文，以及一篇与分子性质预测最直接相关的近期预印本。逐篇检查了其 Approach/Methods、agent interface、evaluation protocol 和实验设计。检索截止日期：2026-08-27。

## 1. 文献矩阵

| # | Paper | Venue | Methods 的主要组织方式 | 对 ADMET-EvO 写作最有用的经验 |
|---:|---|---|---|---|
| 1 | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | ICLR 2023 | 先形式化 agent–environment interaction，再把 reasoning trace 扩展进 action space，随后按任务实例化 | 在组件介绍前先定义 \(S_t\)、\(a_t\)、executor 与更新规则；用一个短公式统领全篇 |
| 2 | [Reflexion: Language Agents with Verbal Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) | NeurIPS 2023 | Actor、Evaluator、Self-Reflection、Memory 四个模块逐一给出职责，随后提供算法框 | 将 planner、executor、evaluator、memory 写成不同权限主体；补一个闭环伪代码 |
| 3 | [Toolformer: Language Models Can Teach Themselves to Use Tools](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html) | NeurIPS 2023 | 定义 API-call 表示，再按 sampling、execution、filtering 三阶段展开 | 每个阶段都写成 input → operation → output → acceptance rule，避免泛泛描述“调用工具” |
| 4 | [AgentBench: Evaluating LLMs as Agents](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e9df36b21ff4ee211a8b71ee8b7e9f57-Abstract-Conference.html) | ICLR 2024 | 先定义统一交互框架，再说明八个环境及通用评测设置 | ADMET 端点可共享统一 ResearchState/ActionSpec/EvidencePacket，但 endpoint contract 必须单独实例化 |
| 5 | [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://proceedings.iclr.cc/paper_files/paper/2024/hash/28e50ee5b72e90b50e7196fde8ea260e-Abstract-Conference.html) | ICLR 2024 | API collection、instruction generation、solution-path annotation、training 和 evaluation 顺序清晰 | 外部数据检索应写成可审计的阶段管线，而不是一段“Agent 自动搜索数据”的叙述 |
| 6 | [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4410c0711e9154a7a2d26f9b3816d1ef-Abstract-Conference.html) | ICLR 2024 | 用 \(E=\langle S,A,O,T\rangle\) 定义环境；再写站点、任务、observation/action 和 functional evaluator | 把数据分区、可用工具和 evaluator 所有权写成环境定义；以实际产物/状态变化而非语言判断作为证据 |
| 7 | [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://proceedings.iclr.cc/paper_files/paper/2024/hash/edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html) | ICLR 2024 | task object → dataset construction/filtering → executable evaluation → dataset properties | 先定义 endpoint-level analysis unit 和 reporting unit，再写数据、过滤和测试 |
| 8 | [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6507b115562bb0a305f1958ccc87355a-Abstract-Conference.html) | ICLR 2024 | 以 SOP 分阶段，明确角色间交付物与通信接口 | 借鉴 typed handoff，不需要把 ADMET-EvO 包装成更多角色；突出对象契约比“多 Agent 数量”更重要 |
| 9 | [MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation](https://proceedings.mlr.press/v235/huang24y.html) | ICML 2024 | 定义任务、工具动作、ReAct 实验循环和成功标准 | 把读写文件、执行训练、读取结果等权限列明；实验完成与否必须有机器可验证状态 |
| 10 | [Executable Code Actions Elicit Better LLM Agents](https://proceedings.mlr.press/v235/wang24h.html) | ICML 2024 | 统一 code-action space，并通过执行反馈形成多轮修订 | ActionSpec 应是可执行 schema；自由文本 hypothesis 只有编译为注册动作后才可运行 |
| 11 | [ChatDev: Communicative Agents for Software Development](https://aclanthology.org/2024.acl-long.810/) | ACL 2024 | 按设计、编码、测试等时序阶段组织，并定义阶段内沟通机制 | 主文按科学因果顺序组织，不按代码包名组织；每阶段结束必须留下标准化交付物 |
| 12 | [AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents](https://aclanthology.org/2024.acl-long.850/) | ACL 2024 | environment → benchmark construction → setup/evaluation → solution programs；使用 state-based unit tests | certification 不只检查分数，还要检查数据污染、越权修改和副作用 |
| 13 | [VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks](https://aclanthology.org/2024.acl-long.50/) | ACL 2024 | 在既有环境契约上扩展 observation modality，并保持 execution-based evaluation | 新的 evidence source 应作为受控接口扩展，而不是改变原有评价协议 |
| 14 | [SWE-agent: Agent–Computer Interfaces Enable Automated Software Engineering](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html) | NeurIPS 2024 | 先给 interface design principles，再按 search、viewer/editor、context management 展开；实验设置单列预算和成本 | 把 interface、guardrail、反馈粒度和 compute budget 视为 Methods 核心，而不是实现细节 |
| 15 | [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5d413e48f84dc61244b6be550f1cd8f5-Abstract-Datasets_and_Benchmarks_Track.html) | NeurIPS 2024 D&B | initial-state setup、真实执行环境、execution-based evaluator 与可复现实例 | 每个 endpoint episode 都应能从固定初态重放；认证 executor 应从 frozen recipe 重建结果 |
| 16 | [AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents](https://proceedings.neurips.cc/paper_files/paper/2024/hash/877b40688e330a0e2a3fc24084208dfa-Abstract-Datasets_and_Benchmarks_Track.html) | NeurIPS 2024 D&B | 将最终 success 与 progress rate、subgoal progress 和 failure analysis 分开 | 将预测性能、闭环过程质量、协议安全性分成三套指标，避免用一个 leaderboard score 证明 autonomy |
| 17 | [WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0b82662b6c32e887bb252a74d8cb2d5e-Abstract-Datasets_and_Benchmarks_Track.html) | NeurIPS 2024 D&B | 任务由可组合子目标生成，并提供 ground-truth observation/action traces | 用 endpoint trace 和 action lineage 评估 Agent 是否完成了完整逻辑链，而非只看最后模型 |
| 18 | [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) | COLM 2024 | 先定义 conversable/customizable agent abstraction，再给出 conversation patterns 与应用实例 | 对 ADMET-EvO 应强调模块可替换性和消息 schema；避免把聊天轮数写成自主性证据 |
| 19 | [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7e3767db483c942b883eb4f8cfb74e31-Abstract-Conference.html) | ICLR 2025 | benchmark construction、human calibration、agent scaffolds、resource scaling 和 contamination analysis | 同时报告资源预算、human/strong baseline、backend variation 和 contamination risk |
| 20 | [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f12b4df26344f3be803c06b555252efe-Abstract-Conference.html) | ICLR 2025 | 从真实论文抽取科学任务；分别检查程序、执行结果、成本和污染 | 科学性必须落到可执行产物；将自然语言 hypothesis、运行成功和科学结果分开计分 |
| 21 | [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://openreview.net/pdf?id=roNSXZpUDN) | ICLR 2025 | 用 POMDP 定义工具–Agent–用户交互，并加入 domain policy 与重复运行可靠性 | EndpointContract 相当于不可修改的 domain policy；除均值外报告重复运行的一致性与 worst-case reliability |
| 22 | [Dolphin: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback](https://aclanthology.org/2025.acl-long.1056/) | ACL 2025 | idea generation → implementation/debugging → result analysis → feedback 的闭环 | 闭环应写明反馈究竟来自什么执行结果；ADMET-EvO 应把反馈限定为 EvidencePacket，而非任意总结 |
| 23 | [Agent Laboratory: Using LLM Agents as Research Assistants](https://aclanthology.org/2025.findings-emnlp.320/) | Findings of EMNLP 2025 | literature review、experimentation、report writing 三阶段，兼有 autonomous 与 co-pilot mode | 区分 autonomy level 和人工检查点；主张必须与实际无需人工完成的阶段严格对应 |
| 24 | [Closed-loop Auto Research for Molecular Property Prediction](https://arxiv.org/abs/2606.22731) | arXiv 2026 | reliability question → benchmark/split/metric → held-out protocol → submitted-trial loop → budget/baseline → leakage-safe augmentation | 借鉴其“先固定比较与报告单位、再介绍闭环”的顺序；不沿用其具体系统或比较叙事 |

## 2. 从 24 篇论文中提炼出的 Methods 写作规则

### 规则 1：第一节定义“研究对象”，不要先罗列模块

高质量 Methods 首先回答一次 episode 是什么、输入是什么、最终输出是什么。对 ADMET-EvO，基本分析单位应是一个由 EndpointContract 启动、以 FrozenRecipe 和 ClaimRecord 结束的 **endpoint research episode**。这样读者不会把系统误读为一个常规 AutoML pipeline。

### 规则 2：用最少量形式化语言固定交互闭环

主文只保留一个统领公式：

\[
S_t \xrightarrow{\pi} a_t
\xrightarrow{\mathcal E} e_t
\xrightarrow{\mathcal U} S_{t+1}.
\]

紧接着逐一解释 state、policy、registered action、matched executor、EvidencePacket 和 evidence-gated update。其他阈值公式留在对应小节或补充材料，避免数学符号淹没逻辑。

### 规则 3：系统图画“控制权和信息流”，不画软件包清单

图 1 应明确四条边界：

1. task semantics 与 metric 在 planning 前冻结；
2. planner 只能提出注册动作，不能直接写入 evidence；
3. protected labels 不进入 discovery loop；
4. certification 只能读取 FrozenRecipe，不能回调 planner。

这四条边界比展示十几个内部类更能说明 Agent 的方法学创新。

### 规则 4：每个模块采用同一段落模板

每节按照以下顺序书写：**Input → Operation → Output → Invariant → Failure behavior**。例如，external-data module 的 invariant 是 score-blind admission，failure behavior 是 quarantine/abstention，而不是静默回退。

### 规则 5：proposal、execution 和 evidence 必须分开

语言模型生成的计划不是证据；代码成功运行也不等于科学支持。Methods 应清楚地区分：

- Hypothesis：为什么某干预可能有效；
- ActionSpec：可执行且唯一变化轴的实验契约；
- RunArtifact：完成运行的原始产物；
- EvidencePacket：经 evaluator 标准化后的 effect、uncertainty、cost、provenance 和 verdict。

### 规则 6：把权限与禁止事项写入正文

Agent 论文最容易被审稿人质疑的是信息泄漏和隐式自由度。主文应列权限矩阵，明确 planner、data curator、executor 和 certifier 能看什么、能改什么。尤其要用明确的否定句：could not access、could not modify、was not reused。

### 规则 7：预算、失败、重试和停止不是实现细节

必须说明每个 episode 的 action/compute/time budget、seed 数、失败状态、允许的 bounded correction 次数、timeout、停止与 abstention 条件。否则“自主闭环”可能只是无限重试后挑最好结果。

### 规则 8：从最终性能扩展到过程和安全评测

建议把 Evaluation 分为三层：

1. **Predictive utility**：原生 metric、oriented effect、uncertainty、calibration、OOD/chemical-space transfer；
2. **Research-process quality**：hypothesis validity、matched-pair completeness、evidence update、negative-result retention、correction completion、stopping；
3. **Protocol safety**：sealed-label access、contract mutation、unsafe merge、freeze integrity、certification reconstruction。

### 规则 9：控制实验必须对应能力主张

若主张“诊断驱动路由”，就比较 matched-budget routing controls；若主张“证据门控”，就消融 EvidencePacket/gate；若主张“自我纠错”，就统计从 refutation 到已执行 correction 的完整率。不要用全系统性能统一代替所有机制的证据。

### 规则 10：先定义 selection/reporting unit，再报告最大值

应说明是按 endpoint、seed、research episode 还是 action pair 聚合；明确 candidate selection policy 和 one-standard-error rule；报告 paired effect 分布与不确定性，避免只展示 across-trial maximum。

### 规则 11：主文保留决定 claim 的规则，枚举细节放补充材料

主文保留 EndpointContract 字段类别、动作空间、evidence verdict、权限边界、停止条件、freeze/certification 与核心评价设计。fingerprint 全清单、超参数空间、prompts、source dossier、所有 admission threshold 和 artifact schema 进入 Supplementary Methods。

### 规则 12：用操作性语言说明“fancy”

避免 intelligent、understands、reasons scientifically 等不可检验形容词。替换为可审计能力：

- instantiated an executable endpoint contract；
- selected among registered interventions；
- executed an axis-isolated comparison；
- updated state only from completed evidence；
- retained negative outcomes；
- froze and independently reconstructed the finalist。

## 3. 对当前 ADMET-EvO Methods 草稿的结构诊断

当前草稿的技术细节充分，但主线存在四个排版问题：

1. **external-data 部分出现过早且过长**，打断了 research state → action → evidence 的因果链；
2. **HardnessCard 位于数据扩展之后**，但它实际上是 planner 采取干预前的状态输入；
3. **核心对象散落在多个段落**，读者难以快速区分 EndpointContract、ResearchState、ActionSpec、EvidencePacket 和 FrozenRecipe；
4. **Agent 行为与常规模型细节篇幅倒置**，fingerprint/learner/data threshold 的枚举压过了系统为何构成 research agent 的说明。

## 4. 推荐的新主文顺序

1. Study objective, unit of analysis, and system boundary
2. Agent–environment interface and typed research objects
3. Protocol-safe research loop
4. Endpoint formalization and discovery-only diagnosis
5. Registered interventions and falsifiable hypothesis contract
6. Matched execution, evidence gating, correction, and stopping
7. External-evidence acquisition and governance
8. Predictive portfolio and candidate selection
9. Freeze lock, independent certification, and claim states
10. Evaluation design, statistical analysis, and reproducibility

这个顺序使 Methods 只讲一条连续故事：**定义问题 → 形成状态 → 提出可证伪动作 → 执行并生成证据 → 用证据更新 → 冻结产物 → 独立认证结论**。

## 5. Main text 与 Supplement 的篇幅分配

| 内容 | Main text | Supplement |
|---|---|---|
| EndpointContract、partition ownership、typed objects | 定义和关键字段 | 完整 JSON schema 与实例 |
| HardnessCard | 六维含义和如何影响 planning | 精确公式、权重、阈值、敏感性分析 |
| Action space | 五类动作和 ActionSpec contract | 完整 action registry 与 prompts |
| Evidence gate | effect 方向、verdict、更新规则 | endpoint-specific thresholds 与全部 edge cases |
| External data | 六阶段、score-blind admission、firewall、confirmation | 来源 dossier、所有阈值、查询式、清洗表 |
| Predictive portfolio | 表征/模型类别和 selection principle | 全部 fingerprints、超参数和依赖版本 |
| Freeze/certification | 权限隔离和不可变性 | manifest、hash chain、重建脚本说明 |
| Evaluation | 三层问题、controls、metrics | endpoint tables、完整统计方案和额外分析 |

## 6. 最终写作判断

ADMET-EvO 最有说服力的定位不是“会调用很多模型和数据库的 Agent”，而是：

> **an evidence-gated research system in which autonomy is expressed as constrained experimental choice, and scientific validity is protected by evaluator-owned state transitions and planner-independent certification.**

中文可表述为：

> **一个证据门控的自主研究系统：其自主性体现在受约束的实验选择上，而其科学有效性由 evaluator 掌控的状态更新与独立于 planner 的认证流程保障。**

这句话能够自然统领后续所有技术设计，也避免把贡献建立在“LLM 更聪明”这一难以验证的主张上。
