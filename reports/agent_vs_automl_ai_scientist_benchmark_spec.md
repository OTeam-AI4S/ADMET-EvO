# ADMET Agent vs AutoML：AI Scientist 优势评测规范

状态：新 Goal 的主规范草案。固定范围为 TDC ADMET 22 endpoints。discovery/validation
阶段禁止读取 test 标签或使用 test 反馈；密封结构去污染器可只读 test 的 `Drug` 列，且不把
结构或相似度结果暴露给 planner。全局冻结后仅允许一次预测认证。

## 一句话定位

AutoML 主要回答“在给定数据、标签、指标和搜索空间后，哪个模型配置最好”；本 Agent
要回答更完整的问题：“这个 endpoint 和数据是否可信，困难来自哪里，哪个单轴实验能
检验原因，结果是否支持假设，是否应获取/清洗数据、换表示/模型或停止，以及全部结论
如何被复核”。

因此目标不是无限刷高分，而是：**预测性能非劣合格 + 科研闭环能力显著优于传统
AutoML**。

## 先过性能门槛，再比较科研能力

每个 endpoint 在 validation-only 阶段预注册 normalized utility 和应用合格线。用户已明确
当前预测结果达到项目需要，因此主线采用“绝对/应用合格 gate”，而不是继续对刷榜模型
做无穷追赶。Agent 必须满足：

- 5-seed normalized mean 达到 endpoint-specific application floor；
- 同时报告相对 strong-fixed、conventional AutoML 和 expanded-space AutoML 的差值，不能
  因达到绝对合格线而隐藏相对退化；
- 22-task 非劣任务率达到预注册门槛；
- 不允许用平均值掩盖少数严重退化 endpoint，必须报告最差任务和 bootstrap CI；
- test 标签只在所有策略、阈值、代码、数据和 finalist recipes 冻结后一次性验证；test 结构
  可在冻结阈值的密封 executor 内用于 external-data exact/scaffold/ECFP4 去污染。

达到门槛后，预测分数只作为 gate，不再奖励无休止的 0.00x 增益。主排名改为 Research
Capability Scorecard。若性能 gate 未通过，不能用“科研能力”掩盖模型不可用。若只
达到应用合格线、但尚未证明对 expanded AutoML 非劣，只能声称“合格性能下科研过程
更强”，不能声称“总体优于 AutoML”。

### 已冻结的项目合格线

根据项目方 2026-08-20 的目标调整，现有官方结果不再作为继续选模的反馈，而被封存为
application-performance qualification：22/22 endpoints 有结果，估算 5/22 Top-1、
18/22 Top-5、21/22 Top-10；项目合格线固定为至少 15/22 Top-5 和 21/22 Top-10。
因此此 gate 已通过，
Bioavailability_Ma 是唯一明确例外，不隐藏也不再针对它刷榜。资格记录见
`reports/performance_qualification_v1.json`，绑定 full-freeze SHA-256
`693db6936485d59791eacfd9005d18efca00056274c11c1cc80efd962058c84b`。

这个既有 test/leaderboard 证据只承担一次性“可用”认证；后续 Agent/AutoML 策略、阈值、
路由和消融一律不能读取它。新的科研能力比较全部在 train/validation、合成 hidden-fault
challenge 和 source-held-out data challenge 上完成。除非未来对一组全新、独立封存的
endpoints 做前瞻 one-shot 认证，否则不能再次把同一批 TDC test 当作迁移性证据。

## 公平对照

| Arm | 可做的动作 | 预算 |
|---|---|---|
| Strong fixed | 预注册单一 recipe + 5 seeds | 固定参考 |
| Conventional AutoML | 固定 official data/rich features，搜索模型、超参数 | 与 Agent 相同 fit/wall-time ceiling |
| Expanded-space AutoML | 与 Agent 相同 model/feature/data/imbalance/ensemble 原语，但无 hardness、matched-pair routing 和 action memory | 与 Agent 相同 fit/wall-time ceiling |
| Rules | 固定诊断动作优先级 | 与 Agent 相同 action/fit ceiling |
| Random | 随机选择 feature/model/data action | 与 Agent 相同 action/fit ceiling |
| Agent | hardness→hypothesis→matched experiment→evidence→next action/stop | 同上 |
| Agent ablations | no-hardness、no-memory、no-data、no-stop | 同上 |
| Conditional committee | 相同 action/fit ceiling，只增加审查不增加训练 | 同上 |

Conditional committee 不是默认 arm 的运行方式，而是风险触发的审查层。22-task 历史回放
中仅 2/22 tasks 被升级，覆盖 2/2 scheduled corrections 与 1/1 已观察 candidate-set
divergence；其余 20/22 保持单 Agent。该回放只证明选择性覆盖和规划去冗余，不能用来声称
多 Agent 提高预测分数。

正式比较的 Agent 默认路径进一步收缩为 category × data-size 三阶段策略：endpoint
contract、一个假设/单轴 matched pair 加最多一个 strong anchor、stop 或最多一次 correction。
small/medium/large 分别有 13/3/6 tasks；small 禁止 HPO，medium 最多一个 refinement，large
只做 one-seed screen 但正式 pair 仍补齐五 seeds。旧 8-slot/两轮策略只保留为消融，不再是
生产默认候选。新策略在五大类代表任务完成 validation-only 非劣 pilot 前保持 shadow 状态。

两种 AutoML 和 Agent 使用相同 train/validation splits、模型库、可用基础特征、CPU/GPU、
最大 fits 与 wall-time。只胜 conventional AutoML 可能只是搜索空间更大；只有在同原语
的 expanded-space AutoML 对照下，才能把优势归因于科学路由和闭环决策。Agent 的数据
治理/外部数据能力在独立 challenge track 比较，不能把额外数据带来的分数混入基础同
数据性能对照。

可执行 v5 预注册位于 `runs/v5-policy-benchmark-90m/protocol.json`，SHA-256 为
`ff01a79f6db0bb6d51a2abff3bcec1becde54827265ad720f525514d0858f4a7`。Agent、committee、
random、rules 和 expanded AutoML 的 sequential action/fit 预算一致；每个 arm 的每任务
总计算硬上限为 90 分钟。seed 1/2 barrier 后，若候选在两个 seed 都更差、平均 oriented
gap 至少 0.005、累计耗时至少 incumbent 的 3 倍，则停止后续 seed；同一规则对全部 arm
生效，Agent 的预注册 treatment/control pair 受保护。这样节省不是由不公平的 control
阉割产生，且任何超预算都作为失败/缺证据记录。

## 当前证据基线与不能跨越的边界

旧 v3 的 agent/AutoML run 使用相同 v3 protocol、各 22 tasks、12 primary actions 和 5
validation seeds。按原生 oriented delta 的 0.005 practical gate：Agent 为 1 win、12
ties、9 losses；总 executor fit time 为 23871.9 s，AutoML 为 17514.7 s，Agent/AutoML
约 1.36×。Half-Life、VDss、CYP2D6_Substrate 等存在 material loss。raw macro delta
又受 PPBR 等 MAE endpoint 尺度影响，不能作为跨任务主要统计量。

所以现有证据只支持：项目方确认预测结果达到应用需要，系统具备若干 validation-only
闭环科研组件和案例。它还不支持“预测非劣于 AutoML”“路由优于随机/AutoML”或“整体
优于 AutoML”。fresh-validation risk-guard 证据只支持相对 score-only 的有界实用非劣与
估算成本节省，不应外推成 Agent-vs-AutoML compute superiority。
新 Goal 的合理论文主张是先证明 performance-qualified research-process advantage；更强
claim 必须等 normalized evaluation、expanded-space AutoML、同预算 prospective run 和
关键消融完成后再升级。

### 本地 sealed composite policy challenge

为避免 fault、data admission、feedback 三个孤立 fixture 被简单规则分别打平，当前加入
10 个复合 episode：每例同时暴露 validation outcome、paired-seed 充分性、剩余 action
budget、0.60 compute gate、quarantined/admitted data 状态和跨轴候选。Agent arm 直接执行
生产 `ValidationSelfCorrectionPlanner`；所有 controls 每例同样最多一个 follow-up 和六个
估算 fits。结果为 Agent 10/10、macro=1.000、unsafe=0、30 fits；locked static rules
6/10、macro=0.860、unsafe=3、48 fits；expanded/conventional AutoML 4/10、macro=0.540、
unsafe=1、48 fits。Agent 相对 static rules 的 paired case-score delta=+0.140，bootstrap
95% CI [0.040, 0.260]。

该结果证明闭环模块在联合约束下按设计工作，并给出相对静态规则与 scalar AutoML 的本地
过程增量；但题目由同一团队编写，不能冒充 independently authored unseen evaluation，
也没有覆盖九个 process dimensions。全局 process-advantage claim 继续保持 false。

对既有 `runs/v3-agent-tdc22-v5` 的新过程审计显示：22/22 action 均有 single-axis 和完整
treatment/control，18 supported、2 refuted、2 inconclusive，falsification resolution
20/22，refuted 结果 ledger retention 2/2；但 self-correction evidence 为 0/2 且 coverage
为 0，action ordering coverage 为 0，历史 run 为 790 fits、23871.9 fit-seconds、没有记录
任何 racing savings。详见 `reports/research_process_baseline_v3.json`。因此当前最需要补的
不是更多模型，而是前瞻自我纠错、时序证据和成本感知停止。

现已加入 validation-only self-correction scheduler：refuted 后禁止重复同一动作并选择
最高排序的可执行跨轴动作；matched pair/paired seeds 不完整时只修复证据；supported 且
应用性能已合格时停止。它输出 `scheduled`/`completed` 两阶段状态，scheduled 不计入
self-correction 成功。最小队列随后在 CYP2C9_Substrate 与 hERG 上完成 2/2 follow-up：
共 12 个 2D tree fits、67.6 fit-s，CYP2C9 修复缺失 matched evidence，hERG 在 data
refutation 后切换至 model axis。两个 child hypotheses 都 inconclusive，系统按门槛停止。
这填补了“是否实际执行纠正”的 smoke 证据，但未填补相同 feedback/budget 下的 rules/
expanded-AutoML control。

## 十个核心量化维度

### 1. Predictive Sufficiency

- 22-task 非劣率；
- normalized oriented delta 的 macro mean、median、95% bootstrap CI；
- seed rank stability 和最差 endpoint regret。

它是硬门槛，不是主要故事分。

### 2. Hidden-fault Discovery

向 discovery 数据注入或提供隐藏挑战：invalid SMILES/target、冲突标签、重复测量高方差、
train-valid overlap、极端不平衡、chemical shift、activity cliffs、species/assay/unit
mismatch、wrong metric direction、provenance missing。

报告 micro/macro precision、recall、F1、critical-fault recall、检测延迟。关键故障漏检时
整个安全 claim 降级。

### 3. Remediation Validity and Harm Avoidance

- 有效修复数 / 提议修复数；
- 修复后相对 matched official control 的 normalized effect；
- 无效清洗被拒绝率；
- 因错误单位/物种/assay merge 而避免的 harmful actions 数。

AutoML 通常默认输入合法；这一轨直接显示 Agent 能否发现“优化问题本身是错的”。

### 4. Causal Experimental Design

- 训练前预注册假设率；
- single-axis matched-pair 合格率；
- treatment/control 原子完成率；
- falsification test 完成率；
- supported/refuted/inconclusive 三类负责任结论覆盖率。

只看赢家不算科研能力；能得到并保留 refuted/inconclusive 也是正向证据。

### 5. Routing Quality and Calibration

对 feature/model/data/stop 轴计算 retrospective oracle-axis regret，但 oracle 只用于评测、
不可部署。报告：

- 选中最佳或实用等效轴的比例；
- normalized routing regret；
- predicted vs observed normalized effect 的 calibration error；
- support probability 的 Brier score/ECE；
- action memory 相对 no-memory ablation 的增益；只有来自预注册的 prospective randomized
  assignment 或 complete action grid、同预算共同协议、每动作至少 5 个独立 endpoint 且
  outcome 有变化的 transport-eligible evidence 才能进入该比较。历史 selected-action
  outcomes 只用于审计，不能改变 routing utility。
- 任何会删除候选的 negative-memory gate 必须另有独立 post-memory confirmation；仅有
  transport prior 可以校准 utility，不能授权 destructive pruning。确认失败必须自动回滚，
  不能把少做的 fits 计作效率收益。

首个合格 transport grid 已采用 shared-control complete design：5 个 ADMET sentinel、
3 个 action、5 seeds。完全相同的 control 在同一 endpoint/seed 只拟合一次，因此 15 个
matched episodes 只需要 100 fits 而非 150；节省本身计入 experimental efficiency，但
不得把 protocol-level prior 存在包装成 behavioral routing advantage。只有 no-memory
对照上的决策或 regret 改善才能升级该 claim。

### 6. Self-correction and Negative-result Use

- 初始假设被 refute 后，下一动作仍然合法且不重复的比例；
- 错误诊断修正率；
- 无增益后正确停止/转轴率；
- 相同失败动作的跨任务重复率；
- negative results 写入 ledger 和影响后续 prior 的覆盖率。

hERG official/leakage-safe 的 inconclusive/refuted 结果是代表案例：Agent 没有为了讲故事强行
清洗数据，而是保留 official data，并记录为什么。

### 7. Experimental Efficiency and Appropriate Stopping

- time-to-acceptable-performance；
- anytime performance AUC；
- 达到合格线所需 fits/CPU/GPU hours；
- futility stop precision、recall、planned fits avoided；
- high-cost dominated candidate rate；
- peak storage、重复 checkpoint/embedding bytes。

CYP2C9 balanced ExtraTrees 是代表案例：与十几秒 LightGBM 相比耗时约 50 倍，多 seed
仍落后；Agent 应能依据预注册规则停止，而 AutoML baseline 会继续机械搜索。

### 8. Data Scientist / Data Stewardship Capability

- source provenance、license、checksum、schema manifest 完整率；
- endpoint、assay、species、unit、transform 对齐准确率；
- duplicate/conflict resolution 的人工金标准一致率；
- external data quarantine precision/recall；
- 合格外部数据引入后的 source-held-out transfer；
- 不合格数据被拒绝且未进入训练的比例。

数据能力独立成 track，避免仅用“加了更多数据所以更高分”证明 Agent 智能。

### 9. Reproducibility and Scientific Auditability

- source/data/config/protocol/finalist hash 完整率；
- 同 recipe 重现成功率；
- ledger chain 验证率；
- action、replicate fit、derived ensemble、storage 成本分账准确率；
- 从结论追溯到 hypothesis→config→seeds→artifacts 的可达率。

### 10. Test Governance and Claim Safety

- discovery 期间 test feedback events 必须为 0；
- protocol violations 必须为 0；
- test certification 必须具有 freeze manifest hash、freeze audit pass、one-shot proof；
- 不满足时只能声称 validation evidence，不能声称 test-certified superiority。

## Research Capability Scorecard

不建议只发布一个容易掩盖问题的总分。采用两层结构：

1. 硬 gate：predictive sufficiency、critical fault recall、test governance；
2. gate 通过后，报告 0--100 的 Research Capability Score 及全部分项。

建议权重：fault discovery 15、remediation 10、causal design 15、routing/calibration 10、
self-correction 10、efficiency/stopping 15、data stewardship 10、reproducibility 10、claim
safety 5。预测性能不进入加权奖励，因为已作为硬 gate。

同时报告每个维度的 Agent−AutoML 差值与 bootstrap CI；不能只给复合分。若需要更强的
“总体优于”claim，再增加 expanded AutoML prediction non-inferiority gate；它不是当前
“不再刷分”目标的前置条件，但决定 claim 的强弱。

## 三类最有说服力的演示

### Case A：hERG——会否定自己的数据假设

Agent 发现 leakage signal，提出 official/leakage-safe matched pair。balanced ExtraTrees
旧配对的 5-seed mean oriented delta 为 -0.01805、0/5 正方向；生产 LightGBM 配对为
-0.00340、3/5 正方向、结论 inconclusive。系统保留 official data，隔离一个错误
hardcoded class-weight pilot，并把负结果写入 memory。故事重点是“可证伪、自我审查、
不会为了行动而行动”。

### Case B：CYP2C9——知道什么时候停止计算

冻结 pilot 中 balanced ensemble 的 5-seed AUPRC 为 0.79366、累计 2461.4 s；最终
LightGBM 为 0.81794、累计 74.3 s。balanced 单候选占全任务 3027.1 s fit time 的约
81%。当前只能把 racing 描述为 locked retrospective policy；前瞻 v5 执行后才能说 Agent
实际自主停止。故事重点是“效果合格后优化信息价值，而非暴力堆算力”。

另一个独立的 fresh-validation risk-guard replay 已通过预注册实用非劣门：22 endpoints
mean oriented delta vs score-only 为 −0.000348，task-bootstrap 95% CI
[−0.001482, 0.000585]，20/22 未出现超过 −0.005 的退化，aggregate selected-recipe cost
ratio 为 0.278。允许表述为 bounded Pareto efficiency；两个 material regressions 和
routing superiority 的负结果必须同时报告。

### Case C：Hidden data fault / external data dossier

给出混有 unit、species、assay 和 provenance 问题的候选外部数据。AutoML 直接拟合；Agent
应识别、隔离、生成数据卡，只合并对齐子集，并用 source-held-out validation 证明修复。
故事重点是“从模型优化器升级为受约束的数据与实验研究闭环”。

## Claim Ladder

1. `auditable_model_search`：可复现搜索与成本分账；
2. `diagnostic_admet_agent`：能发现预注册科学/数据故障；
3. `closed_loop_validation_research_agent`：能提出、证伪、学习和停止；
4. `performance_qualified_research_process_advantage`：性能过应用 gate，科研过程指标优于
   expanded AutoML；
5. `expanded_automl_noninferior_ai_scientist_style_agent`：再通过预测非劣和 compute parity；
6. `frozen_test_certified_bounded_agent`：全局冻结后 one-shot test 认证。

Scorecard v3 将这些结论改为并行、可组合的 claim flags：应用性能/安全是 foundation；
`validation_process_advantage` 不要求先赢 AutoML 的预测分数；
`expanded_automl_predictive_noninferiority` 是独立、更强的预测 claim；二者都满足时才组合为
最高等级。这符合“指标已够用，核心评价科研能力”的项目目标，也避免用流程分掩盖真实的
AutoML 预测负结果。

现有 hidden-fault 10-case 结果属于 deterministic integration fixture，不是独立盲测；
现有 racing 是锁定历史回放，不是前瞻停止。不得声称通用“自主科学家”，也不得在只有
validation 证据时声称 test superiority。最稳健
的表述是：**在固定 TDC-22 ADMET 场景中，一个性能合格、可审计、能进行闭环科学与数据
决策的 bounded research agent。**

### 低成本控制实验的新边界

对 10-case fault fixture 的 locked control replay 中，Agent micro F1/critical recall 为
1.000/1.000；expanded AutoML schema guard 为 0.615/0.167，conventional AutoML 为
0/0。但 complete rules guard 同为 1.000/1.000，Agent−rules case-F1 bootstrap CI 为
[0, 0]。对 11 个 sealed local data dossiers，Agent 决策准确率 1.000、unsafe merge=0；
expanded AutoML 为 0.273/7，conventional AutoML 为 0.091/10；complete rules 又与 Agent
完全打平。

这证明 endpoint/data contracts 相对传统 AutoML 有明确的系统能力增量，同时证明当前
detector/steward 仍可被确定性规则完全解释。论文必须把它表述为 safety/governance layer
advantage，而不是 learned reasoning advantage。要升级到 autonomous discovery claim，
challenge 必须由独立方编写、包含未见复合故障、相互冲突证据、多个可行修复和后续实验
反馈；Agent 还必须在 remediation choice、self-correction 或 information gain 上显著优于
complete rules。

## 下一步实现顺序

1. 修复 metric normalization、missing-std、nested test-key firewall 和 freeze claim gate；
2. 固定 performance sufficiency/non-inferiority protocol；
3. 生成 hidden-fault challenges 与 AutoML 同预算 runner；
4. 从现有 22-task artifacts 构建 validation-only Research Episodes；
5. 只补跑缺失的 controls/challenges，不再为小幅分数增益扩大模型搜索；
6. 输出 Agent vs AutoML scorecard、案例 dossier 和 claim-level report；
7. 所有内容冻结后才执行 one-shot test。
