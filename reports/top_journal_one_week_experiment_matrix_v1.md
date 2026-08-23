# ADMET-EvO 顶刊 / AI4Science 一周实验矩阵 v1

日期：2026-08-21
证据边界：所有现有结果均为 development-only；OpenADMET blind labels 与 TDC 19-task confirmation cohort 未访问。

## 1. 当前结论与缺口

EC-VOI v3 的七臂、三任务、协议冻结开发门控已经完成。安全、哈希、identity-safe 数据审计、完整覆盖和 AutoML 非劣检查通过，但 promotion gate 失败：完整 Agent 在 BBB_Martins、CYP2C9_Veith、CYP2D6_Veith 上均选择 shared strong anchor，对 shared anchor 的严格正增益为 0/3；相对 random 为 0 胜、2 平、1 负。因而 19 个确认任务保持封存，不能声称策略优越。

机制审计进一步发现：三个任务的诊断首选均为 `feature-domain-shift`，但首次探索动作均为 `model-imbalance-aware`，诊断—动作一致率为 0/3。现有观测动作效果上，打乱诊断标签不会改变离线策略结果。因此，当前最关键缺口不是 predictor 不够大，而是“诊断没有可识别地驱动正确动作”。

OpenADMET 外部 propensity 闭环已经提供一个正向但有限的结果：六个 CYP endpoint 的冻结 B1 交叉验证中，四个回归任务的 ST-RAE 全部改善（+0.001400 至 +0.007864），两个 TDI 分类任务的 MCC 均退化。因此已冻结的 task-type router 只在回归 endpoint 采用外部 propensity，分类和未知类型回退 B1 control。该结果是开发期候选证据，不是 blind superiority 证据。

## 2. 论文的算法核心

论文不把 LLM 本身包装成预测模型，而定义一个可检验的研究策略：

`Endpoint state encoder -> action-effect posterior -> harm/cost-aware EC-VOI -> execute/replicate/stop -> evidence update`

状态包含 EndpointContract、HardnessCard、数据来源与相似性分布、baseline 饱和度、候选间分歧和已有 action lineage。对每个授权动作，模型预测 endpoint-local 增益分布、伤害概率、成本和信息价值：

`U(a|s) = E[delta|s,a] - lambda_h * P(delta < -epsilon|s,a) - lambda_c * cost(a) + lambda_i * information_gain(a)`

只有当诊断条件化在未见 endpoint 上产生可重复的 action-selection advantage，这个结构才构成算法贡献。否则应把论文降格为可审计平台/benchmark 论文，不能声称“科学诊断提升预测”。

## 3. 必补实验（按优先级）

| 优先级 | 实验 | 设计 | 首要指标 / 通过条件 | 支撑或否定的主张 |
|---|---|---|---|---|
| P0 | Matched-state × action 因果矩阵 | 在已开放开发 endpoint 和额外非确认任务上冻结多个自然/受控状态；同一 split、seed、模型与预算下，分别执行 Feature、Model、Data、Replicate、Stop/none 动作 | action×state 交互；正确动作 regret；harm rate；打乱 state 后优势必须显著下降 | 诊断是否真的选择了“对该状态正确”的动作 |
| P0 | Diagnosis-conditioned policy 消融 | Full v4、无诊断、打乱诊断、规则诊断、无 memory、无 harm model、无 cost/stop；共享相同动作效果数据库与预算 | 未见 endpoint 上 Full 的 regret 更低、harm 更少；shuffled 必须退化；预注册 practical margin | 算法组件是否可识别，而非 prior 或搜索次数作用 |
| P0 | Leave-one-endpoint/source-out 泛化 | policy 只在其他 endpoint/source 的 action effects 上拟合；被留出 endpoint 的标签不进入 policy；重复 endpoint-cluster bootstrap | LOEO normalized gain、selection regret、bootstrap CI、跨来源 transfer | 策略是否能迁移到未见科学问题 |
| P0 | 同预算强对照 | Full、shared anchor、fixed、random、rules、普通 AutoML、expanded AutoML、planner-without-diagnosis；匹配 actions、fits、wall-clock 上限和数据权限 | endpoint-level paired delta、win/tie/loss、gain/trial、gain/GPU-hour、gain/dollar | Agent 是否优于搜索，而非仅更昂贵 |
| P1 | 五轴 leave-one-axis-out | Full 与 -Feature、-Model、-Data、-Replicate、-Stop；同时给 matched-max-budget 与 matched-realized-cost | 性能、sign reversal、selection error、harm、cost、coverage-risk | 五类研究动作的独立作用与边界 |
| P1 | OOD / calibration / abstention | identity/scaffold/相似性距离分层，能用时间或来源时优先 source/time holdout；比较 ensemble disagreement、distance、combined UQ、conformal | AUROC-error detection、AURC、ECE/Brier、coverage-risk、shift 分层校准 | 系统在新化学空间中是否知道“不知道” |
| P1 | 外部数据闭环 | 高质量外部数据作为 supervised treatment；一般数据只做预训练/propensity；严格 assay contract、许可、冲突、identity firewall；source-held-out | 对 control 的 held-out delta、负迁移率、source-held-out CI、污染 0 | Data Agent 是否增加证据而非制造泄漏 |
| P1 | 真正前瞻盲测 | OpenADMET CYP 路由和模型在 blind labels 前冻结；一次性提交；不按榜单反馈改同一版本 | blind primary metric、排名、CV→blind transfer、sign reversal；完整 timestamp/hash | 最强的现实世界外部有效性 |
| P2 | Agent 科研能力盲评 | 由未参与开发的 ADMET 专家编写任务；冻结 rubric；专家对匿名 action plan 做成对排序 | correctness、falsifiability、safety、novelty、cost awareness、inter-rater agreement | Agent 的科研价值不仅是榜单分数 |
| P2 | 安全、复现与故障注入 | test-loader、split/metric 篡改、污染外部数据、标签冲突、超时、坏 SMILES、提示扰动；随机 recipe replay | 越权拦截率、false rejection、artifact 完整率、prediction correlation、恢复一致性 | protocol-safe 与 reproducible 的系统贡献 |

## 4. 一周内的最小可发表版本

### Day 1：冻结问题与数据

- 冻结 v4 state schema、动作定义、效用函数、practical margins 和 promotion gate。
- 建立 matched-state 数据集；严格排除 TDC 19 个 confirmation endpoint 的结果反馈。
- 冻结 OpenADMET task-type router、外部来源 manifest 和 blind firewall。

### Day 2–3：生成动作因果证据

- 优先运行廉价、同模型、同 seed 的 matched-state × action grid。
- 每个动作对保留完整失败记录；不按结果临时删状态或换指标。
- 拟合 action-effect posterior，并做 endpoint/source cluster bootstrap。

### Day 4：核心消融与 OOD

- 跑 Full / no diagnosis / shuffled / rules / no harm / no stop。
- 运行 similarity/scaffold/source shift 分层、calibration 与 coverage-risk。
- 若 shuffled 不退化或 Full 没有 regret/harm 优势，停止进入 19-task confirmation。

### Day 5：系统级五轴与成本

- 在代表性任务上跑五轴 LOAO，不强行在全部任务做昂贵全因子实验。
- 汇总 GPU-hour、wall-clock、fit 数、失败数、LLM token/API cost 和 stop 节省。

### Day 6：前瞻认证

- 只有 v4 开发门控通过，才一次性解封预注册 confirmation；否则保留为诚实负结果。
- 对 OpenADMET 使用冻结 route 一次性盲提交，不允许榜单驱动同版本重选。

### Day 7：复现、统计与论文图

- recipe replay、协议攻击、hash/manifest 审计。
- 生成 endpoint-level paired plots、gain-cost curves、action×state heatmap、OOD coverage-risk 和完整失败图。
- claim gate 逐条决定正文可写的主张，未通过的主张删除或改为 limitation。

## 5. Go / No-go 门控

进入 TDC 19-task confirmation 之前，至少满足：

1. Full v4 在未见 endpoint/source 上相对 no-diagnosis 与 shuffled diagnosis 的 selection regret 有预注册幅度的改善；
2. shuffled diagnosis 的表现低于 Full，证明 state 信息可识别；
3. harmful intervention rate 不高于 shared anchor/rules，且至少一个开发任务严格优于 shared anchor；
4. 所有数据为 identity-safe，test/leaderboard access 为 0；
5. 同预算，不以更多 fits 或隐藏 wall-clock 获益。

若未通过，论文应转为“可审计自主 ADMET 研究 benchmark + 负结果 + 外部数据路由”方向；不再消耗封存任务，也不声称诊断策略优越。

## 6. 最可能形成顶刊故事的结果组合

最强组合不是“22 个榜都涨一点”，而是：

1. 一个明确的新算法：风险/成本约束、证据条件化的 action-effect policy；
2. matched-state 与 shuffled-diagnosis 证明诊断具有因果/可识别价值；
3. LOEO/source-out 证明在未见 endpoint 上仍能选对动作；
4. 同预算强对照证明效率和可靠性优于 AutoML/random/rules；
5. OpenADMET 一次性盲测和 source-held-out 外部数据闭环提供前瞻真实性；
6. OOD abstention、安全审计、完整失败报告限定系统边界。

仅有 TDC 排名、LLM 工作流图或新 predictor 架构，不足以支持 AI4Science 顶刊主张。

## 7. 2026-08-23 实施更新

- v4 matched-state v1 已完成 135/135 fits，0 failure；所有状态构造均为 development-only、identity-safe、label-blind with respect to validation-state construction。
- 逐状态绝对动作 gate 失败：Full/no-diagnosis/shuffled 均在 15/15 状态选择 feature-rich，诊断不可识别。
- 在 confirmation seeds 输出前冻结的 50% cohort compute-budget 分配中，Full total oriented gain 为 0.068074，高于 no-diagnosis 0.058620 和 shuffled 0.060002；Full oracle regret 为 0.018255，低于 0.027709 和 0.026327。
- Full 仍有一次超过 −0.005 margin 的 harmful intervention，因此预算 gate 失败，TDC-19 保持封存。
- 当前允许的新机制结论被缩小为：诊断在中间计算预算下显示前瞻 allocation value，但 harm calibration 和跨 endpoint/source 泛化尚未通过。
- OpenADMET 训练 OOF 的 blind-safe UQ/适用域 v3 已完成：组合信号在 5/6 endpoints 上识别高误差（AUROC > 0.5），低相似度在 5/6 endpoints 上误差更高；但组合 AURC 在 0/6 endpoints 同时优于最佳单一信号，因此 UQ promotion gate 失败。四个回归 endpoint 的 adaptive cross-conformal pooled coverage 约 0.898–0.900，低相似度区间更宽且更保守。
- 已创建 v5 harm-calibrated cross-source ticket；不允许在已读 v4 confirmation 上调阈值后冒充前瞻验证，TDC-19 继续保持封存。
- v5 public cohort v4 治理 gate 已通过并冻结：25,642 条端点记录，17 个候选端点，其中 16 个达到 >=100 身份门槛；覆盖 4 个来源、分类与回归，identity/group fold overlap 均为 0，test labels/leaderboard access 均为 0。该结论只解锁 matched-budget development 实验，不构成算法优越性证据。
- OpenADMET CYP 的 B1 recipe 已在内部 certification 前冻结；一次性 certification 四项 gate 全部通过（macro ST-RAE 0.684642，macro MCC 0.276628）。随后已在无盲标签/无榜单反馈下生成官方 schema 对齐的 750-row regression 与 classification content-addressed submission 文件。外部上传与官方 readout 尚未发生，因此当前主张是 prospective freeze，而不是 blind superiority/rank。
- v5 public source-held-out gate 已完成并失败：Full 的 total gain 1.230112 与 mean oracle regret 0.045966 均劣于 no-diagnosis（1.881659 / 0.005245）和 shuffled（1.771212 / 0.012148）；source-cluster direction probabilities 仅 0.056/0.3585。Full harmful interventions 为 0，但这不足以晋级。按预注册 early stop，fresh seeds 与 TDC-19 confirmation 不运行。
