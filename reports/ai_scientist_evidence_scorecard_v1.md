# TDC ADMET-22：AI Scientist 证据向量评分卡

- Overall status: `qualified_prediction_local_process_advantage_external_validation_pending`
- Scorecard build loaded TDC test: false
- Disclosed post-qualification test-access incidents: 1
- 原则：A weighted average could hide an unsafe action or failed prediction gate; capability dimensions remain separate.

## 一眼结论

| 能力 | Agent | 对照/边界 | 证据状态 |
|---|---|---|---|
| 预测合格 | 18/22 Top-5；21/22 Top-10 | 只作应用资格，不宣称 AutoML 优越 | certified |
| 22-task 同预算预测对照 | Agent−expanded AutoML -0.0098 | 95% CI [-0.0277, +0.0065]；W/T/L=[5, 7, 10] | complete; noninferiority FAIL |
| 复合研究决策 | macro 1.00；unsafe=0 | rules 0.86；expanded AutoML 0.54 | local sealed |
| Agent−rules | +0.14 | bootstrap 95% CI [0.04, 0.26] | local positive |
| 数据治理 | accuracy 1.00；unsafe merge=0 | expanded AutoML accuracy 0.27；unsafe=7；complete rules 与 Agent 持平 | AutoML advantage only |
| 真实外部数据票据 | frozen attempts=2；fits=0；unsafe merge=0 | ChEMBL API HTTP 500，未接受部分数据 | failed closed / gain pending |
| 自动数据准入栈 | 发现 3 个候选；bridge=3；direct merge=0 | endpoint judge + 密封三重结构防火墙；真实增益未证明 | mechanism ready |
| 图表征门控 | 2D/GIN/gate ready=True/True/True | distinct MPNN ready=False；票据上限 20 fits | blocked before training |
| 反馈纠错 | macro 1.00 | expanded AutoML 0.30；complete rules 持平 | 2 observed cases + composite |
| 成本停止 | pilot 14/20 fits，节省 30% | composite 估算节省 37.5% | pilot measured / composite estimated |
| 跨任务学习 | 100 fits，2 actions eligible | routing outcome advantage 尚未证明 | causal priors ready |
| no-memory 消融 | selection changed 0/22 | legacy mean gain -0.00031 | negative result retained |
| no-hardness 消融 | critical diagnosis recall 0.00 | full Agent 1.00 | mechanism ablation |
| no-data 消融 | accuracy 0.36；unsafe=0 | full Agent 1.00 | safe but over-conservative |

## Claim gates

| Gate | Status |
|---|---|
| `application_performance_qualified` | PASS |
| `local_integrated_process_advantage_over_rules` | PASS |
| `local_data_advantage_over_expanded_automl` | PASS |
| `feedback_advantage_over_scalar_automl` | PASS |
| `prospective_cross_task_prior_ready` | PASS |
| `key_process_ablations_executed` | PASS |
| `formal_22_task_same_budget_predictive_control_complete` | PASS |
| `external_data_discovery_endpoint_and_firewall_mechanism_ready` | PASS |
| `independently_authored_external_process_validation_complete` | PENDING |
| `source_held_out_external_data_gain_complete` | PENDING |

## 当前允许的技术叙事

On TDC ADMET-22, the frozen Agent meets the application-performance qualification. On same-team sealed process challenges it outperforms scalar AutoML and locked static rules while making zero unsafe actions, and it shows measured cost-aware stopping. A completed fresh-seed equal-budget predictive control did not establish noninferiority to expanded AutoML and is retained as negative evidence. Two real ChEMBL acquisition attempts failed closed before training; a post-qualification test-access incident is disclosed and no test feedback was used. The autonomous data-admission stack now discovers sources, judges endpoint semantics, and applies a sealed three-way structure firewall, but external predictive gain remains unproven.

## 仍然禁止的表述

- predictively superior to AutoML under a completed prospective same-budget 22-task run
- generally superior autonomous scientist on independent external tasks
- predictive gain from autonomously acquired external data

## 最小剩余证据

- Obtain independently authored composite research episodes before claiming general autonomous-research superiority.
- Complete an admitted source-held-out external-data comparison; two frozen ChEMBL acquisitions failed closed on external HTTP 500 before raw data or training.
- Provide one content-addressed distinct MPNN fingerprint provider, then run the <=20-fit graph representation/fusion ticket; do not expand it to all 22 tasks by default.

## 为什么正式对照分成两条 track

- 22-task predictive track：streamlined Agent、expanded AutoML、strong-fixed。
- same-action-budget process track：rules、random、no-memory、no-hardness、no-data。

Rules/random/ablations test research decisions; forcing every process control to retrain every predictive model would spend compute without answering its causal capability question.

这里不计算单一总分。预测、安全、科研决策和外部数据是不可互相补偿的门控：
例如更高 process score 不能抵消预测不合格，更多 Top-5 也不能抵消 unsafe merge。
