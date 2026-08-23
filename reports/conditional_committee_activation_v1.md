# TDC ADMET 22：条件式多 Agent 激活回放

## 结论

生产默认保持单 Agent；只有会改变数据治理、成本风险或验证后纠错的决策，
才调用专门角色。该门控没有启动训练，也没有读取冻结 test。

- 默认单 Agent：20/22 tasks
- 条件升级：2/22 tasks（CYP2C9_Substrate_CarbonMangels, hERG）
- 避免默认委员会调用：20/22 （90.9%）
- 避免逻辑角色分配：61/66 （92.4%）
- 已安排纠错覆盖：2/2
- 已观察候选分歧覆盖：1/1
- Fit budget multiplier：1.0；新增模型 fits：0

| Endpoint | 原动作 | 验证结论 | 后续决策 | 运行模式 | 激活角色 |
|---|---|---|---|---|---|
| CYP2C9_Substrate_CarbonMangels | `model-imbalance-aware` | inconclusive | repair_evidence | targeted_review | model_researcher, skeptic |
| hERG | `data-leakage-safe` | refuted | switch_axis | full_committee | data_steward, model_researcher, skeptic |

## 怎么解释“有效”

- 机制有效：它覆盖了历史上唯一真正改变候选集合的 hERG，也覆盖了全部
  两个已安排的验证后纠错任务。
- 冗余受控：20/22 tasks 不升级；委员会仍经过同一确定性仲裁器和同一
  candidate budget，因此不会把多角色数量直接乘到训练 fits 上。
- 效果优势尚未证明：hERG 的委员会数据方案验证为 inconclusive，随后
  被安全停止。因此当前证据支持多 Agent 作为审计、反证与异常处理层，
  不支持把它宣传成普遍提高预测分数的原因。

## 正式上线规则

1. 普通 feature/model action 与无后续价值的 stop：单 Agent。
2. 选中的 data action 或外部数据准入：Data Steward + Skeptic。
3. 高成本表示（包括 Uni-Mol2）或 model 纠错：Model Researcher + Skeptic。
4. 跨轴真实分歧：三个角色进入 full committee；确定性仲裁后仍只进入
   一个共享执行队列。
5. hardness 本身、轻微数据告警、inconclusive 标签本身均不能触发委员会；
   必须存在结构化风险或已安排的可执行纠错。

## 证据边界

This replay supports selective process coverage and lower planning redundancy; it does not establish predictive superiority of multi-agent planning over the single-agent policy.
