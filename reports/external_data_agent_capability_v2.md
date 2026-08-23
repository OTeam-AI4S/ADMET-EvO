# ADMET-EvO 外部数据 Agent 能力声明 v2

日期：2026-08-23
状态：**gated semi-autonomous capability with one completed real-source development loop**

## 结论

当前系统能在 EndpointContract 约束下自主完成候选来源发现、语义判定、质量/许可元数据检查、结构污染防火墙、数据用途分级和 source-held-out 净收益验证。它不能对任意网页或数据库无约束下载数据并自动并入训练；自动下载被明确禁止，直接监督训练还要求自动审计通过、release state 晋级和科学审阅签字。

因此允许的准确表述是：

> ADMET-EvO implements a gated, endpoint-contract-first external-data research workflow that can discover registered sources, adjudicate semantic compatibility, fail closed on unsafe evidence, and promote an audited source only after matched source-held-out validation.

不允许表述为“Agent 已能对所有 ADMET endpoint 完全自主获取最佳数据”。

## 五层能力审计

| 层级 | 当前能力 | 状态 | 证据 |
|---|---|---|---|
| 1. 发现 | 根据 endpoint 调用已注册 connector，形成候选来源 register | 已实现 | `ExternalDataScout`; ChEMBL/TDC-related hERG connectors |
| 2. 质量与语义判断 | 检查 target、task type、species、assay、label threshold/direction、unit、transform、provenance、license | 已实现 | `judge_endpoint`; EndpointContract |
| 3. 获取 | 初始化 collection bundle，并通过 connector/本地冻结来源收集 | 部分自主 | `discover/audit` 自动；`collect/update` 需要 connector 或人工提供访问路径；automatic download=false |
| 4. 安全准入 | 标准化结构，排除 train/validation/test identity、来源重叠和近邻污染；保留 quarantine/reject 原因 | 已实现 | external structure firewall 与 OpenADMET sealed firewall |
| 5. 建模与净收益验证 | 将来源路由为 direct supervised、bridge/auxiliary、pretrain 或 reject；运行 source-held-out matched treatment/control | 已完成一个真实开发闭环 | OpenADMET–TDC/Veith external propensity v2 |

## 数据用途分级

### Tier A — supervised modeling candidate

必须同时满足 endpoint/assay/species/label/unit/方向兼容、来源和许可清晰、结构污染为零、自动审计无 critical issue、manifest release state 为 `modeling_ready_discovery`，并有科学审阅签字。任何一项缺失均不得直接合并标签。

### Tier B — bridge / auxiliary evidence

endpoint 生物学相关但 assay 或 label 不完全等价时，只允许 source-conditioned auxiliary head、propensity/multifidelity treatment 或显式 bridge study。OpenADMET–TDC/Veith 属于此类：Veith qHTS active/inactive 不等价于 OpenADMET direct-inhibition pIC50 或 TDI 标签。

### Tier C — pretraining-only evidence

质量一般、标签弱或 assay 映射不足，但结构与许可可用的数据，只进入 label-blind/weakly supervised pretraining，不作为目标 endpoint 真值。

### Reject / quarantine

target identity、task type、species、标签方向/阈值、来源、许可或污染检查失败的数据，保留原因并停止，不因样本量大而放宽准入。

## 已完成的真实闭环

- 冻结外部 TDC/Veith CYP bridge：29,008 行；未读取 TDC test 或 OpenADMET blind labels。
- 完成规范结构、identity firewall、外部来源 propensity 和 source-held-out 检验。
- 六个 OpenADMET endpoint 的冻结 B1 交叉验证中：
  - 四个回归 endpoint 的 ST-RAE treatment 增益全部为正：+0.001400 至 +0.007864；
  - 两个 TDI 分类 endpoint 的 MCC treatment 增益均为负：−0.007959、−0.003875。
- 据此在 blind 结果前冻结 task-type route：
  - regression → external propensity treatment；
  - classification/TDI → B1 control；
  - unseen endpoint type → B1 control。
- 该结果证明 Agent 能拒绝负迁移，而不是证明外部数据普遍有效；OpenADMET blind 尚未提供独立优越性证据。

## hERG 发现能力的边界案例

自动 scout 找到三个已注册候选来源，但 3/3 均被判为 `bridge_required`，没有数据被直接拼入训练。这是 fail-closed 的正确行为：来源相关不等于 endpoint 标签可交换。

## 尚未具备的完全自治能力

1. 任意网络/文献的开放式检索与高召回抽取；当前主要依赖注册 connector 和已提供的本地来源。
2. 自动处理登录、付费、API key、robots/rate limit 和复杂许可授权。
3. 在缺少 assay 元数据时可靠推断标签等价性。
4. 无人签字地把外部标签升级为 direct supervised training。
5. 在所有 22 个 endpoint 上证明外部数据动作带来独立 held-out 增益。

这些限制是当前 claim boundary，也是防止“自动抓取大量但不兼容数据”导致泄漏和负迁移的安全设计。
