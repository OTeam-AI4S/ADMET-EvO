# Test-firewall incident：2026-08-20

状态：`contained_not_reversible`。

在检查本地 hERG 数据文件结构时，一个 shell/pandas 诊断命令错误地包含了官方
`test.csv`，使 test 标签汇总出现在终端输出中。该信息没有进入训练、候选选择、阈值、
路由或项目结果文件，也没有修改任何既有冻结 recipe；但对当前外部数据研究票据而言，
“从未访问 test”的声明已经不成立，不能用 `test_loaded=false` 掩盖。

处置：

- 后续外部数据票据只能读取 `data/admet_group/herg/train_val.csv`。
- 本票据只能声明 `test_feedback_used=false`，不能声明 `test_accessed=false`。
- 任何结果不得修改现有 TDC-22 冻结 recipe 或应用资格。
- `validate_tdc_discovery_path()` 已加入路径层防火墙，canonical ADMET `test.csv` 会在读取前失败。
- 本事件永久保留为负结果与自我纠错证据。

既有应用资格与冻结选择发生在该事件之前，因此其历史选择过程不被追溯性改写；但任何在
本事件之后生成的新科研证据都必须引用本 incident 并清楚限定 claim。
