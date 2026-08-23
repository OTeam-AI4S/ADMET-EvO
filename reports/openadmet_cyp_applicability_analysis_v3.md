# OpenADMET CYP OOF 不确定性与适用域分析 v3

日期：2026-08-23
证据范围：OpenADMET 训练集的 5-fold chemistry-group OOF 预测；官方 blind test、leaderboard feedback 和 sealed certification labels 均未访问。

## 结论

该实验提供了**部分通过、整体不晋级**的 UQ 证据：组合不确定性能够在 5/6 endpoints 上以 AUROC > 0.5 识别高误差分子，最低相似度四分位在 5/6 endpoints 上误差更高；但是组合分数在 0/6 endpoints 上同时优于最好的单一信号，因此预注册 promotion gate 为 **FAIL**。

这说明 applicability diagnosis 有真实信号，但当前“距离 + seed 分歧 + 分类 margin 的等权平均”不是一个足够好的通用 UQ 算法。论文可以报告 shift/error association 和条件化区间结果，不能声称组合 UQ 优于最佳单信号或已通过前瞻 blind 校准。

## 协议与完整性

- 6 endpoints，10,219 个 endpoint-records，3 个训练 seed，5 个完整 chemistry-group outer folds。
- 距离信号：`1 - nearest outer-train Morgan2/2048 Tanimoto`。
- 模型信号：回归为 seed disagreement；分类为 seed disagreement 与预测 margin 的等权 percentile rank。
- 组合信号：所有可用 label-free 信号的等权 endpoint-local percentile rank。
- 分类 row loss 为 Brier loss；回归为官方 ST-RAE 的 interval-aware numerator contribution。
- v1 在指标计算前因 source-record collapse key 不完整中止；v2 在指标计算前因 RDKit API 兼容性中止。两次失败均有独立 audit，v3 只加入记录键和 API 适配，不改变科学定义或 gate。
- v3 protocol self-hash：`9f5be7a4df914ca70beeff3fbec377bb7155d1f2eccc0f5fd33bfd2c91a904cb`。
- v3 analysis file SHA-256：`1cab0376a7a2ad973dac18173d5ec58d67867353dc1d5f7939833de7ad554731`。

## 高误差检测与 selective risk

高误差定义为 endpoint 内 row loss 严格高于第 75 百分位；AURC 越低越好。

| Endpoint | Type | Distance AUROC | Model AUROC | Combined AUROC | Distance AURC | Model AURC | Combined AURC |
|---|---|---:|---:|---:|---:|---:|---:|
| CYP1A2 direct | reg | 0.499 | 0.561 | 0.540 | 0.3924 | **0.3727** | 0.3884 |
| CYP2C9 direct | reg | 0.522 | 0.469 | 0.492 | **0.2058** | 0.2199 | 0.2173 |
| CYP2D6 TDI | cls | 0.524 | 0.656 | 0.637 | 0.1734 | **0.1654** | 0.1656 |
| CYP2D6 direct | reg | 0.477 | 0.555 | 0.524 | 0.3713 | **0.3362** | 0.3487 |
| CYP3A4 TDI | cls | 0.551 | 0.844 | 0.808 | 0.1210 | **0.0727** | 0.0757 |
| CYP3A4 direct | reg | 0.530 | 0.494 | 0.518 | **0.2581** | 0.2775 | 0.2650 |

核心负结果很稳定：回归 endpoints 中距离和 seed disagreement 各自占优；分类 endpoints 中模型置信信号明显更强。固定等权融合会稀释 endpoint-specific 的最佳信号。

## 相似性分层与 conformal

- 最低相似度四分位的 mean loss 在 5/6 endpoints 高于最高相似度四分位；唯一反例是 CYP2D6 direct regression。
- 四个回归 endpoint 的 combined-adaptive 90% cross-conformal pooled coverage 为 0.8984、0.8994、0.8984、0.9003。
- 最低相似度四分位 coverage 为 0.9335–0.9518，最高相似度四分位为 0.8063–0.8727。区间会在远离训练化学空间时主动变宽，但当前 rank scaling 偏保守：相对 constant conformal，pooled mean width 增加约 4%–9%。

因此可以支持“适用域距离与误差相关，并能驱动更保守的局部区间”这一受限主张；不能支持“所有相似性层都达到条件覆盖”或“adaptive interval 更窄”的主张。

## Gate

| Check | Observed | Requirement | Result |
|---|---:|---:|---:|
| Combined beats both single signals | 0/6 | ≥4/6 | FAIL |
| Combined high-error AUROC > 0.5 | 5/6 | ≥4/6 | PASS |
| Low-similarity loss > high-similarity loss | 5/6 | ≥4/6 | PASS |
| Blind/test/leaderboard access | 0 | 0 | PASS |

整体 promotion：**FAIL**。下一版不得在本结果上继续调融合权重后宣称独立验证；应在新 endpoint/source 上比较预冻结的 endpoint-conditional router，例如 classification 使用 margin/disagreement、regression 根据 source-shift state 在 distance 与 disagreement 之间路由。

## 权威 artifacts

- `reports/openadmet_cyp_applicability_protocol_v3.json`
- `reports/openadmet_cyp_applicability_analysis_v3.json`
- `reports/openadmet_cyp_applicability_v1_aborted_audit.json`
- `reports/openadmet_cyp_applicability_v2_aborted_audit.json`
- `admet_agent/applicability.py`
- `scripts/analyze_openadmet_applicability_v3.py`
- `tests/test_applicability.py`
