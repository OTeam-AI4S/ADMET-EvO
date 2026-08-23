# ADMET Agent 跨 22 endpoints 的固定模型比较

榜单快照：`2026-08-17T13:50:54.441113+00:00`

## 主要结果

- 估算 Top-1：5/22（AMES, Half_Life_Obach, Lipophilicity_AstraZeneca, PPBR_AZ, Solubility_AqSolDB）
- Top-5 / Top-10：18/22 / 21/22
- 联合平均名次：4.545；中位名次：4.0
- 平均 rank percentile：0.818
- 在 Agent + 14 个完整覆盖固定模型中：第 2 名
- 平均名次超过：13/14 个固定模型
- 尚未超过：MapLight + GNN

## 完整覆盖固定模型

| 模型 | 22-task 联合平均名次 | 平均 rank percentile | Agent 对它 W/T/L |
|---|---:|---:|---:|
| MapLight + GNN | 3.864 | 0.862 | 11/0/11 |
| MapLight | 5.318 | 0.783 | 13/0/9 |
| Chemprop-RDKit | 10.227 | 0.537 | 20/0/2 |
| ContextPred | 11.273 | 0.493 | 16/0/6 |
| AttrMasking | 11.545 | 0.480 | 17/1/4 |
| Chemprop | 12.636 | 0.416 | 21/0/1 |
| RDKit2D + MLP (DeepPurpose) | 13.455 | 0.381 | 19/0/3 |
| AttentiveFP | 15.318 | 0.284 | 21/0/1 |
| Euclia ML model | 16.136 | 0.241 | 21/0/1 |
| GCN | 16.182 | 0.241 | 21/0/1 |
| NeuralFP | 16.182 | 0.242 | 21/0/1 |
| Basic ML | 16.409 | 0.226 | 21/0/1 |
| Morgan + MLP (DeepPurpose) | 17.273 | 0.187 | 20/1/1 |
| CNN (DeepPurpose) | 17.682 | 0.169 | 20/0/2 |

## 解释边界

- Ranks are estimated by inserting frozen Agent scores into a captured public leaderboard; they are not official submitted ranks.
- Raw metrics are not averaged because AUROC, AUPRC, Spearman and MAE have different scales and directions.
- Only exact model labels present on all 22 captured pages are eligible for the complete-coverage comparison.
- Mean rank is descriptive and does not by itself establish statistical superiority or equal compute budgets.

因此可表述为：Agent 已超过绝大多数完整覆盖的单一固定模型，但当前
证据不能表述为‘22-task 平均超过所有模型’；MapLight + GNN 的联合
平均名次仍更好。
