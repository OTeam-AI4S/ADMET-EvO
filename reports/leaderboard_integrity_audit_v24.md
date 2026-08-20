# Leaderboard consistency audit

Snapshot: `2026-08-13T07:09:00.663311+00:00`

Status: **manual_verification_required**

High cross-task overlap of exact (model, mean, std) triples is a leaderboard-integrity warning. It does not prove which page is wrong and must not be used to silently replace a threshold.

| Left task | Right task | Exact shared rows | Union rows | Jaccard |
|---|---|---:|---:|---:|
| Bioavailability_Ma | Pgp_Broccatelli | 16 | 20 | 0.800 |

## Exact duplicated rows

### Bioavailability_Ma / Pgp_Broccatelli

| Model | Mean | Std |
|---|---:|---:|
| AttentiveFP | 0.892 | 0.012 |
| AttrMasking | 0.929 | 0.006 |
| Basic ML | 0.818 | 0 |
| CNN (DeepPurpose) | 0.908 | 0.012 |
| Chemprop | 0.86 | 0.036 |
| Chemprop-RDKit | 0.886 | 0.016 |
| ContextPred | 0.923 | 0.005 |
| Euclia ML model | 0.845 | 0.003 |
| GCN | 0.895 | 0.021 |
| MapLight | 0.93 | 0.002 |
| MapLight + GNN | 0.938 | 0.002 |
| Morgan + MLP (DeepPurpose) | 0.88 | 0.006 |
| NeuralFP | 0.902 | 0.02 |
| RDKit2D + MLP (DeepPurpose) | 0.918 | 0.007 |
| SimGCN | 0.929 | 0.01 |
| ZairaChem | 0.935 | 0.006 |
