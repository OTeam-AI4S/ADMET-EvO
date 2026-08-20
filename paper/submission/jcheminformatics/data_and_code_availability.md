# Data and code availability draft

The public benchmark data are distributed by Therapeutics Data Commons through
PyTDC. ADMET-EvO source code, frozen recipes, environment specifications,
derived validation evidence, prediction hashes, and figure/SI builders are
prepared for public release under the MIT license.

Both required third-party checkpoints were anonymously downloadable on
2026-08-17 and are identified by immutable SHA-256 hashes in the release audit.
Uni-Mol2's public, ungated model repository declares MIT. The separately hosted
Molfeat `gin_supervised_masking` artifact is public and hash-matched. Its 57
parameter tensors are bitwise identical to DGL-LifeSci's official checkpoint,
but its artifact metadata has no explicit license field; Apache-2.0 on source
code is not treated as a binary-checkpoint redistribution license.

Submission blocker: deposit the exact reviewed link-only repository snapshot
in a registration-free archival repository (recommended: Zenodo release linked
to GitHub), and replace `ARCHIVE_DOI_REQUIRED`. The reviewer bundle
redistributes neither third-party checkpoint and includes no official test
labels. Third-party checkpoint files must not be added unless their artifact
licenses are made explicit.
