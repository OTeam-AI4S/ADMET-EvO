"""Build a deterministic, label-free reviewer bundle for ADMET-EvO v2.7."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

from scripts.build_paper_evidence_manifest import hash_file, manifest_digest


VERSION = "v2.7"
BUNDLE_NAME = "admet-evo-v2.7-reviewer-bundle"
DEFAULT_OUTPUT_ROOT = Path("release")
GENERATED_SENTINEL = ".admet_evo_generated_bundle"
ROOT_FILES = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "environment-cpu.yml",
    "environment-gpu.yml",
    "requirements-repro.txt",
)
VENDOR_FILES = (
    "vendor/tdc-admet-bench-audit/models/maplight_gnn/README.md",
    "vendor/tdc-admet-bench-audit/models/maplight_gnn/upstream/LICENSE",
    "vendor/tdc-admet-bench-audit/models/maplight_gnn/upstream/PROVENANCE.md",
    "vendor/tdc-admet-bench-audit/models/maplight_gnn/upstream/maplight_gnn.py",
    "vendor/tdc-admet-bench-audit/models/maplight_gnn/upstream/submission_gnn.ipynb",
    "vendor/tdc-admet-bench-audit/models/maplight_gnn/wrapper/environment.yml",
    "vendor/tdc-admet-bench-audit/models/maplight_gnn/wrapper/run_all.py",
    "vendor/tdc-admet-bench-audit/.git-commit",
    "vendor/tdc-admet-bench-audit/data/tdc_admet_leaderboard.json",
)
REBUILD_EVIDENCE_FILES = (
    "reports/goal_scorecard_v26.json",
    "runs/v24-router-replay/report.json",
    "runs/v23-ablation-study/lipophilicity_scale_comparison.json",
    "runs/v26-cyp2c9-freeze/full-22-task.json",
    "runs/v26-cyp2c9-freeze/full-22-task.mapping.json",
    "runs/v27-maplight-gnn-confirmation/fresh_predictions.npz",
    "runs/v27-maplight-gnn-confirmation/all_development_features.npy",
    "runs/v27-maplight-gnn-confirmation/all_development_features.manifest.json",
    "runs/unimol2-pilot-v22/DILI/results.json",
    "runs/unimol2-pilot-v22/hERG/results.json",
)
FORBIDDEN_PATH_PARTS = {
    ".git",
    "__pycache__",
}
FORBIDDEN_ROOTS = {".venvs", "data", "logs"}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bghp_[A-Za-z0-9]{20,}\b"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbidden_relative_path(relative: Path) -> bool:
    return (
        bool(relative.parts)
        and relative.parts[0] in FORBIDDEN_ROOTS
    ) or any(part in FORBIDDEN_PATH_PARTS for part in relative.parts)


def _tree_files(root: Path, relative: str, suffixes: set[str] | None = None) -> Iterable[Path]:
    base = root / relative
    if not base.exists():
        return ()
    return (
        path
        for path in base.rglob("*")
        if path.is_file()
        and not _forbidden_relative_path(path.relative_to(root))
        and (suffixes is None or path.suffix in suffixes)
    )


def _collect_files(root: Path) -> set[Path]:
    paper_manifest = _load(root / "reports/paper_evidence_manifest_v27.json")
    scorecard = _load(root / "reports/goal_scorecard_v27.json")
    submission = _load(root / "paper/submission/jcheminformatics/manifest.json")
    figures = _load(root / "paper/figures/manifest.json")

    relative_paths = set(ROOT_FILES) | set(VENDOR_FILES) | set(REBUILD_EVIDENCE_FILES)
    relative_paths.update(item["path"] for item in paper_manifest["evidence"].values())
    relative_paths.update(paper_manifest["source_files"])
    relative_paths.update(submission["outputs"])
    relative_paths.update(figures["outputs"])
    relative_paths.update(
        {
            "reports/paper_evidence_manifest_v27.json",
            "paper/submission/jcheminformatics/manifest.json",
            "paper/figures/manifest.json",
        }
    )

    for directory, suffixes in (
        ("admet_agent", {".py"}),
        ("scripts", {".py"}),
        ("tests", {".py"}),
        ("docs", {".md", ".json", ".yml", ".yaml"}),
        ("examples", None),
        ("skills", None),
    ):
        relative_paths.update(str(path.relative_to(root)) for path in _tree_files(root, directory, suffixes))

    for relative in (
        "runs/unimol2-scale-ablation-84m-b8/84m/Lipophilicity_AstraZeneca/lr_0.0001",
        "runs/unimol2-scale-ablation-b8/164m/Lipophilicity_AstraZeneca/lr_0.0001",
    ):
        relative_paths.update(
            str(path.relative_to(root))
            for path in (root / relative).glob("seed_*/results.json")
        )
    for task in ("Bioavailability_Ma", "CYP2C9_Substrate_CarbonMangels"):
        smoke_root = root / "runs/v27-maplight-gnn-smoke" / task / "seed_1"
        relative_paths.add(str((smoke_root / "result.json").relative_to(root)))
        relative_paths.add(
            str((smoke_root / "artifacts/maplight_gnn/runtime_report.json").relative_to(root))
        )

    for provenance in scorecard["provenance"].values():
        source = root / provenance["source"]
        relative_paths.add(str(source.relative_to(root)))
        final_dir = source.parent
        for prediction in final_dir.glob("predictions_seed_*.npz"):
            relative_paths.add(str(prediction.relative_to(root)))
        freeze = provenance.get("freeze")
        if freeze:
            relative_paths.add(freeze)
        elif (final_dir / "freeze.json").exists():
            relative_paths.add(str((final_dir / "freeze.json").relative_to(root)))
        environment = final_dir.parent / "environment.json"
        if environment.exists():
            relative_paths.add(str(environment.relative_to(root)))
        for seed in provenance.get("seed_records", {}).values():
            relative_paths.add(seed["execution_audit"])
            relative_paths.add(seed["runtime_report"])
            runtime_path = Path(seed["runtime_report"])
            relative_paths.add(str(runtime_path.with_name("input.json")))
            relative_paths.add(str(runtime_path.with_name("predictions.npy")))
            runtime = _load(root / runtime_path)
            relative_paths.add(runtime["feature_manifest"])

    files = {root / relative for relative in relative_paths}
    missing = sorted(str(path.relative_to(root)) for path in files if not path.is_file())
    if missing:
        raise ValueError(f"Reviewer bundle sources are missing: {missing}")
    return files


def _audit_source(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root)
    if path.is_symlink():
        raise ValueError(f"Symlinks are forbidden in the reviewer bundle: {relative}")
    if _forbidden_relative_path(relative):
        raise ValueError(f"Forbidden path in reviewer bundle: {relative}")
    lowered = path.name.lower()
    if lowered == ".env" or any(word in lowered for word in ("credential", "private_key")):
        raise ValueError(f"Credential-like file forbidden in reviewer bundle: {relative}")
    data = path.read_bytes()
    for pattern in SECRET_PATTERNS:
        if pattern.search(data):
            raise ValueError(f"Secret-like content detected in reviewer bundle: {relative}")
    if path.suffix == ".npz":
        members = zipfile.ZipFile(path).namelist()
        if any(re.search(r"(?:label|target|y_true)", name, flags=re.IGNORECASE) for name in members):
            raise ValueError(f"Label-like array detected in reviewer prediction file: {relative}")
        return {"npz_members": members, "role": "official_prediction_only"}
    return {"role": "source_or_derived_evidence"}


def _bundle_readme(file_count: int) -> str:
    return f"""# ADMET-EvO v2.7 reviewer bundle

This generated bundle contains {file_count} project/evidence files required to
audit the manuscript, frozen 22-task scorecard, main figures, Supplementary
Information, and source/test implementation. It deliberately excludes local
TDC caches, official test labels, model caches, credentials, logs, virtual
environments, and the full historical `runs/` tree.

## Fast audit

Create the CPU environment and run the claim verifiers:

```bash
conda env create -f environment-cpu.yml
conda activate admet-agent
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python scripts/build_v27_goal_scorecard.py
PYTHONPATH=. python scripts/build_paper_figures_v27.py
PYTHONPATH=. python scripts/build_jcheminf_submission_v27.py
PYTHONPATH=. python scripts/build_paper_evidence_manifest_v27.py
```

The bundle manifest hashes every included file and records the immutable full
freeze. Prediction NPZ files contain model outputs only; no official test label
array is included.

## Full model reproduction boundary

PyTDC downloads the public benchmark data on first use. Classical finalists use
the pinned project environments. Uni-Mol2 runs additionally require
`unimol-tools==0.1.5` and the public 84M checkpoint. MapLight+GNN uses the
included isolated environment and upstream source plus the public
`gin_supervised_masking` checkpoint, whose required SHA-256 is
`6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9`.
Third-party checkpoints are not redistributed in this bundle. Their licenses
and access boundaries are pinned in `reports/third_party_release_audit_v27.*`.
Both artifacts were anonymously downloadable on 2026-08-17. Uni-Mol2's model
repository declares MIT; the separately hosted GIN artifact has no explicit
artifact-level license, so it remains link-only. Its 57 parameter tensors are
bitwise identical to the official DGL-LifeSci checkpoint. Because this bundle
redistributes neither artifact, it is ready for external deposit; third-party
checkpoint redistribution remains prohibited. Historical peak memory was not
recorded, so no absolute memory claim is reproducible or made.

## Conservative leaderboard boundary

The dated result is 18/22 estimated Top-5 and 21/22 Top-10. Bioavailability is
kept below Top-10 until TDC authoritatively resolves the duplicated Bio/Pgp
leaderboard table. The discrepancy audit is evidence, not permission to alter
the displayed threshold.
"""


def _prepare_output(output_root: Path) -> tuple[Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_root / BUNDLE_NAME
    archive = output_root / f"{BUNDLE_NAME}.tar.gz"
    receipt = output_root / f"{BUNDLE_NAME}.archive.json"
    if bundle_dir.exists():
        if not (bundle_dir / GENERATED_SENTINEL).is_file():
            raise ValueError(f"Refusing to replace non-generated directory: {bundle_dir}")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir()
    (bundle_dir / GENERATED_SENTINEL).write_text("generated\n", encoding="utf-8")
    return bundle_dir, archive, receipt


def _write_deterministic_tar(bundle_dir: Path, archive: Path) -> None:
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(bundle_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = Path(BUNDLE_NAME) / path.relative_to(bundle_dir)
                    data = path.read_bytes()
                    info = tarfile.TarInfo(str(relative))
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    tar.addfile(info, io.BytesIO(data))


def build_bundle(repo_root: Path, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = repo_root.resolve()
    output_root = (root / output_root).resolve() if not output_root.is_absolute() else output_root
    files = _collect_files(root)
    bundle_dir, archive, receipt_path = _prepare_output(output_root)
    audits: dict[str, dict[str, Any]] = {}
    for source in sorted(files):
        relative = source.relative_to(root)
        audits[str(relative)] = _audit_source(root, source)
        destination = bundle_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    readme = _bundle_readme(len(files))
    (bundle_dir / "REVIEWER_README.md").write_text(readme, encoding="utf-8")
    inventory = {
        str(path.relative_to(bundle_dir)): hash_file(path)
        for path in sorted(bundle_dir.rglob("*"))
        if path.is_file() and path.name not in {GENERATED_SENTINEL, "MANIFEST.json"}
    }
    paper_manifest = _load(root / "reports/paper_evidence_manifest_v27.json")
    release_audit = _load(root / "reports/third_party_release_audit_v27.json")
    payload = {
        "version": VERSION,
        "bundle_name": BUNDLE_NAME,
        "full_freeze_sha256": paper_manifest["full_freeze_sha256"],
        "scorecard_status": paper_manifest["scorecard_status"],
        "official_test_labels_included": False,
        "official_predictions_included": True,
        "credentials_included": False,
        "local_data_cache_included": False,
        "historical_run_cache_included": False,
        "third_party_checkpoints_included": False,
        "registration_free_checkpoint_access_verified": release_audit["gates"][
            "registration_free_checkpoint_access_verified"
        ],
        "checkpoint_specific_licenses_verified": release_audit["gates"][
            "checkpoint_specific_licenses_verified"
        ],
        "file_count": len(inventory),
        "total_uncompressed_bytes": sum(item["bytes"] for item in inventory.values()),
        "official_prediction_file_count": sum(
            Path(path).name.startswith("predictions_seed_") and path.endswith(".npz")
            for path in inventory
        ),
        "validation_npz_file_count": sum(
            path.endswith(".npz") and not Path(path).name.startswith("predictions_seed_")
            for path in inventory
        ),
        "source_audit": {
            "audited_file_count": len(audits),
            "secret_patterns_detected": 0,
            "label_like_prediction_arrays_detected": 0,
        },
        "files": inventory,
    }
    manifest = {**payload, "manifest_sha256": manifest_digest(payload)}
    (bundle_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_deterministic_tar(bundle_dir, archive)
    receipt_payload = {
        "version": VERSION,
        "bundle_manifest_sha256": hash_file(bundle_dir / "MANIFEST.json")["sha256"],
        "archive": {"path": archive.name, **hash_file(archive)},
        "bundle_manifest": {"path": f"{BUNDLE_NAME}/MANIFEST.json", **hash_file(bundle_dir / "MANIFEST.json")},
        "ready_for_external_deposit": release_audit["gates"][
            "ready_for_external_deposit"
        ],
        "remaining_external_gate": release_audit["gates"]["remaining_external_gate"],
    }
    receipt = {**receipt_payload, "receipt_sha256": manifest_digest(receipt_payload)}
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    receipt = build_bundle(args.repo_root, args.output_root)
    print(json.dumps({
        "archive": receipt["archive"],
        "bundle_manifest_sha256": receipt["bundle_manifest_sha256"],
        "ready_for_external_deposit": receipt["ready_for_external_deposit"],
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
