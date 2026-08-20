"""Fail-closed metadata and citation-coverage audit for the v2.7 paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "huang2021tdc",
    "swanson2024admetai",
    "notwell2023maplight",
    "zhao2026benchmarkadmet",
    "wang2025omnimol",
    "jiang2025asemol",
    "jing2025mumo",
    "zheng2025llm4sd",
    "cai2026chemfm",
    "klaser2024minimol",
    "chan2025mlebench",
    "huang2024mlagentbench",
    "koleiev2026tdcaudit",
    "ji2024unimol2",
    "zhou2023unimol",
}
REQUIRED_FIELDS = {"author", "title", "year", "url"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_entries(text: str) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    starts = list(re.finditer(r"(?m)^@(\w+)\{([^,\s]+),\s*$", text))
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.end() : end]
        key = match.group(2)
        if key in entries:
            raise ValueError(f"Duplicate BibTeX key: {key}")
        fields = {
            field.lower(): value.strip()
            for field, value in re.findall(
                r"(?ms)^\s*(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$", block
            )
        }
        entries[key] = {"type": match.group(1).lower(), "fields": fields}
    return entries


def audit(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    bibliography = root / "paper/references.bib"
    manuscript = root / "MANUSCRIPT_V24.md"
    entries = parse_entries(bibliography.read_text(encoding="utf-8"))
    keys = set(entries)
    missing_required_keys = sorted(REQUIRED_KEYS - keys)
    missing_fields = {
        key: sorted(REQUIRED_FIELDS - set(entry["fields"]))
        for key, entry in entries.items()
        if REQUIRED_FIELDS - set(entry["fields"])
    }
    missing_identifiers = sorted(
        key
        for key, entry in entries.items()
        if not ({"doi", "eprint"} & set(entry["fields"]))
        and entry["type"] not in {"inproceedings"}
    )
    identifiers: dict[str, list[str]] = {}
    for key, entry in entries.items():
        fields = entry["fields"]
        for field in ("doi", "eprint"):
            if field in fields:
                normalized = f"{field}:{fields[field].lower()}"
                identifiers.setdefault(normalized, []).append(key)
    duplicate_identifiers = {
        value: sorted(found) for value, found in identifiers.items() if len(found) > 1
    }

    manuscript_text = manuscript.read_text(encoding="utf-8")
    cited = set(re.findall(r"@([A-Za-z0-9_:-]+)", manuscript_text))
    unresolved_citations = sorted(cited - keys)
    uncited_entries = sorted(keys - cited)
    passed = not any(
        (
            missing_required_keys,
            missing_fields,
            missing_identifiers,
            duplicate_identifiers,
            unresolved_citations,
            uncited_entries,
        )
    )
    return {
        "version": "v2.7",
        "entry_count": len(entries),
        "required_entry_count": len(REQUIRED_KEYS),
        "required_keys_present": not missing_required_keys,
        "missing_required_keys": missing_required_keys,
        "missing_required_fields": missing_fields,
        "missing_persistent_identifiers": missing_identifiers,
        "duplicate_identifiers": duplicate_identifiers,
        "manuscript_citation_count": len(cited),
        "unresolved_manuscript_citations": unresolved_citations,
        "uncited_bibliography_entries": uncited_entries,
        "passed": passed,
        "sources": {
            "bibliography": {
                "path": "paper/references.bib",
                "sha256": _sha256(bibliography),
            },
            "manuscript": {
                "path": "MANUSCRIPT_V24.md",
                "sha256": _sha256(manuscript),
            },
        },
    }


def markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    return "\n".join(
        [
            "# Bibliography and citation audit",
            "",
            f"Status: **{status}**.",
            "",
            f"- BibTeX entries: {report['entry_count']}",
            f"- Distinct manuscript citation keys: {report['manuscript_citation_count']}",
            f"- Missing required keys: {len(report['missing_required_keys'])}",
            f"- Entries with missing required fields: {len(report['missing_required_fields'])}",
            f"- Entries missing a persistent identifier: {len(report['missing_persistent_identifiers'])}",
            f"- Duplicate DOI/arXiv identifiers: {len(report['duplicate_identifiers'])}",
            f"- Unresolved manuscript citations: {len(report['unresolved_manuscript_citations'])}",
            f"- Uncited bibliography entries: {len(report['uncited_bibliography_entries'])}",
            "",
            "This audit checks metadata completeness and internal citation resolution; it does not replace publisher-side metadata validation.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/bibliography_audit_v27")
    )
    args = parser.parse_args()
    report = audit(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(json.dumps(report, sort_keys=True))
    print(json.dumps({"passed": True, "entry_count": report["entry_count"]}))


if __name__ == "__main__":
    main()
