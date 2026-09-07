from __future__ import annotations

import re
import tomllib
from pathlib import Path

import admet_agent


ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_are_consistent():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_version = re.search(r"^version: (\S+)$", citation, re.MULTILINE)

    assert citation_version is not None
    assert project["project"]["version"] == admet_agent.__version__
    assert citation_version.group(1) == admet_agent.__version__


def test_release_urls_target_the_organization_repository():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    urls = project["project"]["urls"]

    assert all("github.com/OTeam-AI4S/ADMET-EvO" in value for value in urls.values())
