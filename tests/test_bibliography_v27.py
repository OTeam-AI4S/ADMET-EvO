import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/audit_bibliography_v27.py"


def _module():
    spec = importlib.util.spec_from_file_location("audit_bibliography_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bibliography_is_complete_unique_and_fully_cited():
    report = _module().audit(ROOT)
    assert report["entry_count"] == report["required_entry_count"] == 15
    assert report["required_keys_present"] is True
    assert report["missing_required_fields"] == {}
    assert report["missing_persistent_identifiers"] == []
    assert report["duplicate_identifiers"] == {}
    assert report["unresolved_manuscript_citations"] == []
    assert report["uncited_bibliography_entries"] == []
    assert report["passed"] is True
