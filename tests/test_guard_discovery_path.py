from pathlib import Path

import pytest

from admet_agent.guard import ProtocolViolation, validate_tdc_discovery_path


def test_tdc_discovery_guard_allows_train_val_and_rejects_test():
    train = validate_tdc_discovery_path("data/admet_group/herg/train_val.csv")
    assert train.name == "train_val.csv"
    with pytest.raises(ProtocolViolation, match="official TDC test path"):
        validate_tdc_discovery_path("data/admet_group/herg/test.csv")


def test_external_source_heldout_namespace_is_not_confused_with_tdc_test():
    path = validate_tdc_discovery_path(Path("data/external/chembl/herg/source_heldout.csv"))
    assert path.name == "source_heldout.csv"
