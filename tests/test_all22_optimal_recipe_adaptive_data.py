from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_all22_optimal_recipe_adaptive_data import (
    _load_mapping,
    _mass_capped_frame,
    _runtime_blocker,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v27_mapping_covers_all_22_tasks() -> None:
    mapping, digest = _load_mapping(
        ROOT / "runs/v27-maplight-gnn-freeze/full-22-task.mapping.json"
    )
    assert len(mapping) == 22
    assert len(digest) == 64


def test_unweighted_materialization_never_upsamples_external_rows() -> None:
    native = pd.DataFrame(
        {
            "standard_smiles": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "Y": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    external = pd.DataFrame(
        {"standard_smiles": ["N", "NN", "NNN", "NNNN"], "Y": [5.0, 6.0, 7.0, 8.0]}
    )
    first, audit = _mass_capped_frame(
        native,
        external,
        external_mass_ratio=0.40,
        external_row_weight_cap=0.50,
        seed=17,
    )
    second, second_audit = _mass_capped_frame(
        native,
        external,
        external_mass_ratio=0.40,
        external_row_weight_cap=0.50,
        seed=17,
    )
    pd.testing.assert_frame_equal(first, second)
    assert audit == second_audit
    assert len(first) == 7
    assert np.sum(first["standard_smiles"].isin(native["standard_smiles"])) == 5
    assert np.sum(first["standard_smiles"].isin(external["standard_smiles"])) == 2
    assert audit["no_row_upsampling"] is True


def test_deep_recipes_fail_closed_without_required_runtime() -> None:
    root = ROOT
    assert _runtime_blocker(
        {"kind": "recipe", "recipe": "finetuned_unimol2"}, "cpu", root
    ) == "frozen Uni-Mol2 fine-tuning requires an allocated CUDA device"
    blocker = _runtime_blocker(
        {"kind": "recipe", "recipe": "maplight_gnn"}, "cpu", root
    )
    assert blocker in (None, "frozen MapLight+GNN isolated runtime is unavailable")
