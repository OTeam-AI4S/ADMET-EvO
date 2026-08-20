import numpy as np
import pandas as pd

from scripts.run_endpoint_teacher_transfer import (
    CANDIDATES,
    exclude_target_molecules,
    summarize,
)


def test_teacher_training_excludes_canonical_target_molecules():
    auxiliary = pd.DataFrame(
        {
            "Drug": ["C(C)O", "CCN", "c1ccccc1"],
            "Y": [1, 0, 1],
        }
    )

    filtered, excluded = exclude_target_molecules(auxiliary, "Drug", ["CCO"])

    assert excluded == 1
    assert filtered["Drug"].tolist() == ["CCN", "c1ccccc1"]


def test_transfer_summary_uses_leave_one_seed_out_selection():
    seed_scores = {
        seed: {
            name: 0.5 + 0.01 * index + 0.001 * seed
            for index, name in enumerate(CANDIDATES)
        }
        for seed in range(1, 6)
    }

    report = summarize(seed_scores)

    expected = list(CANDIDATES)[-1]
    assert report["best_fixed_candidate"] == expected
    assert set(report["loso"]["lcb_0.5"]["selected"].values()) == {expected}
    assert np.isfinite(report["loso"]["lcb_0.5"]["mean"])
