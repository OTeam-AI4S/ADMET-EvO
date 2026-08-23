import pytest

torch = pytest.importorskip("torch")

from scripts.openadmet_multifidelity_model import (
    AssayConditionedCYPModel,
    ModelConfig,
    interval_distance,
    mixed_supervised_loss,
    tdi_consistency_loss,
)


def test_assay_conditioned_model_shapes_and_context_changes_prediction():
    torch.manual_seed(7)
    model = AssayConditionedCYPModel(
        ModelConfig(molecule_width=8, trunk_width=12, context_width=4, hidden_width=10)
    ).eval()
    molecule = torch.ones(3, 8)
    base = model(
        molecule,
        torch.tensor([0, 0, 0]),
        torch.tensor([0, 0, 0]),
        torch.tensor([0, 0, 0]),
        torch.tensor([0, 0, 0]),
    )
    changed = model(
        molecule,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2]),
    )
    assert base["regression_mean"].shape == (3,)
    assert base["classification_logit"].shape == (3,)
    assert not torch.allclose(
        base["regression_mean"], changed["regression_mean"]
    )
    assert model.manifest()["trainable_parameters"] > 0


def test_interval_distance_is_zero_inside_and_uses_point_fallback():
    distance = interval_distance(
        torch.tensor([5.5, 7.0, 4.0]),
        torch.tensor([5.4, 6.0, 5.0]),
        torch.tensor([5.2, 5.5, float("nan")]),
        torch.tensor([5.8, 6.5, float("nan")]),
    )
    assert torch.allclose(distance, torch.tensor([0.0, 0.5, 1.0]))


def test_joint_loss_backpropagates_and_tdi_consistency_rewards_aligned_shift():
    output = {
        "regression_mean": torch.tensor([0.1, 0.0], requires_grad=True),
        "regression_log_variance": torch.zeros(2, requires_grad=True),
        "classification_logit": torch.tensor([0.0, 1.0], requires_grad=True),
    }
    loss = mixed_supervised_loss(
        output,
        target=torch.tensor([0.0, 1.0]),
        task_is_classification=torch.tensor([0, 1]),
        lower=torch.tensor([-0.2, float("nan")]),
        upper=torch.tensor([0.2, float("nan")]),
        measurement_variance=torch.tensor([0.1, 0.0]),
        row_weight=torch.ones(2),
        interval_weight=1.0,
        positive_weight=torch.ones(2),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert output["regression_mean"].grad is not None
    assert output["regression_mean"].grad[0] == 0

    aligned = tdi_consistency_loss(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        direct_mean=torch.tensor([5.0]),
        direct_std=torch.tensor([1.0]),
        tdi_mean=torch.tensor([5.0]),
        tdi_std=torch.tensor([1.0]),
    )
    contradicted = tdi_consistency_loss(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        direct_mean=torch.tensor([5.0]),
        direct_std=torch.tensor([1.0]),
        tdi_mean=torch.tensor([5.0]),
        tdi_std=torch.tensor([1.0]),
    )
    assert aligned < contradicted
