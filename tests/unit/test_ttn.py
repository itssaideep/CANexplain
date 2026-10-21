"""Unit tests for Binary Tree Tensor Network (TTN) core and Born machine."""

import os
import tempfile

import numpy as np

from canexplain.models.ttn.born_machine import TTNBornMachine
from canexplain.models.ttn.optimizer import TTNCorrelationOptimizer
from canexplain.models.ttn.tree_tensor import BinaryTreeTensorNetwork
from canexplain.preprocessing.encoding import trigonometric_feature_map


def test_ttn_initialization_and_layer_structure():
    """Validates binary tree depth, tensor shapes, and leaf count."""
    for num_leaves in [4, 8, 16, 32]:
        ttn = BinaryTreeTensorNetwork.initialize_random(
            num_leaves=num_leaves,
            physical_dim=2,
            bond_dimension=8,
            seed=42,
        )
        assert ttn.num_leaves == num_leaves
        assert ttn.num_layers == int(np.log2(num_leaves))

        # Check leaf layer has num_leaves // 2 nodes
        assert len(ttn.layers[0]) == num_leaves // 2
        # Check root layer has 1 node
        assert len(ttn.layers[-1]) == 1
        # Root tensor parent dimension is 1
        assert ttn.layers[-1][0].shape[2] == 1


def test_ttn_sample_and_batch_contraction_consistency():
    """Asserts contract_sample matches contract_batch across all batch elements."""
    num_leaves = 16
    ttn = BinaryTreeTensorNetwork.initialize_random(
        num_leaves=num_leaves,
        physical_dim=2,
        bond_dimension=8,
        seed=101,
    )

    rng = np.random.default_rng(202)
    batch_size = 10
    # Generate random normalized quantum states for 16 leaves
    raw_feats = rng.uniform(0.0, 1.0, size=(batch_size, num_leaves))
    phi_batch = np.array([trigonometric_feature_map(row) for row in raw_feats])

    batch_amplitudes = ttn.contract_batch(phi_batch)
    assert len(batch_amplitudes) == batch_size

    for b in range(batch_size):
        single_amp = ttn.contract_sample(phi_batch[b])
        np.testing.assert_allclose(
            single_amp,
            batch_amplitudes[b],
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"Discrepancy between single and batch contraction at sample {b}",
        )


def test_ttn_norm_and_born_machine_nll():
    """Validates state norm Z > 0 and finite NLL anomaly scores."""
    num_leaves = 8
    bm = TTNBornMachine(
        num_leaves=num_leaves,
        physical_dim=2,
        bond_dimension=4,
        seed=42,
    )

    z = bm.norm_squared
    assert isinstance(z, float)
    assert z > 0.0
    assert np.isfinite(z)

    # Test scoring
    rng = np.random.default_rng(77)
    sample_feats = rng.uniform(0.0, 1.0, size=(5, num_leaves))
    phi_batch = np.array([trigonometric_feature_map(row) for row in sample_feats])

    scores = bm.score_samples(phi_batch)
    assert len(scores) == 5
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_ttn_born_machine_serialization():
    """Asserts TTNBornMachine saves to JSON and reloads with identical predictions."""
    num_leaves = 16
    bm = TTNBornMachine(
        num_leaves=num_leaves,
        physical_dim=2,
        bond_dimension=6,
        seed=999,
    )

    rng = np.random.default_rng(42)
    test_feats = rng.uniform(0.0, 1.0, size=(8, num_leaves))
    phi_test = np.array([trigonometric_feature_map(row) for row in test_feats])
    orig_scores = bm.score_samples(phi_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "ttn_model.json")
        bm.save(save_path)
        assert os.path.exists(save_path)

        loaded_bm = TTNBornMachine.load(save_path)
        assert loaded_bm.num_leaves == bm.num_leaves
        assert loaded_bm.bond_dimension == bm.bond_dimension
        loaded_scores = loaded_bm.score_samples(phi_test)

        np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-10, atol=1e-10)


def test_ttn_correlation_optimizer():
    """Validates that TTNCorrelationOptimizer sweeps complete and track loss history."""
    num_leaves = 8
    bm = TTNBornMachine(
        num_leaves=num_leaves,
        physical_dim=2,
        bond_dimension=4,
        seed=123,
    )

    rng = np.random.default_rng(456)
    train_feats = rng.uniform(0.1, 0.4, size=(20, num_leaves))
    phi_train = np.array([trigonometric_feature_map(row) for row in train_feats])

    val_feats = rng.uniform(0.1, 0.4, size=(10, num_leaves))
    phi_val = np.array([trigonometric_feature_map(row) for row in val_feats])

    optimizer = TTNCorrelationOptimizer(
        max_bond_dimension=4,
        learning_rate=0.2,
        num_sweeps=2,
    )
    loss_history = optimizer.fit(bm, phi_train, phi_val=phi_val)

    assert len(loss_history) == 3  # initial + 2 sweeps
    assert len(optimizer.val_loss_history) == 3
    assert all(np.isfinite(loss_val) for loss_val in loss_history)
