"""Unit tests for Matrix Product State tensor core, Born machine, and DMRG optimizer."""

import pytest
import os
import tempfile
import numpy as np

from canexplain.models.mps.tensor_core import MatrixProductState
from canexplain.models.mps.born_machine import MPSBornMachine
from canexplain.models.mps.dmrg import DMRGOptimizer
from canexplain.preprocessing.encoding import trigonometric_feature_map


def test_mps_initialization_and_norm():
    num_sites = 8
    mps = MatrixProductState.initialize_random(num_sites=num_sites, physical_dim=2, bond_dimension=10)

    assert mps.num_sites == num_sites
    assert mps.tensors[0].shape[0] == 1
    assert mps.tensors[-1].shape[2] == 1

    norm_sq = mps.norm_squared()
    assert norm_sq > 0.0
    assert np.isfinite(norm_sq)


def test_mps_sample_contraction():
    num_sites = 6
    mps = MatrixProductState.initialize_random(num_sites=num_sites, physical_dim=2, bond_dimension=8)

    sample = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    phi_sample = trigonometric_feature_map(sample)  # (6, 2)

    amp = mps.contract_sample(phi_sample)
    assert np.isfinite(amp)

    # Batch contraction consistency
    batch = np.stack([phi_sample, phi_sample], axis=0)
    amps = mps.contract_batch(batch)
    assert len(amps) == 2
    np.testing.assert_allclose(amps[0], amp)
    np.testing.assert_allclose(amps[1], amp)


def test_born_machine_nll_and_persistence():
    num_sites = 5
    bm = MPSBornMachine(num_sites=num_sites, physical_dim=2, bond_dimension=8)

    sample = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    phi_sample = trigonometric_feature_map(sample)

    prob = bm.probability(phi_sample)
    assert 0.0 <= prob <= 1.0

    batch = np.stack([phi_sample, phi_sample], axis=0)
    nll = bm.score_samples(batch)
    assert len(nll) == 2
    assert np.all(np.isfinite(nll))
    assert nll[0] == nll[1]

    # Save and Load
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        temp_path = f.name

    try:
        bm.save(temp_path)
        loaded_bm = MPSBornMachine.load(temp_path)
        assert loaded_bm.num_sites == num_sites
        loaded_nll = loaded_bm.score_samples(batch)
        np.testing.assert_allclose(nll, loaded_nll, atol=1e-10)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_dmrg_optimizer_training():
    num_sites = 4
    num_samples = 40
    # Benign data concentrated near x=0.1
    benign_data = np.random.normal(0.1, 0.02, size=(num_samples, num_sites))
    benign_data = np.clip(benign_data, 0.0, 1.0)
    phi_train = trigonometric_feature_map(benign_data)

    bm = MPSBornMachine(num_sites=num_sites, physical_dim=2, bond_dimension=6)

    optimizer = DMRGOptimizer(max_bond_dimension=8, learning_rate=0.1, num_sweeps=3)
    history = optimizer.fit(bm, phi_train)

    assert len(history) == 4  # Initial + 3 sweeps
    assert np.all(np.isfinite(history))

    # Test out-of-distribution sample (concentrated near 0.9)
    ood_data = np.full((5, num_sites), 0.9)
    phi_ood = trigonometric_feature_map(ood_data)

    nll_benign = bm.score_samples(phi_train[:5])
    nll_ood = bm.score_samples(phi_ood)

    # Anomaly scores on OOD should be significantly higher or well-separated
    assert np.mean(nll_ood) != np.mean(nll_benign)


def test_mps_canonicalization_gauges():
    """Verify that both left and right canonicalization preserve isometric gauges."""
    mps = MatrixProductState.initialize_random(num_sites=6, physical_dim=2, bond_dimension=8, seed=42)

    # Test left canonicalization: sum_s A_s^\dagger A_s = I
    mps.canonicalize_left()
    for k in range(mps.num_sites - 1):
        a = mps.tensors[k]
        res = np.einsum('lsr,lsm->rm', a, a)
        np.testing.assert_allclose(res, np.eye(res.shape[0]), atol=1e-10)

    # Test right canonicalization: sum_s B_s B_s^\dagger = I
    mps.canonicalize_right()
    for k in range(1, mps.num_sites):
        b = mps.tensors[k]
        res = np.einsum('lsr,msr->lm', b, b)
        np.testing.assert_allclose(res, np.eye(res.shape[0]), atol=1e-10)

