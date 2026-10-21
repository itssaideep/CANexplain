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
    mps = MatrixProductState.initialize_random(
        num_sites=6, physical_dim=2, bond_dimension=8, seed=42
    )

    # Test left canonicalization: sum_s A_s^\dagger A_s = I
    mps.canonicalize_left()
    for k in range(mps.num_sites - 1):
        a = mps.tensors[k]
        res = np.einsum("lsr,lsm->rm", a, a)
        np.testing.assert_allclose(res, np.eye(res.shape[0]), atol=1e-10)

    # Test right canonicalization: sum_s B_s B_s^\dagger = I
    mps.canonicalize_right()
    for k in range(1, mps.num_sites):
        b = mps.tensors[k]
        res = np.einsum("lsr,msr->lm", b, b)
        np.testing.assert_allclose(res, np.eye(res.shape[0]), atol=1e-10)


def test_exact_analytical_product_state():
    """Analytically verifies MPS on a known N=3 product state |000>."""
    # Construct exact |000> state: site 0, 1, 2 have state |0> = [1.0, 0.0]
    t1 = np.array([[[1.0], [0.0]]], dtype=np.float64)  # shape (1, 2, 1)
    t2 = np.array([[[1.0], [0.0]]], dtype=np.float64)  # shape (1, 2, 1)
    t3 = np.array([[[1.0], [0.0]]], dtype=np.float64)  # shape (1, 2, 1)

    mps = MatrixProductState([t1, t2, t3])
    bm = MPSBornMachine(mps=mps)

    # Norm must be exactly 1.0
    assert mps.norm_squared() == pytest.approx(1.0, abs=1e-12)
    assert bm.partition_function == pytest.approx(1.0, abs=1e-12)

    # Sample [0, 0, 0] -> amplitude = 1.0, P = 1.0, NLL = 0.0
    phi_zero = trigonometric_feature_map(np.array([0.0, 0.0, 0.0]))
    assert mps.contract_sample(phi_zero) == pytest.approx(1.0, abs=1e-12)
    assert bm.probability(phi_zero) == pytest.approx(1.0, abs=1e-12)
    assert bm.score_samples(phi_zero[None, :, :])[0] == pytest.approx(0.0, abs=1e-12)

    # Sample [1, 1, 1] -> amplitude = 0.0, P = 0.0 (clipped to floor)
    phi_one = trigonometric_feature_map(np.array([1.0, 1.0, 1.0]))
    assert mps.contract_sample(phi_one) == pytest.approx(0.0, abs=1e-12)
    assert bm.probability(phi_one) == pytest.approx(1e-15, abs=1e-16)


def test_exact_analytical_entangled_state_and_non_adjacent_rdm():
    """Analytically tests exact non-adjacent RDM and mutual information on GHZ state:
    |Psi> = 1/sqrt(2) (|000> + |111>).
    """
    from canexplain.explainability.rdm import RDMExtractor
    from canexplain.explainability.entropy import von_neumann_entropy
    from canexplain.explainability.mutual_info import compute_mutual_information_matrix

    # Construct exact 3-site GHZ state in MPS form (bond dimension D=2)
    # Site 0: shape (1, 2, 2)
    t0 = np.zeros((1, 2, 2), dtype=np.float64)
    t0[0, 0, 0] = 1.0 / np.sqrt(2.0)
    t0[0, 1, 1] = 1.0 / np.sqrt(2.0)

    # Site 1: shape (2, 2, 2)
    t1 = np.zeros((2, 2, 2), dtype=np.float64)
    t1[0, 0, 0] = 1.0
    t1[1, 1, 1] = 1.0

    # Site 2: shape (2, 2, 1)
    t2 = np.zeros((2, 2, 1), dtype=np.float64)
    t2[0, 0, 0] = 1.0
    t2[1, 1, 0] = 1.0

    mps = MatrixProductState([t0, t1, t2])
    assert mps.norm_squared() == pytest.approx(1.0, abs=1e-12)

    extractor = RDMExtractor(mps)

    # Single-site RDM at site 0: rho_0 = 1/2 I_2 -> entropy = 1.0 bit
    rho_0 = extractor.get_single_site_rdm(0)
    assert rho_0.shape == (2, 2)
    np.testing.assert_allclose(rho_0, 0.5 * np.eye(2), atol=1e-12)
    assert von_neumann_entropy(rho_0) == pytest.approx(1.0, abs=1e-10)

    # Non-adjacent two-site RDM between site 0 and site 2 (intervening site 1 is traced out!)
    rho_02 = extractor.get_two_site_rdm(0, 2)
    assert rho_02.shape == (4, 4)

    # Tracing out site 1 yields mixed state: 0.5 |00><00| + 0.5 |11><11|
    expected_rho_02 = np.zeros((4, 4), dtype=np.float64)
    expected_rho_02[0, 0] = 0.5  # |00><00|
    expected_rho_02[3, 3] = 0.5  # |11><11|
    np.testing.assert_allclose(rho_02, expected_rho_02, atol=1e-12)

    # S(rho_02) must be exactly 1.0 bit
    s_02 = von_neumann_entropy(rho_02)
    assert s_02 == pytest.approx(1.0, abs=1e-10)

    # True non-adjacent quantum mutual information:
    # I(0; 2) = S(rho_0) + S(rho_2) - S(rho_02) = 1.0 + 1.0 - 1.0 = 1.0 bit!
    mi_mat = compute_mutual_information_matrix(extractor)
    assert mi_mat[0, 2] == pytest.approx(1.0, abs=1e-10)
    assert mi_mat[2, 0] == pytest.approx(1.0, abs=1e-10)


def test_gradient_born_machine_optimizer():
    """Verifies GradientBornMachineOptimizer execution and loss tracking."""
    from canexplain.models.mps.dmrg import GradientBornMachineOptimizer

    num_sites = 4
    bm = MPSBornMachine(num_sites=num_sites, physical_dim=2, bond_dimension=4, seed=42)

    data = np.clip(np.random.normal(0.2, 0.05, size=(30, num_sites)), 0.0, 1.0)
    phi_train = trigonometric_feature_map(data)

    opt = GradientBornMachineOptimizer(learning_rate=0.02, num_epochs=2, batch_size=15)
    loss_hist = opt.fit(bm, phi_train)

    assert len(loss_hist) == 3  # Initial + 2 epochs
    assert np.all(np.isfinite(loss_hist))


