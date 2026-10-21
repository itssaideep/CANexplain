# Baseline Audit & Architectural Inspection Report

**Project:** CANExplain-QI (Quantum-Inspired Automotive CAN-Bus Intrusion Detection & Forensics)  
**Document:** `docs/BASELINE_AUDIT.md`  
**Date:** 2026-10-21  
**Author:** Senior Research Software Engineer & Applied ML Researcher  
**Status:** Completed Baseline Audit (Stage 1)

---

## Executive Summary

This audit establishes the empirical, algorithmic, and software-engineering baseline of the `CANExplain-QI` repository prior to architectural refactoring. While the repository contains functional core tensor network algorithms (MPS Born machine, vectorized BLAS sample contractions, Reduced Density Matrix extraction, and von Neumann entropy estimation), systematic inspection revealed critical scientific discrepancies, methodological data leakage, label inflation, and heuristic approximations masquerading as formal algorithms.

The audit findings are structured across six technical dimensions, classified by scientific severity:
1. **The 10 vs. 11 Feature Mismatch** (*Severity: HIGH*)
2. **Window Feature Averaging & Temporal Information Destruction** (*Severity: HIGH*)
3. **Data Splitting, Inter-Window Leakage & "Any-Frame" Label Inflation** (*Severity: CRITICAL*)
4. **DMRG Optimizer Mathematical Truthfulness** (*Severity: CRITICAL*)
5. **Reduced Density Matrices & Mutual Information Approximations** (*Severity: MEDIUM*)
6. **Supervised Baseline Test-Set Contamination** (*Severity: CRITICAL*)

---

## 1. Repository Structure & End-to-End Data Flow

The repository implements an anomaly detection and forensic attribution pipeline structured as follows:

```
Raw CAN Log (.csv / text)
       │
       ▼
[canexplain.data.loaders.load_can_csv]
       │ Standardizes timestamp (epoch base), parses hex IDs, pads payload to 8 bytes
       ▼
[canexplain.preprocessing.features.extract_10_features]
       │ Computes timestamp_offset, delta_t, arbitration_id, byte_0..byte_7
       ▼
[canexplain.preprocessing.normalization.CANFeatureScaler]
       │ Per-feature MinMax scaling to [0, 1] fitted on training data
       ▼
[canexplain.preprocessing.windows.construct_sliding_windows]
       │ Slices (T, F) into sliding windows (N_win, W, F); label = 1 if ANY frame is 1
       ▼
[Feature Pooling: np.mean(win, axis=1)]  <-- CRITICAL: Temporal destruction
       │ Collapses (N_win, W, F) -> (N_win, F)
       ▼
[canexplain.preprocessing.encoding.trigonometric_feature_map]
       │ Maps x_k in [0, 1] to qubit state |phi(x_k)> = [cos(pi/2 * x_k), sin(pi/2 * x_k)]^T
       ▼
[canexplain.models.mps.born_machine.MPSBornMachine]
       │ Computes P(v) = |<Psi|Phi(v)>|^2 / Z via vectorized BLAS contraction
       │ Evaluates Negative Log-Likelihood: NLL(v) = -2 ln |<Psi|Phi(v)>| + ln Z
       ▼
[canexplain.models.threshold.ThresholdCalibrator]
       │ Calibrates tau = Percentile_95(NLL_val) on benign validation split
       ▼
[canexplain.evaluation.metrics.compute_ids_metrics]
       │ Evaluates TP, FP, TN, FN, TPR, FPR, F1 (weighted & binary), PR-AUC, ROC-AUC
       ▼
[canexplain.explainability / forensics.report]
       │ Extracts single-site RDMs (rho_i), von Neumann entropy S_i = -Tr(rho_i log2 rho_i)
       │ Generates structured ISO/SAE 21434 Clause 13 YAML incident report
```

---

## 2. Detailed Technical Findings

### Finding 1: The 10-Feature vs. 11-Feature Mismatch
* **Severity**: HIGH
* **Evidence Path**: `canexplain/preprocessing/features.py` (lines 41–73), `canexplain/cli.py` (line 240), `.planning/REQUIREMENTS.md` (line 10).
* **Technical Issue**:
  The function `extract_10_features` and the project documentation repeatedly claim a **10-feature representation** grounded in `can-sleuth` (Kidmose et al. 2025). However, the code actually constructs **11 columns**:
  1. `timestamp_offset`: Continuous seconds elapsed from start of trace ($t - t_0$).
  2. `delta_t`: Inter-arrival time for the same arbitration ID.
  3. `arbitration_id`: Numeric arbitration ID.
  4–11. `byte_0` through `byte_7`: 8 subdivided raw payload bytes.
* **Scientific Ramifications**:
  `timestamp_offset` is strictly monotonic non-decreasing ($t_0 \to \infty$). Fitting a `MinMaxScaler` on a training slice (e.g. 0 to 60 seconds) means any evaluation trace longer than 60 seconds yields `timestamp_offset > 1.0`. When passed to `trigonometric_feature_map(x)`, $\cos(\frac{\pi}{2} x)$ oscillates periodically, producing unphysical quantum state vectors.
  In true automotive literature (`can-sleuth`), the 10 stationary features are:
  $$\mathbf{x} = [\Delta t, \text{ID}, b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7] \in \mathbb{R}^{10}$$
* **Resolution**:
  Standardize the primary canonical feature set to **exactly 10 stationary features** $[\Delta t, \text{ID}, b_0, \dots, b_7]$. Retain an optional, explicitly documented 11-feature variant $[t_{\text{rel}}, \Delta t, \text{ID}, b_0, \dots, b_7]$ only where bounded window-relative timestamps are strictly desired.

---

### Finding 2: Window Feature Averaging & Temporal Destruction
* **Severity**: HIGH
* **Evidence Path**: `scripts/run_study.py` (lines 74–76, 244–246), `canexplain/cli.py` (lines 73, 116).
* **Technical Issue**:
  After constructing sliding temporal windows `win` of shape $(N_{\text{windows}}, W, F)$ (e.g. $W=32$ frames), the code executes:
  ```python
  phi_train = trigonometric_feature_map(np.mean(win_train, axis=1))
  ```
* **Scientific Ramifications**:
  1. Averaging over the temporal dimension $W$ completely discards frame sequencing. A sequence $[f_1, f_2, f_3, f_4]$ has the exact same representation as $[f_4, f_3, f_2, f_1]$ or any permutation.
  2. Single-frame injection attacks (e.g., occasional spoofed wheel speed) are diluted by a factor of $W$. In a window of $W=32$, a single anomalous frame accounts for only $3.1\%$ of the averaged signal, effectively masking the attack.
* **Resolution**:
  1. Formally designate the current mean-pooling approach as the **Pooled-Window Baseline (`MeanPooledFeatureMap`)**.
  2. Implement an explicit **Sequence-Preserving Representation** where the temporal structure is maintained (e.g., flattening $W$ frames into an ordered MPS chain of sites $k = 1 \dots (W \times F)$ or time-step sites).

---

### Finding 3: "Any-Frame" Label Inflation & Window Contamination
* **Severity**: CRITICAL
* **Evidence Path**: `canexplain/preprocessing/windows.py` (line 65), `scripts/run_study.py`.
* **Technical Issue**:
  Line 65 of `windows.py` sets:
  ```python
  window_labels.append(1 if np.any(sub_labels == 1) else 0)
  ```
* **Empirical Discovery**:
  When tested on real automotive traffic from `can-train-and-test-v1.5`:
  - In `standstill-3.csv`: Raw attack frames constitute only **4.64%** (232 out of 5,000 frames) because standstill spoofing injects frames intermittently.
  - Slicing into sliding windows of $W=32$ with stride $S=16$ resulted in **87.5% of windows** (272 out of 311) containing at least one attack frame!
  - Consequently, the test slice had only **39 benign windows**. Any false alarm by a model (e.g. Isolation Forest) was divided by 39, producing an apparent False Positive Rate of $94.4\%$.
* **Scientific Ramifications**:
  1. "Any-frame" positive labeling artificially inflates attack prevalence by over $18\times$ in intermittent attacks.
  2. Models are penalized as "false alarms" on windows where 31 out of 32 frames are completely benign.
* **Resolution**:
  1. Report evaluation at the **exact unit of evaluation**: frame-level vs. window-level.
  2. Track `attack_frame_ratio` per window ($r_{\text{att}} = \frac{1}{W} \sum \mathbf{1}_{\text{att}}$) and support configurable anomaly thresholds (e.g., window positive if $r_{\text{att}} \ge \theta_{\text{ratio}}$).
  3. Ensure non-overlapping separation between train, validation, and test segments with buffer gaps $\ge W$ to prevent adjacent window frame contamination.

---

### Finding 4: Optimizer Audit — Heuristic Correlation Update vs. Standard DMRG
* **Severity**: CRITICAL
* **Evidence Path**: `canexplain/models/mps/dmrg.py` (lines 70–96).
* **Technical Issue**:
  The module claims to implement "Two-site DMRG-inspired variational optimizer". In physics and quantum chemistry, the Density Matrix Renormalization Group (DMRG) projects the Hamiltonian into the block environment basis and solves a local generalized eigenvalue problem $H_{\text{eff}} B = E B$.
  In generative unsupervised tensor networks (Han et al. 2018), minimizing Negative Log-Likelihood requires contracting the sample environment tensors $E_L(x)$ and $E_R(x)$ to compute exact gradients $\nabla_B \text{NLL}$.
  In `canexplain/models/mps/dmrg.py`:
  ```python
  phi_pair = np.einsum("bi,bj->bij", phi_1, phi_2)
  data_corr = np.mean(phi_pair, axis=0)  # (p1, p2)
  grad_step = np.zeros_like(b_tensor)
  scale = self.learning_rate / (d_l * d_r)
  grad_step[:, :, :, :] = scale * data_corr[None, :, :, None]
  b_updated = b_tensor + grad_step
  ```
  The code simply calculates the outer product of adjacent data features and broadcasts it uniformly across all virtual bonds. It does **not** evaluate sample environment contractions, does **not** divide by sample amplitudes $\langle \Psi | \Phi(x) \rangle$, and does **not** evaluate the partition function gradient $\nabla_B \ln \mathcal{Z}$.
* **Scientific Ramifications**:
  Claiming this is "standard DMRG" is scientifically indefensible. In practice, the empirical NLL fluctuates or increases slightly across sweeps rather than monotonically descending.
* **Resolution**:
  1. Formally document and rename the class as `HeuristicCorrelationOptimizer` (DMRG-inspired heuristic).
  2. Implement a true, mathematically verifiable unsupervised training objective (e.g., exact analytical gradient descent or cross-entropy minimization on synthetic targets).
  3. Validate convergence against a known synthetic reference state.

---

### Finding 5: Approximated Non-Adjacent Two-Site RDMs & Mutual Information
* **Severity**: MEDIUM
* **Evidence Path**: `canexplain/explainability/rdm.py` (lines 104–108).
* **Technical Issue**:
  In `RDMExtractor.get_two_site_rdm(site_i, site_j)`:
  - For adjacent sites ($j = i + 1$), the exact two-site tensor contraction is executed.
  - For non-adjacent sites ($|i - j| > 1$), line 108 executes:
    ```python
    return np.kron(rho_i, rho_j)
    ```
* **Scientific Ramifications**:
  Because $\rho_{i, j}$ is approximated by the tensor product $\rho_i \otimes \rho_j$, the joint entropy $S(\rho_{i, j})$ identically equals $S(\rho_i) + S(\rho_j)$.
  Therefore, the pairwise quantum mutual information:
  $$I(i; j) = S_i + S_j - S_{ij} \equiv 0$$
  for **every single non-adjacent feature pair**! The mutual information matrix was artificially constrained to be tridiagonal.
* **Resolution**:
  Implement exact transfer matrix contraction across the intervening sites $k \in \{i+1, \dots, j-1\}$ to extract the true non-local Reduced Density Matrix $\rho_{i, j}$ across arbitrary distances.

---

### Finding 6: Supervised Baseline Test-Set Contamination
* **Severity**: CRITICAL
* **Evidence Path**: `scripts/run_study.py` (lines 278–279).
* **Technical Issue**:
  In `run_real_dataset_multitrial`:
  ```python
  # 3. MLP Reference (Supervised Oracle)
  mlp_x = np.concatenate([win_tr, win_te[:10]], axis=0)
  mlp_y = np.concatenate([np.zeros(len(win_tr)), y_win_te[:10]], axis=0)
  mlp = MLPDetector(...).fit(mlp_x, mlp_y)
  ```
  The supervised MLP baseline was provided 10 windows directly from the test set (`win_te[:10]`) during training.
* **Scientific Ramifications**:
  This violates fundamental experimental ethics (test set leakage). Even though labeled as an "oracle", feeding test slices into a model invalidates the comparative benchmark.
* **Resolution**:
  Eliminate all test-set concatenation. Train supervised references strictly on designated training/validation splits, or use synthetic injection frames generated exclusively from training data.

---

## 3. Summary of Baseline Tests & Test Status

* **Pytest Test Suite**: 29 passed in 4.39s (81% code coverage).
  - Baselines: 4 tests passing (`test_baselines.py`).
  - CLI: 3 tests passing (`test_cli.py`).
  - Data: 5 tests passing (`test_data.py`).
  - Evaluation: 2 tests passing (`test_evaluation.py`).
  - Explainability: 4 tests passing (`test_explainability.py`).
  - Forensics: 2 tests passing (`test_forensics.py`).
  - MPS Core: 5 tests passing (`test_mps.py`).
  - Preprocessing: 4 tests passing (`test_preprocessing.py`).
* **Linter (Ruff)**: 263 formatting/import/line-length warnings identified.
* **Missing Components**:
  - No sequence-preserving MPS encoding tests.
  - No exact transfer-matrix non-adjacent RDM tests.
  - No multi-dataset adapters (ROAD, HCRL, OTIDS).
  - No perturbation or explanation fidelity validation tests.

---

## 4. Stage-by-Stage Implementation Plan

Based on these audit findings, the repository will be refactored across the following sequential stages:

```
[Stage 2: Data Model & Schema]
  ├── Canonical CANFrame dataclass (standardized metadata, timestamps, 8 bytes, DLC)
  └── Standardized 10-feature stationary extractor (remove monotonic elapsed time)

[Stage 3: Dataset Adapters & Manifest]
  ├── Multi-dataset adapters: can-train-and-test-v1.5, ROAD, HCRL Car-Hacking, OTIDS
  ├── Offline synthetic fixtures for CI testing
  └── docs/DATASETS.md

[Stage 4: MPS Numerical & Objective Validation]
  ├── Exact analytical NLL / variational training objective
  ├── Rename heuristic optimizer transparently
  ├── Full non-adjacent RDM transfer-matrix contraction
  └── Sequence-preserving vs. Pooled representation configuration

[Stage 5: Interpretability & Forensic Integrity]
  ├── Mathematical PSD / Hermitian / Trace normalization verification
  ├── Canonical SHA-256 evidence integrity hashing
  └── Disclaimer: research prototype, not certified legal admissibility

[Stage 6: Leakage-Safe Cross-Dataset Experiments]
  ├── Separate trace-based splits with buffer gaps
  ├── Binary attack-class metrics (F1, Recall, Precision, FPR, PR-AUC, confusion counts)
  ├── Multi-seed reproducibility runs
  └── docs/EXPERIMENT_PROTOCOL.md

[Stage 7: Explanation Quality & Perturbation Validation]
  ├── Feature perturbation tests (fidelity & sensitivity)
  └── Comparative attribution against baseline feature importances

[Stage 8: Software Quality, Ruff & Type Hints]
  ├── Clean Ruff warnings
  └── Type hint enforcement

[Stage 9: Documentation & Thesis Alignment]
  ├── Rewrite README.md with verifiable claims
  └── docs/MODEL_AND_EXPLANATIONS.md & docs/RESULTS_AND_LIMITATIONS.md
```

All implementations will be executed progressively with test validation at each step.
