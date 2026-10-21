# CANExplain-QI

**Interpretable Automotive CAN-Bus Intrusion Detection and Forensic Analysis Using Quantum-Inspired Matrix Product States (MPS)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Test Suite: Pytest](https://img.shields.io/badge/tests-51%20passed%20(81%25%20cov)-brightgreen.svg)](https://github.com/itssaideep/CANexplain)
[![Status: Academic Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](docs/BASELINE_AUDIT.md)

---


## 1. Primary Research Question & Scope

### 1.1 Core Research Question
> **Can a Matrix Product State (MPS) generative anomaly detector provide useful, reproducible, and computationally practical explanations of CAN-bus intrusions, including when evaluated across different datasets or vehicle platforms?**

### 1.2 Scope and Context
Modern road vehicles integrate dozens of Electronic Control Units (ECUs) interconnected over the broadcast **Controller Area Network (CAN)** bus (ISO 11898). Because CAN lacks message authentication, encryption, and sender identity validation, in-vehicle networks are vulnerable to message injection, masquerade, and timing attacks via OBD-II ports, telematics gateways, or compromised ECUs.

While deep neural networks and tree ensembles detect anomalies, their black-box nature necessitates post-hoc approximation algorithms (such as SHAP or LIME) that:
1. Require thousands of artificial perturbations per window, inducing substantial computational latency;
2. Produce local linear surrogates that may not be faithful to non-linear model decision boundaries;
3. Lack direct mathematical provenance back to physical bus signals.

**CANExplain-QI** addresses this gap by utilizing **Matrix Product States (MPS)**—a 1D tensor network factorization from quantum many-body physics—as a **Born machine generative anomaly detector**. By operating directly on exact analytical contractions, CANExplain-QI extracts **Reduced Density Matrices (RDMs)**, **von Neumann entanglement entropy**, and **quantum mutual information** in microseconds without post-hoc perturbations.

---

## 2. System Architecture & Data Flow

```
                      +--------------------------------------------------------+
                      |                 Raw CAN Traffic Source                 |
                      |   [Timestamp, Arbitration ID, DLC, Payload Bytes 0..7] |
                      +---------------------------+----------------------------+
                                                  |
                                                  v
                      +--------------------------------------------------------+
                      |         Modular Dataset Adapters & Schema Parser       |
                      |    (can-train-and-test, ROAD, HCRL Car-Hacking, OTIDS) |
                      +---------------------------+----------------------------+
                                                  |
                                                  v
                      +--------------------------------------------------------+
                      |         Stationary 10-Feature Vectorization            |
                      |    [Delta_t, Arbitration_ID, Byte_0, ..., Byte_7]      |
                      +---------------------------+----------------------------+
                                                  |
                                                  v
                      +--------------------------------------------------------+
                      |         Zero-Leakage Robust Feature Scaling            |
                      |    Fitted on Benign Training Split Only -> x_k in [0,1]|
                      +---------------------------+----------------------------+
                                                  |
                                                  v
                      +--------------------------------------------------------+
                      |           Quantum Trigonometric Feature Map            |
                      |    |phi(x_k)> = [cos(pi/2 * x_k), sin(pi/2 * x_k)]^T   |
                      +---------------------------+----------------------------+
                                                  |
                                                  v
                      +--------------------------------------------------------+
                      |              MPS Born Machine Probability Core         |
                      |       |Psi> = sum A^(1) A^(2) ... A^(10) |s_1 ... s_10>|
                      |               P(v) = |<Psi | Phi(v)>|^2 / Z            |
                      +---------------------------+----------------------------+
                                                  |
                               +------------------+------------------+
                               |                                     |
                               v                                     v
                +-----------------------------+       +-----------------------------+
                |   Generative NLL Detector   |       | Intrinsic Quantum Explainer |
                |  Score S(v) = -ln P(v) > tau|       | * Single-Site RDMs (rho_k)  |
                |  Latency: 10 - 12 us/window |       | * von Neumann Entropy S_k   |
                |  Zero-Leakage 95th Calib.   |       | * Exact Non-Adjacent I(i;j) |
                +--------------+--------------+       | * Feature Marginal P(x_k)   |
                               |                      +--------------+--------------+
                               +------------------+------------------+
                                                  |
                                                  v
                      +--------------------------------------------------------+
                      |      Forensic Incident Report & Evidence Integrity     |
                      | * Window Provenance & Root-Cause Feature Attribution   |
                      | * Deterministic SHA-256 Hash of Canonical Evidence     |
                      | * ISO/SAE 21434 Clause 13 Monitoring Alignment         |
                      +--------------------------------------------------------+
```

---

## 3. Feature Schema & Representation Models

### 3.1 The 10-Feature Stationary Specification
The codebase resolves the historical 10 vs. 11 feature discrepancy by establishing the canonical 10-feature stationary specification:
1. `delta_t`: Inter-arrival interval between consecutive frames with identical arbitration ID (seconds).
2. `arbitration_id`: Numeric CAN message identifier (11-bit standard or 29-bit extended).
3. `byte_0` to `byte_7`: Subdivided 8 individual payload bytes (normalized $[0.0, 1.0]$).

> **Why Monotonic Timestamp Offset Was Removed:** Monotonic timestamps $t - t_0$ grow linearly across time. When projected through the trigonometric feature map, out-of-distribution test frames suffer artificial high-frequency quantum phase rotation oscillations, severely distorting anomaly scoring. The 10 stationary features ensure temporal stationarity. (A legacy 11-feature mode is retained for backward compatibility via `include_timestamp_offset=True`).

### 3.2 Sequence-Preserving vs. Pooled-Window Baseline
- **Mean-Pooled Feature Baseline (Default):** Sliding windows of size $W$ are averaged across the temporal axis to produce a 10-dimensional stationary vector $\bar{\mathbf{x}} \in [0, 1]^{10}$.
- **Sequence-Preserving Representation:** Concatenates all $W \times F$ scalar features into an ordered temporal sequence without collapsing time, enabling temporal ordering analysis via high-order tensor networks.

---

## 4. Multi-Dataset Support & Acquisition

`CANExplain-QI` provides native, schema-validated adapters for four major automotive CAN intrusion datasets:

| Dataset | Vehicle Platform | Format / Attacks | Access / License | Adapter Class |
| :--- | :--- | :--- | :--- | :--- |
| **can-train-and-test-v1.5** | 2011 Chevrolet Impala | On-road CSV; DoS, Standstill, Masquerade | Public / Open Research | `SharedCANAdapter` |
| **ROAD CAN IDS Dataset** | 2017 KIA Soul | Real highway CSV; Correlated, Max-Freq | Open Access (Zenodo 10462796) | `ROADDatasetAdapter` |
| **HCRL Car-Hacking** | Hyundai YF Sonata | Real vehicle CSV; DoS, Fuzzy, RPM, Gear | Academic Non-Commercial (Attribution) | `HCRLCarHackingAdapter` |
| **OTIDS Intrusion Dataset**| KIA Soul | Bus logging CSV; DoS, Fuzzy, Impersonation| Academic Non-Commercial | `OTIDSAdapter` |

For full licensing details, citation guidelines, and manual download instructions, refer to [docs/DATASETS.md](docs/DATASETS.md).

---

## 5. Empirical Benchmark Results

All benchmarks were evaluated over **$K = 3$ independent trials** with seeds `[42, 101, 202]` using window size $W=32$ and stride $S=16$. All metrics are recorded directly from `experiment_results.json`.

### 5.1 Within-Dataset Multi-Trial Benchmark

| Scenario | Model Architecture | Attack $F1_{\text{binary}}$ | Attack Recall | Attack Precision | False Positive Rate (FPR) | True Negative Rate (TNR) | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DoS Flooding** | Statistical Baseline | 0.2105 ± 0.0000 | 0.1667 ± 0.0000 | 0.2857 ± 0.0000 | 0.1351 ± 0.0000 | 0.8649 ± 0.0000 | 0.6633 ± 0.0000 |
| | Isolation Forest | 0.7459 ± 0.0543 | 0.9167 ± 0.1443 | 0.6402 ± 0.0587 | 0.1622 ± 0.0468 | 0.8378 ± 0.0468 | 0.8804 ± 0.0336 |
| | MLP Reference | 0.2105 ± 0.0000 | 0.1667 ± 0.0000 | 0.2857 ± 0.0000 | 0.1261 ± 0.0000 | 0.8739 ± 0.0000 | 0.6633 ± 0.0000 |
| | **MPS Born Machine** | 0.0476 ± 0.0000 | 0.0833 ± 0.0000 | 0.1250 ± 0.0000 | **0.0631 ± 0.0154** | **0.9369 ± 0.0154** | 0.7024 ± 0.0112 |
| **Fuzzing Attack**| Statistical Baseline | 0.4211 ± 0.0000 | 0.3333 ± 0.0000 | 0.5714 ± 0.0000 | 0.0811 ± 0.0000 | 0.9189 ± 0.0000 | 0.7601 ± 0.0000 |
| | Isolation Forest | 0.7839 ± 0.0673 | 1.0000 ± 0.0000 | 0.6488 ± 0.0766 | 0.1802 ± 0.0620 | 0.8198 ± 0.0620 | 0.8986 ± 0.0392 |
| | MLP Reference | 0.8504 ± 0.0412 | 1.0000 ± 0.0000 | 0.7619 ± 0.0480 | 0.1261 ± 0.0267 | 0.8739 ± 0.0267 | 0.9354 ± 0.0204 |
| | **MPS Born Machine** | 0.2097 ± 0.0298 | 0.2500 ± 0.0000 | 0.3333 ± 0.0577 | **0.0450 ± 0.0154** | **0.9550 ± 0.0154** | 0.7618 ± 0.0185 |
| **Payload Spoof** | Statistical Baseline | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1081 ± 0.0000 | 0.8919 ± 0.0000 | 0.6698 ± 0.0000 |
| | Isolation Forest | 0.0351 ± 0.0608 | 0.0278 ± 0.0481 | 0.0476 ± 0.0825 | 0.1441 ± 0.0267 | 0.8559 ± 0.0267 | 0.6672 ± 0.0270 |
| | MLP Reference | 0.2105 ± 0.0000 | 0.1667 ± 0.0000 | 0.2857 ± 0.0000 | 0.1261 ± 0.0000 | 0.8739 ± 0.0000 | 0.6633 ± 0.0000 |
| | **MPS Born Machine** | **0.2667 ± 0.0000** | **0.2500 ± 0.0000** | **0.2857 ± 0.0000** | **0.0541 ± 0.0000** | **0.9459 ± 0.0000** | **0.7812 ± 0.0000** |

### 5.2 Key Takeaways
1. **Payload Multi-Byte Correlation Superiority:** On subtle Payload Spoofing attacks (where timing and IDs remain normal), timing detectors fail ($F1_{\text{binary}} = 0.000$), while MPS Born Machine achieves the highest attack detection ($F1_{\text{binary}} = 0.2667$) with low false alarm rate ($5.4\%$).
2. **Class Imbalance Inflation Warning:** While models report high **Weighted F1** (66% - 93%) due to overwhelming normal traffic, reporting the uninflated **Binary Attack F1** reveals true operational performance.
3. **Ultra-Low Contraction Latency:** MPS sample contraction executes in **10.1 to 12.0 microseconds** per window, operating at $>80,000$ windows per second on CPU.

---

## 6. Advanced Architectures: TTN & OpenDBC Semantic Tokenization

To address the limitations identified in earlier experiments, two advanced modules are integrated into `CANExplain-QI`:

### 6.1 Hierarchical Binary Tree Tensor Networks (`canexplain.models.ttn`)
- **Motivation:** Resolves the $O(N)$ linear path length decay of 1D Matrix Product States on long sequence windows.
- **Topology:** Balanced binary tree with $L = 2^K$ leaf sites (e.g. $L=16, 32, 64$).
- **Logarithmic Contraction Depth:** The maximum virtual bond distance between any two sequence sites is $2 \log_2(L) = O(\log L)$, mitigating gradient vanishing across extended temporal frames.
- **Components:** `BinaryTreeTensorNetwork` core, `TTNBornMachine` generative scorer, and `TTNCorrelationOptimizer` hierarchical layer-by-layer sweep optimizer.

### 6.2 OpenDBC Semantic Signal Tokenization (`canexplain.data.dbc`)
- **Motivation:** Resolves OEM-proprietary arbitration ID distribution shifts across vehicle makes.
- **Universal Kinematics:** Decodes OEM bitfields (Hyundai/Kia, GM/Chevrolet) into standardized physical engineering quantities:
  `["vehicle_speed", "steering_angle", "engine_rpm", "throttle_pct", "brake_active", "gear_num"]`.
- **Sample & Hold Normalization:** Normalizes signals into $[0.0, 1.0]$ based on universal physical vehicle limits, allowing a tensor network trained on one vehicle architecture to evaluate another without structural mismatch.

---

## 7. Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/itssaideep/CANexplain.git
cd CANexplain

# Install package in editable mode with development dependencies
pip install -e ".[dev,viz]"
```

### 7.1 Running the Verified Test Suite
```bash
# Execute unit tests with code coverage (51 tests, 81% coverage)
python -m pytest tests/

# Execute strict code quality and formatting checks
python -m ruff check .
```

### 7.2 Executing the Full Reproducible Experiment Suite
```bash
# Runs Experiment A, B, D, E, F and generates experiment_results.json
python scripts/run_experiments.py --trials 3 --output experiment_results.json
```

### 7.3 Command Line Interface (CLI)
```bash
# 1. Audit repository dependencies and data integrity
canexplain audit

# 2. Train an MPS Born Machine on benign CAN traffic
canexplain train --data tests/fixtures/hcrl_sample.csv --bond-dim 16 --output artifacts/mps_model.npz

# 3. Detect intrusions in a target trace
canexplain detect --model artifacts/mps_model.npz --data tests/fixtures/road_sample.csv --threshold 15.0

# 4. Generate forensic incident report with SHA-256 evidence hashing
canexplain explain --model artifacts/mps_model.npz --data tests/fixtures/road_sample.csv --output-dir reports/
```

---

## 8. Forensic Incident Report Example

The following sample report illustrates deterministic provenance tracking and SHA-256 evidence hashing generated by `canexplain explain` and verified by `canexplain.forensics.report`:

```yaml
forensic_incident_report:
  report_id: "ALERT-2026-0042"
  evidence_integrity_sha256: "b40a332a688bcaec6914bbcdb5c87a552bf3f90e9e144a9e88bf0a0300df2d85"
  iso_sae_21434_clause: "ISO/SAE 21434 Clause 13 - Conceptual Alignment (Cybersecurity Monitoring & Response)"
  generated_at: "2026-10-21T21:01:45.312948Z"
  incident_summary:
    severity: "CRITICAL"
    detection_mechanism: "MPS_Born_Machine_NLL_Exceedance"
    anomaly_nll_score: 38.4512
    calibrated_threshold: 14.1205
    confidence_margin: 24.3307
  provenance_trace:
    window_idx: 42
    start_frame_idx: 1344
    end_frame_idx: 1375
    start_timestamp: 1478198374.1205
    end_timestamp: 1478198374.3142
    duration_seconds: 0.1937
    frame_count: 32
    attack_frame_count: 28
    attack_frame_ratio: 0.875
    subclass: "DoS"
    dataset_name: "can-train-and-test-v1.5"
    trace_name: "DoS-3.csv"
  anomalous_feature_attributions:
    - feature_name: "arbitration_id"
      marginal_probability: 0.0001
      entropy: 0.0215
      deviation_factor: 12.4
    - feature_name: "delta_t"
      marginal_probability: 0.0014
      entropy: 0.1042
      deviation_factor: 8.1
  model_metadata:
    model_type: "MPS_Born_Machine"
    num_sites: 10
    bond_dimension: 20
    optimizer: "HeuristicCorrelationOptimizer"
  investigative_guidance:
    - "Inspect physical bus wiring and OBD-II port for aftermarket rogue devices."
    - "Verify whether relevant ECUs logged Diagnostic Trouble Codes (DTCs)."
    - "Review gateway firewall rules and inspect for buffer overflow or bus-off states."
  disclaimer: "RESEARCH PROTOTYPE NOTICE: Statistical anomaly identified via Matrix Product State Born machine generative distribution. Marginal probability collapses indicate dimensions deviating from normal baseline, not confirmed causal attacker intent. Not certified for formal judicial legal admissibility without corroborating physical bus captures."
```

---

## 9. Documentation Sitemap

- [docs/BASELINE_AUDIT.md](docs/BASELINE_AUDIT.md): Comprehensive audit of original codebase, 10 vs 11 feature analysis, leakage evaluation, and optimizer validation.
- [docs/DATASETS.md](docs/DATASETS.md): Acquisition guides, schemas, licensing, and ethical usage rules for supported automotive datasets.
- [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md): Rigorous experimental design, zero-leakage split protocol, and multi-seed replication policy.
- [docs/MODEL_AND_EXPLANATIONS.md](docs/MODEL_AND_EXPLANATIONS.md): Mathematical derivation of Born machines, tensor contractions, exact RDM transfer matrices, and entanglement entropy.
- [docs/RESULTS_AND_LIMITATIONS.md](docs/RESULTS_AND_LIMITATIONS.md): In-depth synthesis of within-dataset and cross-dataset benchmarks, representation ablations, and threats to validity.

---

## 10. Citation

If you use this software or methodology in your academic research, please cite:

```bibtex
@misc{canexplain_qi_2026,
  author = {Sai Deep},
  title = {CANExplain-QI: Interpretable Automotive CAN-Bus Intrusion Detection and Forensic Analysis Using Quantum-Inspired Matrix Product States},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/itssaideep/CANexplain}}
}
```
