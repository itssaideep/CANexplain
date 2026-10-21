# Experimental Results, Performance Benchmarks, and Scientific Limitations

This document presents empirical findings from executing the standardized evaluation protocol (`docs/EXPERIMENT_PROTOCOL.md`) via `scripts/run_experiments.py`, persisted in `experiment_results.json`.

---

## 1. Experimental Environment & Verification Metadata

All benchmarks were executed on the following validated environment:
- **Python Runtime:** Python 3.10.11 (win32, x86-64)
- **Dependencies:** NumPy 1.24+, SciPy 1.10+, scikit-learn 1.3+, PyYAML 6.0+
- **Hardware Platform:** AMD / Intel x86-64, single-threaded CPU execution
- **Evaluation Units:** Window-level evaluation ($W=32, S=16$) for multi-frame attacks; frame-level evaluation for single-frame transfers.
- **Seeds for Multi-Trial Replication:** `[42, 101, 202]` ($K=3$ independent trial runs per model configuration).
- **Leakage Prevention:** Zero test leakage. All feature transformations (`CANFeatureScaler`) and anomaly thresholds (`ThresholdCalibrator`, 95th percentile) are fit strictly on benign training/validation splits.

---

## 2. Experiment A: Within-Dataset Multi-Trial Benchmark

Evaluates four architectures across distinct automotive cyberattack modes:
1. **Statistical Deviation Detector:** Transparent baseline tracking inter-arrival timing ($\Delta t$) and message frequencies.
2. **Isolation Forest (iForest):** Classical tree-based ensemble baseline ($n=50$ estimators).
3. **MLP Reference:** Supervised multi-layer perceptron trained on synthetic distribution anomalies.
4. **MPS Born Machine:** Quantum-inspired tensor network generative anomaly detector ($N=10, d=2, D=20$).

### 2.1 Attack-Class Performance Summary (Mean ± Sample Standard Deviation)

#### A. DoS Flooding (High-Frequency Injection of Dominant 0x000 Arbitration ID)
| Model | F1 (Binary Attack) | Precision (Attack) | Recall / TPR | False Positive Rate (FPR) | True Negative Rate (TNR) | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Statistical Baseline** | 0.2105 ± 0.0000 | 0.2857 ± 0.0000 | 0.1667 ± 0.0000 | 0.1351 ± 0.0000 | 0.8649 ± 0.0000 | 0.6633 ± 0.0000 |
| **Isolation Forest** | 0.7459 ± 0.0543 | 0.6402 ± 0.0587 | 0.9167 ± 0.1443 | 0.1622 ± 0.0468 | 0.8378 ± 0.0468 | 0.8804 ± 0.0336 |
| **MLP Reference** | 0.2105 ± 0.0000 | 0.2857 ± 0.0000 | 0.1667 ± 0.0000 | 0.1261 ± 0.0000 | 0.8739 ± 0.0000 | 0.6633 ± 0.0000 |
| **MPS Born Machine** | 0.0476 ± 0.0000 | 0.1250 ± 0.0000 | 0.0833 ± 0.0000 | **0.0631 ± 0.0154** | **0.9369 ± 0.0154** | 0.7024 ± 0.0112 |

#### B. Fuzzing Attack (Randomized Arbitration IDs and Random Payload Noise)
| Model | F1 (Binary Attack) | Precision (Attack) | Recall / TPR | False Positive Rate (FPR) | True Negative Rate (TNR) | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Statistical Baseline** | 0.4211 ± 0.0000 | 0.5714 ± 0.0000 | 0.3333 ± 0.0000 | 0.0811 ± 0.0000 | 0.9189 ± 0.0000 | 0.7601 ± 0.0000 |
| **Isolation Forest** | 0.7839 ± 0.0673 | 0.6488 ± 0.0766 | 1.0000 ± 0.0000 | 0.1802 ± 0.0620 | 0.8198 ± 0.0620 | 0.8986 ± 0.0392 |
| **MLP Reference** | 0.8504 ± 0.0412 | 0.7619 ± 0.0480 | 1.0000 ± 0.0000 | 0.1261 ± 0.0267 | 0.8739 ± 0.0267 | 0.9354 ± 0.0204 |
| **MPS Born Machine** | 0.2097 ± 0.0298 | 0.3333 ± 0.0577 | 0.2500 ± 0.0000 | **0.0450 ± 0.0154** | **0.9550 ± 0.0154** | 0.7618 ± 0.0185 |

#### C. Payload Sensor Spoofing (Targeted Manipulation of Diagnostic & Sensor Bytes)
| Model | F1 (Binary Attack) | Precision (Attack) | Recall / TPR | False Positive Rate (FPR) | True Negative Rate (TNR) | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Statistical Baseline** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1081 ± 0.0000 | 0.8919 ± 0.0000 | 0.6698 ± 0.0000 |
| **Isolation Forest** | 0.0351 ± 0.0608 | 0.0476 ± 0.0825 | 0.0278 ± 0.0481 | 0.1441 ± 0.0267 | 0.8559 ± 0.0267 | 0.6672 ± 0.0270 |
| **MLP Reference** | 0.2105 ± 0.0000 | 0.2857 ± 0.0000 | 0.1667 ± 0.0000 | 0.1261 ± 0.0000 | 0.8739 ± 0.0000 | 0.6633 ± 0.0000 |
| **MPS Born Machine** | **0.2667 ± 0.0000** | **0.2857 ± 0.0000** | **0.2500 ± 0.0000** | **0.0541 ± 0.0000** | **0.9459 ± 0.0000** | **0.7812 ± 0.0000** |

---

## 3. Critical Analysis of Metrics & Class Imbalance Inflation

### 3.1 The Weighted F1 vs. Attack F1 Discrepancy
In typical automotive traffic captures, benign messages account for 75% to 99% of total frames.
- When computing **weighted F1** or **overall accuracy**, models that perform poorly on the minority attack class still report seemingly impressive scores (e.g. weighted F1 of 0.66 to 0.78 for models with 0% attack recall).
- In `CANExplain-QI`, we mandate reporting **Binary Attack F1 ($F1_{\text{binary}}$)**, **TPR (Recall)**, and **FPR**.
- **Key Insight on Payload Spoofing:** When attacks do not alter frame transmission frequency, timing-based detectors fail completely ($F1_{\text{binary}} = 0.000$). Isolation Forest struggles on subtle multi-byte correlations ($F1_{\text{binary}} = 0.0351$). In contrast, the MPS Born Machine captures the joint multi-byte probability manifold, achieving an attack $F1_{\text{binary}}$ of **0.2667** while maintaining a low false positive rate (**FPR = 5.4%** vs. 14.4% for iForest).

---

## 4. Experiment B: Cross-Dataset Zero-Shot Transfer

Evaluates an MPS Born Machine trained exclusively on synthetic/baseline normal traffic and transferred without calibration on external datasets:

| Target Dataset | Vehicle Platform | Transfer Accuracy | Attack Precision | Attack Recall / TPR | FPR | TNR | PR-AUC | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ROAD Dataset** | KIA Soul | 62.5% | 0.0000 | 0.0000 | **0.0000** | **1.0000** | 0.2758 | 0.0000 |
| **HCRL Car-Hacking** | Hyundai YF Sonata | **66.7%** | **1.0000** | **0.3333** | **0.0000** | **1.0000** | **0.7333** | **0.6667** |

### Takeaways on Cross-Dataset Generalization:
1. **Conservative False Alarm Suppression:** Both transfer runs maintained **0.0% False Positive Rate (FPR = 0.0, TNR = 1.0)**. The detector did not flag legitimate vehicle baseline messages as intrusions.
2. **Platform Specificity:** Because arbitration IDs and signal scaling differ between automotive OEM architectures (e.g. Hyundai vs. KIA), zero-shot transfer without per-platform calibration yields conservative recall (33.3% on HCRL, 0% on ROAD). Zero-shot cross-platform IDS remains an open research frontier.

---

## 5. Experiment D: Representation Ablation (Sequence-Preserving vs. Mean-Pooled)

Ablates temporal representation architectures on identical window segments ($W=32, F=10$):
- **Mean-Pooled Feature Baseline (10 Sites):** Collapses time dimension into window mean $\bar{\mathbf{x}} \in \mathbb{R}^{10}$, contracted via a 10-site MPS Born Machine.
- **Sequence-Preserving Flattened Model (320 Sites):** Preserves ordered temporal structure by concatenating all $W \times F = 320$ scalar features into a 320-site MPS.

| Representation | Sites ($N$) | Bond Dim ($D$) | Attack $F1_{\text{binary}}$ | Recall / TPR | Parameter Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean-Pooled Baseline** | 10 | 16 | **0.7500** | **0.6000** | ~640 parameters |
| **Sequence-Preserving** | 320 | 16 | 0.0000 | 0.0000 | ~20,480 parameters |

### Scientific Explanation for the Representation Gap:
- In a 320-site 1D tensor chain with open boundary conditions, long-range correlations between frame 0 and frame 31 must propagate across up to 320 virtual bonds. With moderate bond dimensions ($D=16$) and gradient-free heuristic updates, gradient vanishing and amplitude underflow occur across the chain.
- Mean-pooling acts as an effective temporal low-pass filter, reducing the problem to 10 stationary physical sites where low bond dimensions ($D \le 24$) adequately capture cross-sensor covariance.
- **Recommendation:** Sequence-preserving tensor network IDS should utilize hierarchical tree tensor networks (TTN) or multiscale entanglement renormalization ansatz (MERA) rather than simple 1D MPS chains.

---

## 6. Experiment E: Bond Dimension Scaling & Computational Complexity

Measures scaling of detection performance, inference latency, and training memory against virtual bond dimension $D \in \{8, 16, 24, 32\}$ on a 10-site MPS Born Machine:

| Bond Dimension ($D$) | Attack $F1_{\text{binary}}$ | Inference Contraction Latency | Training Time | Peak Memory |
| :--- | :--- | :--- | :--- | :--- |
| $D = 8$ | 0.0000 | **10.1 µs / window** | 0.38 s | < 2.0 MB |
| $D = 16$ | **0.5926** | 10.1 µs / window | 0.40 s | < 2.0 MB |
| $D = 24$ | 0.0000 | 11.5 µs / window | 0.44 s | < 2.5 MB |
| $D = 32$ | 0.0000 | 12.0 µs / window | 0.49 s | < 3.0 MB |

### Key Takeaway:
- Contraction latency remains extraordinarily fast (**10.1 to 12.0 microseconds per window**, corresponding to over **80,000 windows per second** on a standard CPU).
- At $D=16$, the bond dimension is optimal for balancing capacity with regularization on this feature dimension.

---

## 7. Remaining Scientific & Engineering Limitations

1. **Heuristic Optimizer vs. Full Variational DMRG:**
   The primary sweep optimizer performs local two-site empirical correlation updates (`HeuristicCorrelationOptimizer`). While computationally efficient, it is not standard Hamiltonian DMRG. The exact gradient optimizer (`GradientBornMachineOptimizer`) is implemented and validated, but requires batch tuning for large datasets.
2. **Arbitration ID Re-identification Across Vehicle Platforms:**
   Automotive CAN standards (ISO 11898) standardize the physical and data link layers, but CAN application-layer message IDs and DBC database encodings are OEM-proprietary. Without vehicle-specific DBC knowledge, cross-vehicle zero-shot transfer will remain fundamentally bounded by distribution shift.
3. **Class Imbalance in CAN IDS Benchmarks:**
   Public CAN datasets frequently present extreme class imbalance. Reporting only overall accuracy or weighted F1 creates an illusion of high performance. Real evaluation must always report binary attack metrics, true negative rates, and false alarm frequencies.
4. **Legal and Regulatory Status:**
   `CANExplain-QI` is an **academic research prototype**. The forensic reports and SHA-256 evidence integrity hashes demonstrate alignment with ISO/SAE 21434 and UNECE R155 monitoring concepts, but do not confer formal statutory certification or judicial admissibility without corroborating physical instrumentation.
