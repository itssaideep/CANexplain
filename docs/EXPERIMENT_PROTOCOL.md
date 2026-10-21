# Scientific Experiment Protocol: Quantum-Inspired CAN Intrusion Detection

**Project:** CANExplain-QI (RISE MAGIC Topic: Interpretable Automotive CAN-Bus Intrusion Detection using Tensor Networks)  
**Document:** `docs/EXPERIMENT_PROTOCOL.md`  
**Date:** 2026-10-21  
**Status:** Protocol Specification for Stage 6 Experiments  

---

## 1. Primary Research Question & Hypotheses

### Primary Research Question
*Can an MPS-based generative anomaly detector provide useful, reproducible, and computationally practical explanations of CAN-bus intrusions, including when evaluated across different datasets or vehicle platforms?*

### Formal Hypotheses
* **$H_1$ (Generative Anomaly Detection Fidelity)**: An unsupervised MPS Born machine trained solely on benign CAN traffic achieves higher attack-class precision and PR-AUC on subtle payload falsification attacks compared to shallow statistical timing baselines, because it models joint multi-feature probability distributions $P(\mathbf{x})$.
* **$H_2$ (Sequence Preservation vs. Pooling)**: A sequence-preserving representation retains temporal burst signatures that are diluted by temporal mean-pooling ($1/W$ dilution), leading to higher recall on low-prevalence intermittent attacks (e.g. standstill spoofing).
* **$H_3$ (Explainability & Entanglement Structure)**: The Reduced Density Matrices (RDMs) and pairwise quantum mutual information $I(i; j)$ extracted from the MPS identify the specific CAN arbitration IDs and payload bytes that collapse in marginal probability during an intrusion, providing actionable forensic attribution without ad-hoc heuristic perturbation.
* **$H_4$ (Cross-Platform Transferability & Domain Shift)**: When evaluated cross-dataset or cross-vehicle without retraining, generative models trained on stationary features $[\Delta t, \text{ID}, b_0..b_7]$ exhibit lower false-alarm escalation than models trained on non-stationary features like monotonic timestamp offsets.

---

## 2. Experimental Datasets & Split Strategy

### Datasets
1. **Primary Automotive Baseline**: `can-train-and-test-v1.5` (Modern passenger car testbed).
   - Training: `accessory-1.csv` (attack-free, benign driving).
   - Test 1: `DoS-3.csv` (aggressive 0x000 message injection).
   - Test 2: `standstill-3.csv` (stealthy sensor/speedometer spoofing, 4.64% attack frames).
2. **Cross-Platform Datasets**:
   - `ROAD`: Kia Soul passenger vehicle (Oak Ridge National Laboratory).
   - `HCRL Car-Hacking`: Hyundai YF Sonata (Korea University).
   - `OTIDS`: KIA Soul (ETRI / HCRL).

### Data Splitting & Leakage Prevention Rules
* **Trace-Separated Splits**: Training, validation, and test sets are drawn from independent trace segments or separate recording files.
* **Zero Train-Test Window Contamination**: Sliding windows are constructed within splits after segment separation. When splitting a single capture, a temporal buffer gap of length $\Delta \ge W$ frames is inserted between splits to ensure no sliding window spans across the boundary.
* **Zero Test Leakage into Scalers**: `CANFeatureScaler` parameters (min, max) are fitted strictly on the benign training split. Test data is transformed using the fixed fitted scaler.
* **Zero Test Leakage into Threshold Calibration**: Anomaly detection decision thresholds $\tau$ are calibrated exclusively on a designated benign validation split (e.g., 95th percentile of validation NLL). Ground-truth attack labels in the test set are never consulted during threshold calibration.
* **Supervised Baseline Partitioning**: Supervised models (MLP) are trained strictly on training data and validation splits; test frames are never concatenated into training sets.

---

## 3. Evaluated Model Architectures

| Architecture | Paradigm | Training Objective / Loss | Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Statistical Deviation Detector** | Unsupervised Timing Baseline | Inter-arrival timing z-score deviation | $\Delta t$ feature tracking |
| **Isolation Forest** | Unsupervised Ensemble Baseline | Path-length tree isolation | $n_{\text{estimators}} = 50$, contamination = 0.05 |
| **Supervised MLP Reference** | Supervised Neural Baseline | Binary Cross-Entropy (BCE) | Hidden layers $(32, 16)$, max iterations 100, Adam |
| **MPS Born Machine (Pooled)** | Quantum-Inspired Generative | Negative Log-Likelihood (NLL via Born rule) | $D \in \{8, 16, 24, 32\}$, trigonometric feature map |
| **MPS Born Machine (Seq-Preserving)**| Quantum-Inspired Generative | Negative Log-Likelihood on flattened temporal chain | $D \in \{8, 16, 24\}$, $W \times F$ sites |

---

## 4. Quantitative Evaluation Metrics

All evaluations report metrics at the specified unit of evaluation (window vs. frame) and explicitly log class prevalence:

1. **Attack-Class Binary Metrics** (Primary under extreme class imbalance):
   - **Precision**: $\text{Prec} = \frac{TP}{TP + FP}$
   - **Recall (TPR / Sensitivity)**: $\text{Rec} = \frac{TP}{TP + FN}$
   - **Binary F1**: $\text{F1}_{\text{binary}} = 2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$
2. **False Positive & Specificity Metrics**:
   - **False Positive Rate (FPR)**: $\text{FPR} = \frac{FP}{FP + TN}$
   - **True Negative Rate (TNR / Specificity)**: $\text{TNR} = \frac{TN}{TN + FP} = 1 - \text{FPR}$
3. **Threshold-Independent Metrics**:
   - **PR-AUC**: Area under the Precision-Recall curve (average precision).
   - **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
4. **Computational & Resource Metrics**:
   - **Inference Latency**: Microseconds per window/sample.
   - **Training Time**: CPU wall-clock seconds.
   - **Peak Memory**: Resident set size (RSS) memory delta in MB.

---

## 5. Multi-Seed Replication & Statistical Rigor

* All stochastic models (Isolation Forest, MLP, MPS random initialization, train/val partitioning) are executed across multiple fixed seeds:
  $$\text{Seeds} = [42, 101, 202]$$
* Results report sample mean and sample standard deviation:
  $$\bar{x} \pm s = \frac{1}{K} \sum_{k=1}^K x_k \pm \sqrt{\frac{1}{K-1} \sum_{k=1}^K (x_k - \bar{x})^2}$$
* Raw per-run results are preserved in machine-readable JSON format (`experiment_results.json`, `study_results.json`) to prevent hardcoded fabrication.
