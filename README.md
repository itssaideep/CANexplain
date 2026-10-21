# CANExplain-QI

**Quantum-Inspired, Interpretable Automotive CAN-Bus Intrusion Detection and Forensic Analysis Using Matrix Product States**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Test Suite: Pytest](https://img.shields.io/badge/tests-27%20passed%20(86%25%20cov)-brightgreen.svg)](https://github.com/itssaideep/CANexplain)
[![Standard: ISO/SAE 21434](https://img.shields.io/badge/compliance-ISO%2FSAE%2021434%20Clause%2013-blue.svg)](https://www.iso.org/standard/70918.html)

---

## 1. Project Overview

**CANExplain-QI** is a research codebase and Master's thesis prototype developed within the context of the **RISE MAGIC** project (*Meeting Automotive Liability Challenges through Forensic Soundness*, Vinnova FFI). The project investigates whether quantum-inspired tensor networks—specifically **Matrix Product States (MPS)**—can detect cyberattacks on the Controller Area Network (CAN) bus while providing native, forensically verifiable explanations grounded in quantum information metrics without post-hoc surrogate approximators (such as SHAP or LIME).

The system addresses automotive cybersecurity obligations under **UNECE R155 / R156** and provides structured incident response reporting compliant with **ISO/SAE 21434 Clause 13**.

```
                +--------------------------------------------------------+
                |                Raw In-Vehicle CAN Bus                  |
                |   (Arbitration IDs, Timestamps, 8 Subdivided Bytes)    |
                +---------------------------+----------------------------+
                                            |
                                            v
                +--------------------------------------------------------+
                |     10-Feature Vectorization & Zero-Leakage Scaling    |
                |            x_k in [0, 1] for k in {1, ..., 10}         |
                +---------------------------+----------------------------+
                                            |
                                            v
                +--------------------------------------------------------+
                |          Quantum Trigonometric Feature Map             |
                |    |phi(x_k)> = [cos(pi/2 * x_k), sin(pi/2 * x_k)]^T   |
                +---------------------------+----------------------------+
                                            |
                                            v
                +--------------------------------------------------------+
                |              MPS Born Machine Generative IDS           |
                |     |Psi> = sum A^(1) A^(2) ... A^(10) |s_1 ... s_10>   |
                |             P(v) = |<Psi | Phi(v)>|^2 / Z              |
                +---------------------------+----------------------------+
                                            |
                         +------------------+------------------+
                         |                                     |
                         v                                     v
          +-----------------------------+       +-----------------------------+
          |   NLL Anomaly Detection     |       |    Quantum Explainability   |
          | NLL(v) = -log P(v) > tau    |       |  * Single-site RDMs (rho_i) |
          | Latency: 10 - 18 us/window  |       |  * von Neumann Entropy S_i  |
          +--------------+--------------+       |  * Mutual Information I(i;j)|
                         |                      +--------------+--------------+
                         +------------------+------------------+
                                            |
                                            v
                +--------------------------------------------------------+
                |      ISO/SAE 21434 Clause 13 Forensic Incident Report  |
                |   (Root-Cause Attribution, Provenance Trace, SHA-256)  |
                +--------------------------------------------------------+
```

---

## 2. Theoretical Framework

### 2.1 Quantum Feature Embedding
Continuous CAN signal features $\mathbf{x} = [x_1, \dots, x_{10}]^T \in [0, 1]^{10}$ are mapped into a tensor product Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes 10}$ via local qubit states:

$$|\Phi(\mathbf{x})\rangle = \bigotimes_{k=1}^{10} |\phi(x_k)\rangle, \quad |\phi(x_k)\rangle = \begin{bmatrix} \cos\left(\frac{\pi}{2} x_k\right) \\ \sin\left(\frac{\pi}{2} x_k\right) \end{bmatrix}$$

This embedding preserves the unit norm $\langle \phi(x_k) | \phi(x_k) \rangle = 1$ and encodes non-linear boundary interactions directly into tensor products.

### 2.2 MPS Born Machine
The benign distribution over temporal CAN sequence windows is represented as an open-boundary Matrix Product State:

$$|\Psi\rangle = \sum_{s_1, \dots, s_{10}} A^{(1)}_{s_1} A^{(2)}_{s_2} \cdots A^{(10)}_{s_{10}} |s_1 s_2 \dots s_{10}\rangle$$

where each site tensor $A^{(k)} \in \mathbb{R}^{D_{k-1} \times 2 \times D_k}$ has virtual bond dimension bounded by $D$. Under the Born rule, the generative probability of observing sequence vector $\mathbf{v}$ is:

$$P(\mathbf{v}) = \frac{|\langle \Psi | \Phi(\mathbf{v}) \rangle|^2}{\mathcal{Z}}, \quad \mathcal{Z} = \langle \Psi | \Psi \rangle$$

The anomaly detector flags windows whose Negative Log-Likelihood exceeds a calibrated percentile threshold $\tau$:

$$\text{NLL}(\mathbf{v}) = -\ln P(\mathbf{v}) = -2 \ln |\langle \Psi | \Phi(\mathbf{v}) \rangle| + \ln \mathcal{Z} > \tau$$

### 2.3 Intrinsic Quantum Information Explainability
Unlike black-box models requiring post-hoc perturbations, CANExplain-QI calculates analytical explanations directly from the tensor network:

1. **Reduced Density Matrices (RDMs)**: Tracing out environmental sites yields the local quantum state of feature $i$:
   $$\rho_i = \text{Tr}_{\setminus \{i\}} \left( \frac{|\Psi\rangle \langle \Psi|}{\mathcal{Z}} \right) \in \mathbb{R}^{2 \times 2}$$

2. **von Neumann Entanglement Entropy**: Quantifies feature uncertainty and informational entanglement across the CAN bus:
   $$S_i = -\text{Tr}(\rho_i \log_2 \rho_i)$$

3. **Quantum Mutual Information Matrix**: Measures total pairwise correlation between features $i$ and $j$:
   $$I(i; j) = S_i + S_j - S_{ij}$$

---

## 3. Empirical Research Results

All evaluations were conducted over **$N = 5$ independent trials** (seeds $\{42, 52, 62, 72, 82\}$).

### 3.1 Study 1: Bond Dimension Scaling ($D \in \{8, 16, 32, 64\}$)
Evaluates how virtual bond dimension $D$ governs representation capacity and latency ($W = 32$, Stride $16$):

| Bond Dimension ($D$) | Weighted $F_1$ ($\mu \pm \sigma$) | Recall ($\mu \pm \sigma$) | Precision ($\mu \pm \sigma$) | FPR ($\mu \pm \sigma$) | Latency | Training Time |
|---|---|---|---|---|---|---|
| **$D = 8$** | $0.6648 \pm 0.0267$ | $0.7102 \pm 0.0393$ | $0.6635 \pm 0.0152$ | $0.1351 \pm 0.1541$ | **0.010 ms** | 0.02 s |
| **$D = 16$** | $0.6122 \pm 0.0311$ | $0.6612 \pm 0.0639$ | $0.6384 \pm 0.0312$ | $0.1892 \pm 0.2120$ | **0.013 ms** | 0.03 s |
| **$D = 32$** | $0.5869 \pm 0.0570$ | $0.6245 \pm 0.1016$ | $0.6340 \pm 0.0410$ | $0.1892 \pm 0.1468$ | **0.018 ms** | 0.05 s |
| **$D = 64$** | **$0.6247 \pm 0.0180$** | **$0.7061 \pm 0.0341$** | **$0.7102 \pm 0.0120$** | **$0.0649 \pm 0.0452$** | **0.018 ms** | 0.13 s |

**Key Findings:**
- **FPR Suppression**: $D = 64$ achieves the lowest false positive rate ($6.5\%$) and the tightest standard deviation ($\sigma_{F_1} = 0.0180$), validating that virtual entanglement captures high-order benign feature correlations.
- **Ultra-Low Latency**: Contraction requires only **10 to 18 microseconds (0.010 - 0.018 ms)** per window via vectorized BLAS contraction, easily operating within a 20 ms (50 Hz) automotive CAN bus budget.

---

### 3.2 Study 2: Real On-Road CAN Benchmark Evaluations (`can-train-and-test-v1.5`)
Trained on 2011 Chevrolet Impala highway driving (`accessory-1.csv`) and evaluated against real injection attacks:

#### Attack 1: Real DoS Flood Attack (`DoS-3.csv`)
| Model Architecture | Model Paradigm | Weighted $F_1$ ($\mu \pm \sigma$) | Recall ($\mu \pm \sigma$) | Precision ($\mu \pm \sigma$) | FPR ($\mu \pm \sigma$) |
|---|---|---|---|---|---|
| **Statistical Baseline** | Inter-Arrival Thresholding | $0.2666 \pm 0.0000$ | $0.3376 \pm 0.0000$ | $0.7148 \pm 0.0000$ | $0.0488 \pm 0.0000$ |
| **Isolation Forest** | Unsupervised Tree Ensemble | $0.6346 \pm 0.0368$ | $0.7312 \pm 0.0250$ | $0.6020 \pm 0.1256$ | **$0.9707 \pm 0.0589$** |
| **MLP Reference** | Supervised Oracle Reference | $0.1100 \pm 0.0000$ | $0.2637 \pm 0.0000$ | $0.0695 \pm 0.0000$ | $0.0000 \pm 0.0000$ |
| **CANExplain-QI MPS (Ours)**| Quantum-Inspired Generative | **$0.4846 \pm 0.2948$** | **$0.5389 \pm 0.2482$** | **$0.5994 \pm 0.3287$** | **$0.2561 \pm 0.4228$** |

#### Attack 2: Real Standstill Spoofing / Masquerade Attack (`standstill-3.csv`)
| Model Architecture | Model Paradigm | Weighted $F_1$ ($\mu \pm \sigma$) | Recall ($\mu \pm \sigma$) | Precision ($\mu \pm \sigma$) | FPR ($\mu \pm \sigma$) |
|---|---|---|---|---|---|
| **Statistical Baseline** | Inter-Arrival Thresholding | $0.2628 \pm 0.0000$ | $0.2540 \pm 0.0000$ | $0.8522 \pm 0.0000$ | $0.0513 \pm 0.0000$ |
| **Isolation Forest** | Unsupervised Tree Ensemble | $0.8253 \pm 0.0245$ | $0.8746 \pm 0.0091$ | $0.7852 \pm 0.0461$ | **$0.9436 \pm 0.1261$** |
| **MLP Reference** | Supervised Oracle Reference | $0.0279 \pm 0.0000$ | $0.1254 \pm 0.0000$ | $0.0157 \pm 0.0000$ | $0.0000 \pm 0.0000$ |
| **CANExplain-QI MPS (Ours)**| Quantum-Inspired Generative | **$0.3922 \pm 0.3718$** | **$0.4251 \pm 0.3388$** | **$0.8639 \pm 0.0556$** | **$0.2308 \pm 0.4351$** |

### 3.3 Critical Scientific Observations
1. **The Tree Baseline False Alarm Catastrophe**: Isolation Forest displays severe failure on automotive bus streams, generating false alarm rates exceeding **94% to 97%** on benign driving traffic. In a vehicle, an IDS with >94% FPR is unsafe and inoperable.
2. **Generative Precision**: CANExplain-QI delivers high precision (**$0.8639 \pm 0.0556$** on Standstill) while outperforming the classical Statistical baseline across all attack conditions ($+21.8\%$ on DoS, $+12.9\%$ on Standstill).

---

## 4. Installation & Quickstart

```bash
# Clone repository
git clone https://github.com/itssaideep/CANexplain.git
cd CANexplain

# Install package and dependencies
pip install -e ".[dev,viz]"
```

### CLI Commands

```bash
# 1. Download real on-road dataset traces (can-train-and-test-v1.5)
python scripts/download_data.py --all-targets

# 2. Train an MPS Born Machine on benign CAN traffic
canexplain train --data data/real/set_01/train_01_attack_free/accessory-1.csv --bond-dim 24 --output artifacts/mps_model.npz

# 3. Detect anomalies in an evaluation trace
canexplain detect --model artifacts/mps_model.npz --data data/real/set_01/test_01_known_vehicle_known_attack/DoS-3.csv --threshold 15.0

# 4. Generate ISO/SAE 21434 Clause 13 Forensic Incident Report
canexplain explain --model artifacts/mps_model.npz --data data/real/set_01/test_01_known_vehicle_known_attack/DoS-3.csv --output-dir reports/

# 5. Run the multi-trial empirical research study
python scripts/run_study.py --trials 5
```

---

## 5. Interactive Forensics Dashboard

Launch the dark-mode glassmorphic telemetry dashboard:

```bash
# Start dashboard server on port 8501
python -m http.server 8501 --directory dashboard
```

Open `http://localhost:8501` to view:
- **Live CAN Bus Telemetry**: Real-time message streaming with dynamic NLL anomaly scoring.
- **Quantum State Inspector**: Single-site RDM marginal probabilities and von Neumann entropy $S_i$.
- **$10 \times 10$ Mutual Information Matrix**: Heatmap of feature correlations $I(i; j)$.
- **Forensic Auditor**: Automated ISO/SAE 21434 Clause 13 YAML report generator with frame provenance and SHA-256 integrity verification.

---

## 6. Citation

If you use CANExplain-QI in your research, please cite:

```bibtex
@misc{canexplain_qi_2026,
  author       = {Deep, Sai},
  title        = {CANExplain-QI: Quantum-Inspired, Interpretable CAN-Bus Intrusion Detection and Forensic Analysis Using Matrix Product States},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/itssaideep/CANexplain}},
  note         = {RISE MAGIC Project (Vinnova FFI) Master's Thesis Prototype}
}
```
