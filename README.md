# CANExplain-QI

**Quantum-Inspired, Interpretable CAN-Bus Intrusion Detection and Forensic Analysis Using Matrix Product States**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

CANExplain-QI is a research prototype and Master's thesis codebase investigating whether quantum-inspired tensor networks—specifically **Matrix Product States (MPS)**—can detect cyberattacks on the automotive Controller Area Network (CAN) bus while providing native, forensically sound explanations grounded in quantum information metrics (Reduced Density Matrices, von Neumann entropy, and mutual information).

Developed within the context of the **RISE MAGIC** project (Meeting Automotive Liability Challenges through Forensic Soundness, Vinnova FFI), addressing automotive digital forensics and incident response requirements under **UNECE R155/R156** and **ISO/SAE 21434**.

---

## Key Features

1. **Unsupervised Generative Anomaly Detection**:
   - Models benign CAN sequence distributions via an MPS Born machine ($P(\mathbf{v}) = |\langle \Psi | \Phi(\mathbf{v}) \rangle|^2 / \mathcal{Z}$).
   - Employs Negative Log-Likelihood (NLL) thresholding calibrated on normal traffic without attack exposure.
2. **Intrinsic Quantum Information Explainability**:
   - Direct marginal probabilities from single-site Reduced Density Matrices (RDMs).
   - Informational correlation tracking via von Neumann entanglement entropy ($S_i = -\text{Tr}(\rho_i \log_2 \rho_i)$).
   - Inter-feature dependency mapping via pairwise mutual information ($I(i; j) = S_i + S_j - S_{ij}$).
   - Zero reliance on post-hoc surrogate tools (SHAP, LIME).
3. **Forensic Traceability under ISO/SAE 21434**:
   - Structured incident reports linking alerts to specific CAN arbitration IDs, payload byte indices, and microsecond timing anomalies.
4. **Standardized Preprocessing**:
   - 10-feature architecture with subdivided 8-byte payloads and normalized timestamp deltas grounded in `can-sleuth` (Kidmose et al. 2025).

---

## Installation

```bash
git clone https://github.com/itssaideep/CANexplain.git
cd CANexplain
pip install -e ".[dev,viz]"
```

---

## Usage

```bash
# Run CLI help
canexplain --help

# Run tests
pytest
```
