# Model Card: CANExplain-QI (MPS Born Machine)

## Model Overview
- **Model Name**: CANExplain-QI Matrix Product State (MPS) Born Machine
- **Architecture**: Quantum-Inspired 1D Tensor Network with Open Boundary Conditions ($D_0 = D_N = 1$)
- **Optimization**: Two-Site Variational Density Matrix Renormalization Group (DMRG)
- **Objective Function**: Empirical Negative Log-Likelihood (NLL) minimization on benign traffic
- **Primary Use Case**: Unsupervised anomaly detection, quantum information interpretability, and post-incident digital forensics on Controller Area Network (CAN) traffic.

---

## Intended Use & Context
- **Domain**: Automotive in-vehicle network cybersecurity (ISO 11898, SAE J1939).
- **Target Deployment**: Dedicated bus monitoring node or Security Gateway (ECU).
- **Compliance Context**: Aligned with **ISO/SAE 21434 Clause 13** (Cybersecurity Monitoring and Incident Response) and **UNECE R155/R156** automotive liability mandates.

---

## Mathematical Formulation
- **State Vector Product**:
  $$|\Phi(\mathbf{v})\rangle = |\phi(v_1)\rangle \otimes |\phi(v_2)\rangle \otimes \dots \otimes |\phi(v_N)\rangle$$
  where local continuous features $v_i \in [0, 1]$ are mapped via:
  $$|\phi(v_i)\rangle = \begin{pmatrix} \cos(\frac{\pi}{2} v_i) \\ \sin(\frac{\pi}{2} v_i) \end{pmatrix} \in \mathbb{R}^2 \quad (p=2)$$
- **Born Rule Probability**:
  $$\mathcal{P}(\mathbf{v}) = \frac{|\langle \Psi | \Phi(\mathbf{v}) \rangle|^2}{\mathcal{Z}}, \quad \mathcal{Z} = \langle \Psi | \Psi \rangle$$
- **Anomaly Scoring**:
  $$\text{NLL}(\mathbf{v}) = -2 \log |\langle \Psi | \Phi(\mathbf{v}) \rangle| + \log \mathcal{Z}$$
- **Intrinsic Quantum Explainability**:
  - Single-Site RDM: $\rho_k = \text{Tr}_{\setminus k}(|\Psi\rangle\langle\Psi|)$
  - Von Neumann Entanglement Entropy: $S_k = -\text{Tr}(\rho_k \log_2 \rho_k)$
  - Pairwise Mutual Information: $I(j; k) = S_j + S_k - S_{j, k}$

---

## Limitations and Operational Caveats
1. **Statistical Deviation $\neq$ Confirmed Attack**: High NLL scores reflect deviations from the learned normal generative baseline. Rare but non-malicious driving conditions (e.g., emergency braking, diagnostic servicing) may trigger elevated scores.
2. **1D Chain Ordering Sensitivity**: As with all 1D tensor networks, feature ordering across the MPS chain influences short-range vs. long-range virtual entanglement.
3. **Dedicated Hardware Node Required**: MPS tensor contraction requires floating-point operations; deployment should target a dedicated telematics gateway or standalone cybersecurity ECU rather than resource-constrained low-end microcontrollers.
