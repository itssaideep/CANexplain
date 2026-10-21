# Quantum-Inspired Tensor Network Architecture & Mathematical Explanations

This document details the mathematical formulation, tensor contractions, probability model, optimization objectives, and interpretability mechanisms implemented in `CANExplain-QI`.

---

## 1. Mathematical Foundation: Born Machine on Matrix Product States (MPS)

### 1.1 Physical Site State Encoding
Each discrete normalized feature dimension $x_k \in [0, 1]$ (for site $k \in \{0, \dots, N-1\}$) is embedded into a 2-dimensional local Hilbert space $\mathcal{H}_k \cong \mathbb{C}^2$ via a local trigonometric quantum feature map:

$$
|\phi(x_k)\rangle = \begin{pmatrix} \cos\left(\frac{\pi}{2} x_k\right) \\ \sin\left(\frac{\pi}{2} x_k\right) \end{pmatrix} \in \mathbb{R}^2
$$

**Properties:**
- $\langle \phi(x_k) | \phi(x_k) \rangle = \cos^2\left(\frac{\pi}{2} x_k\right) + \sin^2\left(\frac{\pi}{2} x_k\right) = 1.0$ (Strict unit $L_2$ norm).
- Orthogonality at boundaries: $\langle \phi(0) | \phi(1) \rangle = \cos(0)\cos(\frac{\pi}{2}) + \sin(0)\sin(\frac{\pi}{2}) = 0$.
- A multi-feature window sample $\mathbf{x} = (x_0, x_1, \dots, x_{N-1})$ forms an unentangled product state:
  $$
  |\Phi(\mathbf{x})\rangle = |\phi(x_0)\rangle \otimes |\phi(x_1)\rangle \otimes \dots \otimes |\phi(x_{N-1})\rangle \in \bigotimes_{k=0}^{N-1} \mathcal{H}_k
  $$

---

### 1.2 Matrix Product State (MPS) Representation
The collective multi-site joint probability amplitude $|\Psi\rangle$ is factorized into a train of 3-way order tensors $A^{(k)} \in \mathbb{R}^{D_{k} \times d_k \times D_{k+1}}$:

$$
|\Psi\rangle = \sum_{s_0, s_1, \dots, s_{N-1}=0}^{d-1} \left( A^{(0)}_{s_0} A^{(1)}_{s_1} \dots A^{(N-1)}_{s_{N-1}} \right) |s_0 s_1 \dots s_{N-1}\rangle
$$

where:
- $N$ is the number of sites ($N = 10$ for canonical stationary features; $N = W \times F$ for flattened sequence representations).
- $d = 2$ is the physical dimension (qubit equivalent).
- $D_k \le D$ is the virtual bond dimension governing bipartite entanglement capacity according to the Area Law.
- Open boundary conditions (OBC) enforce $D_0 = D_N = 1$.

---

### 1.3 Probability Model & Born Machine Anomaly Scoring
Following the quantum Born rule, the probability density function $P(\mathbf{x})$ assigned by the model to an observed automotive vector $\mathbf{x}$ is proportional to the modulus squared of the contraction amplitude:

$$
P(\mathbf{x}) = \frac{|\langle \Psi | \Phi(\mathbf{x}) \rangle|^2}{Z}, \quad \text{where } Z = \langle \Psi | \Psi \rangle
$$

The generative anomaly score is the **Negative Log-Likelihood (NLL)**:

$$
\mathcal{S}(\mathbf{x}) = -\ln P(\mathbf{x}) = -2 \ln |\langle \Psi | \Phi(\mathbf{x}) \rangle| + \ln Z
$$

When a CAN frame or window exhibits malicious deviations (e.g. injected arbitration IDs, high frequency timing floods, or anomalous sensor payload overrides), $|\langle \Psi | \Phi(\mathbf{x}) \rangle| \to 0$, causing $\mathcal{S}(\mathbf{x})$ to diverge positively.

---

## 2. Contraction Conventions & Numerical Stability

### 2.1 Single Sample Contraction
Given sample vectors $|\phi_k\rangle$:
1. For each site $k$, contract over the physical index $s_k$:
   $$
   M_k = \sum_{s_k=0}^{1} A^{(k)}_{:, s_k, :} \cdot \phi_k[s_k] \in \mathbb{R}^{D_k \times D_{k+1}}
   $$
2. Chain matrix multiplications from left to right:
   $$
   \langle \Psi | \Phi(\mathbf{x}) \rangle = M_0 \cdot M_1 \dots M_{N-1} \in \mathbb{R}^{1 \times 1}
   $$

### 2.2 Global State Norm $Z = \langle \Psi | \Psi \rangle$
The norm is computed via transfer matrix contraction:
$$
E^{(k)} = \sum_{s_k=0}^{1} A^{(k)}_{s_k} \otimes A^{(k)}_{s_k} \in \mathbb{R}^{D_k^2 \times D_{k+1}^2}
$$
In canonical gauge (left-canonical: $\sum_s (A^{(k)}_s)^\dagger A^{(k)}_s = \mathbb{I}_{D_{k+1}}$), $Z = 1.0$, guaranteeing numerical stability and preventing exponential underflow/overflow.

---

## 3. Training Objectives & Optimizers

`CANExplain-QI` provides two mathematically defined optimization engines:

### 3.1 Heuristic Correlation Optimizer (`HeuristicCorrelationOptimizer`)
- **Historical Context:** In early prototypes, this routine was colloquially labeled "DMRG". However, standard Density Matrix Renormalization Group (DMRG) solves eigenvalue problems $(H |\Psi\rangle = E |\Psi\rangle)$ via Lanczos/Davidson diagonalizations.
- **True Mathematical Objective:** This optimizer performs local two-site empirical correlation updates. For adjacent sites $(k, k+1)$, it constructs the empirical two-site density tensor from the training batch:
  $$
  \Theta^{(k, k+1)}_{s_k, s_{k+1}} = \frac{1}{B} \sum_{b=1}^B \phi^{(b)}_k[s_k] \cdot \phi^{(b)}_{k+1}[s_{k+1}]
  $$
  It blends this with the current local tensor $\tilde{B} = (1 - \eta) B^{(k, k+1)} + \eta (L \otimes \Theta \otimes R)$ and decomposes via Singular Value Decomposition (SVD):
  $$
  \tilde{B} = U \Sigma V^\dagger
  $$
  Truncating singular values to $\min(D, \text{rank})$ and absorbing $\Sigma$ into the sweep direction maintains canonical form.
- **Accurate Classification:** A DMRG-inspired correlation sweep optimizer.

### 3.2 Exact Gradient Born Machine Optimizer (`GradientBornMachineOptimizer`)
- **Objective:** Direct minimization of empirical Negative Log-Likelihood over training batch $\mathcal{B}$:
  $$
  \mathcal{L}(\Psi) = -\frac{2}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \ln |\langle \Psi | \Phi^{(b)} \rangle| + \ln \langle \Psi | \Psi \rangle
  $$
- **Analytical Gradient with Respect to Tensor $A^{(k)}$:**
  $$
  \nabla_{A^{(k)}} \mathcal{L} = -\frac{2}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \frac{1}{\langle \Psi | \Phi^{(b)} \rangle} \left[ L^{(b)}_k \otimes \phi^{(b)}_k \otimes R^{(b)}_k \right] + \frac{2}{Z} \left[ E_L^{(k)} \cdot A^{(k)} \cdot E_R^{(k)} \right]
  $$
  where $L^{(b)}_k$ and $R^{(b)}_k$ are the left and right partial sample contraction environments, and $E_L^{(k)}, E_R^{(k)}$ are the MPS norm transfer environments.

---

## 4. Interpretability via Quantum Information Metrics

Unlike black-box post-hoc explainers (such as SHAP or LIME) which require thousands of perturbed evaluations and rely on local surrogate approximations, tensor networks provide **direct, exact mathematical marginals** directly from the quantum state representation.

### 4.1 Exact Reduced Density Matrices (RDMs)

#### Single-Site RDM $\rho_k$
By tracing out all environment sites except site $k$:
$$
\rho_k = \text{Tr}_{\bar{k}}(|\Psi\rangle \langle \Psi|) / Z \in \mathbb{C}^{2 \times 2}
$$
Implemented via exact left and right transfer matrix environments:
$$
(\rho_k)_{s, s'} = \frac{1}{Z} \text{Tr}\left( E_L^{(k)} A^{(k)}_s E_R^{(k+1)} (A^{(k)}_{s'})^\dagger \right)
$$

#### Exact Two-Site Non-Adjacent RDM $\rho_{i, j}$
For arbitrary site pairs $i < j$:
1. If $j = i + 1$ (adjacent), contract $A^{(i)}$ and $A^{(j)}$ directly.
2. If $j > i + 1$ (non-adjacent), intervening sites $k \in \{i+1, \dots, j-1\}$ are traced out by propagating the transfer matrix $\mathbb{E}_k = \sum_s A^{(k)}_s \otimes (A^{(k)}_s)^*$:
   $$
   \mathbb{T}_{i \to j} = \prod_{k=i+1}^{j-1} \mathbb{E}_k
   $$
3. Guaranteed Hermitian, positive semi-definite (PSD via spectral eigenvalue clamping $\lambda_i \ge 0$), and unit trace ($\text{Tr}(\rho) = 1.0$).

---

### 4.2 von Neumann Entanglement Entropy
Quantifies the bipartite quantum entanglement between a single feature site $k$ and the rest of the CAN bus:

$$
S(\rho_k) = -\text{Tr}(\rho_k \log_2 \rho_k) = -\sum_{i=1}^2 \lambda_i \log_2 \lambda_i \quad [0 \le S \le 1.0 \text{ bit}]
$$

- $S(\rho_k) = 0$: Site $k$ is completely unentangled (independent from the rest of the network).
- $S(\rho_k) = 1.0$: Site $k$ is maximally entangled (e.g. strong cross-sensor functional coupling).

---

### 4.3 Quantum Mutual Information
Quantifies total (classical + quantum) correlations between feature $i$ and feature $j$:

$$
I(i; j) = S(\rho_i) + S(\rho_j) - S(\rho_{i, j}) \ge 0
$$

By subadditivity of von Neumann entropy, $I(i; j)$ is strictly non-negative.
- Verified on analytical 3-site GHZ state $(|000\rangle + |111\rangle)/\sqrt{2}$:
  $S(\rho_0) = 1.0$, $S(\rho_2) = 1.0$, $S(\rho_{0, 2}) = 1.0 \implies I(0; 2) = 1.0$ bit, confirming exact long-range entanglement detection across intervening sites.

---

### 4.4 Feature Marginal Probabilities & Root-Cause Attribution
For an incoming vector $\mathbf{x} = (x_0, \dots, x_{N-1})$, the marginal probability of feature $k$ under the benign generative distribution is:

$$
P(x_k) = \langle \phi(x_k) | \rho_k | \phi(x_k) \rangle \in [0, 1]
$$

When an anomaly triggers an alert, the forensic engine sorts features by probability collapse:
$$
k^* = \arg\min_k P(x_k)
$$
This directly isolates whether the intrusion originated from timing anomaly ($\Delta t$), unexpected arbitration ID injection ($\text{ID}$), or specific corrupted payload bytes ($b_0 \dots b_7$).

---

## 5. Perturbation Sensitivity & Explanation Fidelity

To prevent "explanation hallucination", `CANExplain-QI` provides a quantitative perturbation fidelity test (`canexplain.explainability.evaluation`):
- When a specific feature dimension $k$ in a normal frame is monotonically perturbed away from baseline by $\delta \in [0.1, 0.9]$:
  1. Marginal probability $P(x_k + \delta)$ must strictly decrease monotonically.
  2. Total anomaly score $\mathcal{S}(\mathbf{x} + \delta \mathbf{e}_k)$ must monotonically increase.
- Formally verified in `tests/unit/test_explanation_quality.py`.
