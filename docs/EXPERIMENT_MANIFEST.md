# Experiment Manifest: CANExplain-QI

## Reproducibility Protocol

All experiments follow strict zero-leakage protocols and standardized preprocessing grounded in `can-sleuth` (Kidmose et al. 2025):

### Preprocessing Specifications
- **Timestamp Standardization**: Starting timestamp fixed to `2023-01-01 00:00:00.000000000` (epoch 1672531200.0) while preserving microsecond inter-arrival deltas $\Delta t$.
- **Arbitration ID Representation**: Standard 11-bit hex string (`000` to `7FF`), normalized to float in $[0, 1]$.
- **10-Feature Architecture**: $t$, $\Delta t$, $\text{ID}$, $\text{DATA}[0]$ through $\text{DATA}[7]$ (each payload byte normalized to $[0, 1]$).
- **Windowing**: Window length $W=32$ (or $64$), stride $S=16$ (or $32$).
- **Normalization**: Scalers fitted strictly on benign training traffic (`train_01_attack_free`).

---

## Benchmark Execution

To reproduce the benchmark suite and generate the ISO/SAE 21434 forensic report:

```bash
python scripts/run_experiments.py
```

### Artifact Outputs
- `experiment_results.json`: Complete weighted metrics ($F_1$, PR-AUC, G-mean, FPR, Precision, Recall) across all attack scenarios (DoS flood, Fuzzing, Payload spoofing).
- `forensic_incident_report_sample.yaml`: Structured forensic report for detected anomalies.
