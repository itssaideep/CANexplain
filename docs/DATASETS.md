# Automotive CAN-Bus Dataset Registry & Provenance Guide

**Project:** CANExplain-QI (Quantum-Inspired Matrix Product State CAN Intrusion Detection)  
**Document:** `docs/DATASETS.md`  
**Date:** 2026-10-21  
**Status:** Validated Multi-Dataset Adapters and Manifest Specification (Stage 3)  

---

## 1. Overview and Ethical Data Principles

Reproducible research in in-vehicle network security requires rigorous evaluation across diverse vehicle platforms, baud rates, and electronic control unit (ECU) topologies. In strict alignment with ethical research practices:
1. **No Proprietary Dependencies**: The repository operates exclusively with validated public automotive datasets and self-contained synthetic streams.
2. **No Data Bloat in Version Control**: Large binary logs, `.csv`, and `.csv.gz` files are strictly excluded via `.gitignore`.
3. **Explicit Provenance Contracts**: All loaded frames conform to an immutable, canonical `CANFrame` schema tracking dataset origin, vehicle ID, original sequential frame index, unpadded DLC, and attack subclass.
4. **Separation of Ground-Truth Labels**: Labels and subclasses are segregated from model feature matrices and never leaked into training, scaler fitting, or threshold calibration pipelines.

---

## 2. Comprehensive Dataset Registry

| Dataset Identifier | Target Vehicle Platform | Source & Citations | License | Primary Attack Classes | Format & Schema |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`can-train-and-test-v1.5`** | Modern Passenger Car (Instrumented Testbed) | Kidmose et al. 2025 (`can-sleuth` platform) | CC BY 4.0 | DoS, Standstill Spoof, Fuzzing | CSV (`timestamp, arbitration_id, data_field, attack`) |
| **`ROAD`** | Kia Soul (Dynamometer & Road Driving) | Verma et al. 2020 (ORNL, Zenodo: 10462796) | CC BY-SA 4.0 | Correlated Signal, Coolant Temp, Speedometer, Reverse Light | CSV (`Time, ID, Payload, Label`) |
| **`HCRL_CarHacking`** | Hyundai YF Sonata | Song, Woo, & Kim 2020 (Korea Univ. OCS Lab) | Non-commercial Academic Research | DoS (0x000), Fuzzy, RPM Spoof (0x316), Gear (0x43F) | Headerless CSV (`Timestamp, ID, DLC, DATA[0..7], Flag`) |
| **`HCRL_Survival`** | Hyundai YF Sonata | Han et al. 2018 (Korea Univ. HCRL) | Academic Research Only | Injection Floods, Periodic Interval Insertions | CSV (`timestamp, id, dlc, b0..b7, label`) |
| **`OTIDS`** | KIA Soul | Kang & Kang 2016 (Korea Univ. / ETRI) | Non-commercial Academic Research | DoS, Fuzzy, Impersonation (e.g., 0x164) | Space/comma delimited (`Timestamp, ID, DLC, Data, Label`) |

---

## 3. Detailed Dataset Specifications

### 3.1 `can-train-and-test-v1.5` (Integrated Baseline)
* **Confirmed Facts**:
  - Contains separated directory structure: `train_01_attack_free` and `test_01_known_vehicle_known_attack`.
  - Continuous float timestamp representing epoch seconds.
  - Standard 11-bit hex arbitration IDs (e.g., `0F1`, `1E9`, `199`).
  - Payloads range from 4 to 8 bytes (padded to 8 bytes in canonical format).
  - Ground-truth binary flag: `0` (benign) and `1` (attack).
* **Caveats & Audit Notes**:
  - In `standstill-3.csv`, attack frames represent only 4.64% of total frames. Sliding windows of $W=32$ with stride $S=16$ using "any-frame" positive labeling inflated the window-level attack prevalence to 87.5%, leaving only 39 benign test windows.
* **Supported Tasks**: Within-dataset benign baseline fitting, DoS detection, and subtle payload falsification detection.

### 3.2 ROAD Dataset (Real, Open Automotive Dataset)
* **Official URL**: [https://zenodo.org/records/10462796](https://zenodo.org/records/10462796)
* **Confirmed Facts**:
  - Captured on a Kia Soul by Oak Ridge National Laboratory (ORNL).
  - Contains real driving logs on highways, city streets, and chassis dynamometers.
  - Verified physical vehicle reactions (e.g. erratic speedometer motion, engine light activation).
  - Attacks explicitly distinguish advanced fabrication and stealthy falsification attacks from crude DoS floods.
* **Caveats**:
  - Labels in raw files use string tokens (`normal`, `correlated_signal_attack`, `max_speedometer_attack`).
  - Timestamps are continuous relative seconds from capture start.
* **Adapter Implementation**: `canexplain.data.adapters.ROADDatasetAdapter`.

### 3.3 HCRL Car-Hacking Dataset
* **Official URL**: [https://ocslab.hksecurity.net/Datasets/car-hacking-dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
* **Confirmed Facts**:
  - Captured via OBD-II port on a Hyundai YF Sonata while engine running / driving on a test road.
  - 11 positional columns: `Timestamp, CAN ID, DLC, DATA[0] ... DATA[7], Flag`.
  - Flags: `'R'` denotes regular benign frames; `'T'` denotes injected attack frames.
  - In DoS attack captures, injected CAN ID is `0x000` with payload `00 00 00 00 00 00 00 00`.
* **Caveats**:
  - Files are large (hundreds of megabytes each, up to ~17M total frames).
  - Raw files have no CSV header.
* **Adapter Implementation**: `canexplain.data.adapters.HCRLCarHackingAdapter`.

### 3.4 OTIDS (CAN Intrusion Dataset)
* **Official URL**: [https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset](https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset)
* **Confirmed Facts**:
  - Recorded from a KIA Soul.
  - Contains subtle impersonation attacks where injected frames take over legitimate arbitration IDs (e.g. `0x164`), perturbing inter-arrival intervals without changing ID distributions.
* **Adapter Implementation**: `canexplain.data.adapters.OTIDSAdapter`.

---

## 4. Canonical CAN Schema Contract

All adapters normalize heterogeneous raw sources into a single immutable DataFrame conforming to `CANONICAL_DATAFRAME_COLUMNS`:

```python
CANONICAL_DATAFRAME_COLUMNS = [
    "timestamp",  # float: continuous seconds
    "arbitration_id",  # int: validated integer ID (0x000 to 0x7FF or 0x1FFFFFFF)
    "id_hex",  # str: uppercase hex string ("120" or "18DAF110")
    "dlc",  # int: Data Length Code (0 to 8)
    "data_hex",  # str: 16-character uppercase hex string (padded to 8 bytes)
    "payload_bytes",  # bytes: exactly 8 bytes
    "label",  # int: binary 0 (benign) or 1 (attack)
    "subclass",  # str: granular attack name or 'benign'
    "frame_index",  # int: original sequential row position in trace
    "is_extended_id",  # bool: True for 29-bit extended CAN IDs
    "raw_payload_len",  # int: original unpadded byte count
    "dataset_name",  # str: origin dataset identifier
    "vehicle_id",  # str: origin vehicle platform
    "trace_name",  # str: capture file name
]
```

---

## 5. Opt-In Data Acquisition Workflow

To load external datasets:
1. Query metadata programmatically:
   ```python
   from canexplain.data.manifest import get_acquisition_instructions

   print(get_acquisition_instructions("ROAD"))
   ```
2. Download files manually from official portals and unpack into `data/real/<dataset_name>/`.
3. Load through corresponding adapter:
   ```python
   from canexplain.data.adapters import ROADDatasetAdapter

   adapter = ROADDatasetAdapter(vehicle_id="kia_soul")
   df = adapter.load_trace("data/real/road/ambient_highway_1.csv")
   ```
4. CI test runs use pre-packaged offline synthetic fixtures in `tests/fixtures/`, requiring zero external network calls.
