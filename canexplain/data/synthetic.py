"""Deterministic synthetic CAN-bus traffic generator for CI and testing."""

from typing import List, Tuple
import numpy as np
import pandas as pd

from canexplain.data.schema import CANFrame
from canexplain.data.loaders import STANDARD_BASE_TIMESTAMP


def generate_synthetic_can_stream(
    num_frames: int = 2000,
    num_ecus: int = 5,
    attack_type: str = "none",
    attack_start_idx: int = 1000,
    attack_duration: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic CAN traffic with realistic periodicities and attack injections.

    Args:
        num_frames: Total number of CAN frames to generate.
        num_ecus: Number of distinct benign ECUs / arbitration IDs.
        attack_type: One of 'none', 'dos', 'fuzzing', 'spoofing', 'suppress'.
        attack_start_idx: Frame index where attack injection begins.
        attack_duration: Duration of attack in frame count.
        seed: Random seed for deterministic generation.

    Returns:
        Structured pandas DataFrame matching the canonical schema.
    """
    rng = np.random.default_rng(seed)

    # Periodic ECU profiles: (Arbitration ID, nominal interval in ms, payload generator)
    ecu_ids = [0x120, 0x180, 0x220, 0x288, 0x3D0][:num_ecus]
    intervals = [0.010, 0.020, 0.050, 0.020, 0.100][:num_ecus]

    rows = []
    current_time = STANDARD_BASE_TIMESTAMP

    # Track time of each ECU
    next_times = {eid: current_time + rng.uniform(0, intervals[i]) for i, eid in enumerate(ecu_ids)}

    for idx in range(num_frames):
        # Pick next periodic frame
        target_id = min(next_times, key=next_times.get)
        t = next_times[target_id]
        current_time = t
        i_ecu = ecu_ids.index(target_id)
        next_times[target_id] = t + rng.normal(intervals[i_ecu], intervals[i_ecu] * 0.05)

        # Generate realistic periodic payload (counter + constant + sensor drift)
        counter = idx % 256
        sensor_val = int(128 + 50 * np.sin(idx / 50.0))
        payload = bytes([counter, sensor_val, 0x00, 0x80, 0x12, 0x34, 0x56, 0x78])

        is_attack = False
        subclass = "benign"

        # Check for attack window
        if attack_start_idx <= idx < (attack_start_idx + attack_duration):
            if attack_type == "dos":
                # High-priority ID 0x000 flood with zero payload
                target_id = 0x000
                payload = b"\x00" * 8
                is_attack = True
                subclass = "DoS"
                # DoS drastically compresses time
                current_time += 0.0002
            elif attack_type == "fuzzing":
                # Random arbitration ID and random payload
                target_id = int(rng.integers(0x001, 0x7FE))
                payload = bytes(rng.integers(0, 256, size=8, dtype=np.uint8))
                is_attack = True
                subclass = "Fuzzing"
            elif attack_type == "spoofing":
                # Target specific ECU (e.g., 0x120) with out-of-distribution sensor value
                if target_id == ecu_ids[0]:
                    payload = bytes([counter, 255, 0xFF, 0xFF, 0x12, 0x34, 0x56, 0x78])
                    is_attack = True
                    subclass = "RPM_Spoof"
            elif attack_type == "suppress":
                # Suppress messages from the first ECU
                if target_id == ecu_ids[0]:
                    # In a suppress attack, the frame is dropped/absent.
                    # We skip recording this frame
                    continue

        rows.append({
            "timestamp": current_time,
            "arbitration_id": target_id,
            "id_hex": f"{target_id:03X}",
            "dlc": 8,
            "data_hex": payload.hex().upper(),
            "payload_bytes": payload,
            "label": 1 if is_attack else 0,
            "subclass": subclass
        })

    return pd.DataFrame(rows)
