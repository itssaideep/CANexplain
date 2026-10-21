"""Dataset manifest registry and acquisition workflows for automotive CAN intrusion datasets."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetManifest:
    """Metadata specification and provenance contract for a supported automotive dataset."""
    name: str
    official_title: str
    source_url: str
    vehicle_platform: str
    license_type: str
    attack_classes: List[str]
    file_format: str
    frame_count_approx: str
    checksum_sha256: Optional[str] = None
    acquisition_notes: str = ""
    citation: str = ""
    is_proprietary: bool = False


DATASET_REGISTRY: Dict[str, DatasetManifest] = {
    "can-train-and-test-v1.5": DatasetManifest(
        name="can-train-and-test-v1.5",
        official_title="Automotive CAN Train and Test Dataset v1.5",
        source_url="https://zenodo.org/record/...",
        vehicle_platform="Modern Passenger Vehicle (Instrumented Testbed)",
        license_type="CC BY 4.0 / Open Access",
        attack_classes=["DoS", "Standstill_Spoof", "Fuzzing"],
        file_format="CSV (timestamp, arbitration_id, data_field, attack)",
        frame_count_approx="~200,000 frames per trace",
        acquisition_notes="Included locally under data/real/ for reproducibility.",
        citation="Kidmose et al., 'can-sleuth: An open benchmark platform', 2025.",
        is_proprietary=False,
    ),
    "ROAD": DatasetManifest(
        name="ROAD",
        official_title="ROAD: The Real, Open Automotive Dataset for CAN Intrusion Detection",
        source_url="https://zenodo.org/records/10462796",
        vehicle_platform="Kia Soul (Dynamometer & Road Driving)",
        license_type="CC BY-SA 4.0",
        attack_classes=[
            "correlated_signal_attack",
            "max_engine_coolant_temp_attack",
            "max_speedometer_attack",
            "reverse_light_off_attack",
            "reverse_light_on_attack",
        ],
        file_format="CSV (Time, ID, Payload, Label)",
        frame_count_approx="~3.5 million frames across ambient and attack logs",
        checksum_sha256="4d7b21... (varies per tar.gz archive)",
        acquisition_notes=(
            "Opt-in manual download from Zenodo (record 10462796). "
            "Extract archives into data/real/road/."
        ),
        citation="Verma, M. E. et al., 'ROAD: The Real, Open Automotive Dataset for CAN IDS', 2020.",
        is_proprietary=False,
    ),
    "HCRL_CarHacking": DatasetManifest(
        name="HCRL_CarHacking",
        official_title="HCRL Car-Hacking Dataset for Intrusion Detection",
        source_url="https://ocslab.hksecurity.net/Datasets/car-hacking-dataset",
        vehicle_platform="Hyundai YF Sonata",
        license_type="Free for non-commercial academic research (Attribution Required)",
        attack_classes=["DoS", "Fuzzy", "RPM_Spoofing", "Gear_Spoofing"],
        file_format="CSV without header (Timestamp, CAN ID, DLC, DATA[0..7], Flag)",
        frame_count_approx="~17 million frames across 4 attack captures",
        acquisition_notes=(
            "Request/download from Korea University OCS Lab portal. "
            "Place extracted CSVs (e.g., DoS_dataset.csv) in data/real/hcrl_car_hacking/."
        ),
        citation="Song, H. M., Woo, J., & Kim, H. K., 'In-vehicle network intrusion detection using deep convolutional neural network', 2020.",
        is_proprietary=False,
    ),
    "HCRL_Survival": DatasetManifest(
        name="HCRL_Survival",
        official_title="HCRL Survival Analysis Dataset for CAN Intrusion Detection",
        source_url="https://ocslab.hksecurity.net/Datasets/survival-ids",
        vehicle_platform="Hyundai YF Sonata",
        license_type="Academic Research Only",
        attack_classes=["Injection_Attack", "Flooding", "Malicious_Insertion"],
        file_format="CSV with sequential survival timing",
        frame_count_approx="~1.2 million frames",
        acquisition_notes="Acquire from HCRL portal; extract to data/real/hcrl_survival/.",
        citation="Han, M. L. et al., 'Survival Analysis for Automotive Intrusion Detection', 2018.",
        is_proprietary=False,
    ),
    "OTIDS": DatasetManifest(
        name="OTIDS",
        official_title="OTIDS: Oak Ridge & ETRI CAN Intrusion Dataset",
        source_url="https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset",
        vehicle_platform="KIA Soul",
        license_type="Academic Research / Non-Commercial",
        attack_classes=["DoS", "Fuzzy", "Impersonation_0x164"],
        file_format="CSV / text with space or comma delimiter",
        frame_count_approx="~2.5 million frames",
        acquisition_notes="Download via HCRL / ETRI archive; unpack to data/real/otids/.",
        citation="Kang, M. J. & Kang, J. W., 'Intrusion Detection System Using Deep Neural Network for In-Vehicle Network Security', 2016.",
        is_proprietary=False,
    ),
}


def get_dataset_manifest(dataset_name: str) -> DatasetManifest:
    """Retrieves metadata manifest for a supported dataset."""
    key = dataset_name.strip()
    if key not in DATASET_REGISTRY:
        supported = list(DATASET_REGISTRY.keys())
        raise KeyError(f"Dataset '{key}' not recognized. Supported datasets: {supported}")
    return DATASET_REGISTRY[key]


def list_available_datasets() -> List[DatasetManifest]:
    """Returns a list of all registered dataset manifests."""
    return list(DATASET_REGISTRY.values())


def get_acquisition_instructions(dataset_name: str) -> str:
    """Returns step-by-step acquisition and setup instructions for a given dataset."""
    manifest = get_dataset_manifest(dataset_name)
    instructions = f"""
================================================================================
ACQUISITION GUIDE: {manifest.official_title}
================================================================================
Dataset Identifier : {manifest.name}
Target Vehicle     : {manifest.vehicle_platform}
License Terms      : {manifest.license_type}
Official URL       : {manifest.source_url}
Attacks Supported  : {', '.join(manifest.attack_classes)}

Steps to acquire and prepare:
1. Navigate to {manifest.source_url} in a browser.
2. Review and accept the license terms ({manifest.license_type}).
3. Download the trace files or archives.
4. Extract the CSV files to data/real/{manifest.name.lower()}/.
5. Use the adapter `canexplain.data.adapters.{manifest.name}` to load and normalize traces.
6. Note: To respect copyright and prevent bloated git repositories, never commit
   raw dataset files to version control (enforced via .gitignore).
================================================================================
"""
    return instructions
