"""Unit tests for dataset adapters, manifest registry, and schema validation."""

import os

import pytest

from canexplain.data.adapters import (
    CANONICAL_DATAFRAME_COLUMNS,
    HCRLCarHackingAdapter,
    OTIDSAdapter,
    ROADDatasetAdapter,
    SharedCANAdapter,
)
from canexplain.data.manifest import (
    get_acquisition_instructions,
    get_dataset_manifest,
    list_available_datasets,
)

FIXTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fixtures"))


def test_manifest_registry():
    manifests = list_available_datasets()
    assert len(manifests) == 5

    road_manifest = get_dataset_manifest("ROAD")
    assert road_manifest.name == "ROAD"
    assert "Zenodo" in road_manifest.source_url or "zenodo" in road_manifest.source_url
    assert "Kia Soul" in road_manifest.vehicle_platform

    instructions = get_acquisition_instructions("ROAD")
    assert "ACQUISITION GUIDE" in instructions
    assert "ROAD" in instructions

    with pytest.raises(KeyError):
        get_dataset_manifest("non_existent_dataset")


def test_road_dataset_adapter():
    fixture_path = os.path.join(FIXTURES_DIR, "road_sample.csv")
    adapter = ROADDatasetAdapter(vehicle_id="kia_soul")
    df = adapter.load_trace(fixture_path, trace_name="road_sample")

    for col in CANONICAL_DATAFRAME_COLUMNS:
        assert col in df.columns

    assert len(df) == 8
    assert df["dataset_name"].iloc[0] == "ROAD"
    assert df["vehicle_id"].iloc[0] == "kia_soul"

    # Frame 0 is normal -> label 0, subclass 'benign'
    assert df["label"].iloc[0] == 0
    assert df["subclass"].iloc[0] == "benign"

    # Frame 3 is correlated_signal_attack -> label 1
    assert df["label"].iloc[3] == 1
    assert df["subclass"].iloc[3] == "correlated_signal_attack"

    # Frame 7 is max_speedometer_attack -> label 1
    assert df["label"].iloc[7] == 1
    assert df["subclass"].iloc[7] == "max_speedometer_attack"


def test_hcrl_car_hacking_adapter():
    fixture_path = os.path.join(FIXTURES_DIR, "hcrl_sample.csv")
    adapter = HCRLCarHackingAdapter(vehicle_id="hyundai_yf_sonata")
    df = adapter.load_trace(fixture_path, trace_name="hcrl_sample_dos")

    for col in CANONICAL_DATAFRAME_COLUMNS:
        assert col in df.columns

    assert len(df) == 6
    assert df["dataset_name"].iloc[0] == "HCRL_CarHacking"

    # Regular 'R' -> 0
    assert df["label"].iloc[0] == 0
    assert df["subclass"].iloc[0] == "benign"

    # Transmitted 'T' -> 1, DoS subclass from trace_name
    assert df["label"].iloc[2] == 1
    assert df["subclass"].iloc[2] == "DoS"
    assert df["arbitration_id"].iloc[2] == 0x000


def test_otids_adapter():
    fixture_path = os.path.join(FIXTURES_DIR, "otids_sample.csv")
    adapter = OTIDSAdapter(vehicle_id="kia_soul")
    df = adapter.load_trace(fixture_path, trace_name="otids_impersonation")

    for col in CANONICAL_DATAFRAME_COLUMNS:
        assert col in df.columns

    assert len(df) == 5
    assert df["dataset_name"].iloc[0] == "OTIDS"

    # Benign -> 0
    assert df["label"].iloc[0] == 0
    # Attack -> 1
    assert df["label"].iloc[2] == 1
    assert df["subclass"].iloc[2] == "Impersonation"


def test_shared_can_adapter():
    real_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "real",
            "set_01",
            "train_01_attack_free",
            "accessory-1.csv",
        )
    )
    if os.path.exists(real_path):
        adapter = SharedCANAdapter()
        df = adapter.load_trace(real_path, sample_limit=50, trace_name="accessory-1")
        assert len(df) == 50
        for col in CANONICAL_DATAFRAME_COLUMNS:
            assert col in df.columns
        assert (df["label"] == 0).all()
        assert df["dataset_name"].iloc[0] == "can-train-and-test-v1.5"
