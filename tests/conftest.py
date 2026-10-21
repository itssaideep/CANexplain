"""Pytest configuration and shared test fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_can_dataframe():
    """Generates a small synthetic raw CAN DataFrame for testing."""
    return pd.DataFrame(
        {
            "timestamp": [0.001 * i for i in range(100)],
            "arbitration_id": ["0x120", "0x288", "0x120", "0x3D0"] * 25,
            "dlc": [8] * 100,
            "data": ["008000807FFA5589"] * 100,
            "label": [0] * 90 + [1] * 10,
        }
    )
