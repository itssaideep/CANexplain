"""Automotive CAN-bus dataset adapters for reproducible cross-dataset research."""

from canexplain.data.adapters.base import BaseCANAdapter, CANONICAL_DATAFRAME_COLUMNS
from canexplain.data.adapters.shared_can import SharedCANAdapter
from canexplain.data.adapters.road import ROADDatasetAdapter
from canexplain.data.adapters.hcrl_car_hacking import HCRLCarHackingAdapter
from canexplain.data.adapters.hcrl_survival import HCRLSurvivalAdapter
from canexplain.data.adapters.otids import OTIDSAdapter

__all__ = [
    "BaseCANAdapter",
    "CANONICAL_DATAFRAME_COLUMNS",
    "SharedCANAdapter",
    "ROADDatasetAdapter",
    "HCRLCarHackingAdapter",
    "HCRLSurvivalAdapter",
    "OTIDSAdapter",
]
