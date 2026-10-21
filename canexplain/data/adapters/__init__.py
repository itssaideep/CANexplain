"""Automotive CAN-bus dataset adapters for reproducible cross-dataset research."""

from canexplain.data.adapters.base import CANONICAL_DATAFRAME_COLUMNS, BaseCANAdapter
from canexplain.data.adapters.hcrl_car_hacking import HCRLCarHackingAdapter
from canexplain.data.adapters.hcrl_survival import HCRLSurvivalAdapter
from canexplain.data.adapters.otids import OTIDSAdapter
from canexplain.data.adapters.road import ROADDatasetAdapter
from canexplain.data.adapters.shared_can import SharedCANAdapter

__all__ = [
    "BaseCANAdapter",
    "CANONICAL_DATAFRAME_COLUMNS",
    "SharedCANAdapter",
    "ROADDatasetAdapter",
    "HCRLCarHackingAdapter",
    "HCRLSurvivalAdapter",
    "OTIDSAdapter",
]
