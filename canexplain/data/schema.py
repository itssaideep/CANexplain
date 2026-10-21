"""CAN frame data schema and validation utilities."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CANFrame:
    """Represents a validated Controller Area Network (CAN) frame.

    Attributes:
        timestamp: Time of frame transmission (seconds from epoch).
        arbitration_id: Standard (11-bit: 0x000-0x7FF) or extended (29-bit) identifier.
        dlc: Data Length Code (0 to 8 bytes).
        payload: Exactly 8 bytes representing normalized frame payload.
        label: Binary ground truth (0 = benign, 1 = attack).
        subclass: Optional detailed attack label (e.g., 'DoS', 'Fuzzing', 'RPM_Spoof').
        frame_index: Original sequential index of the frame in the source trace.
        is_extended_id: True if arbitration ID uses 29-bit extended CAN format.
        raw_payload_len: Original payload length in bytes before padding (0 to 8).
        dataset_name: Identifier for the source dataset (e.g., 'can-train-and-test-v1.5', 'ROAD').
        vehicle_id: Vehicle or platform identifier if known.
        trace_name: Name of the capture or trace file.
        split: Dataset split ('train', 'val', 'test', or None).
    """

    timestamp: float
    arbitration_id: int
    dlc: int = 8
    payload: bytes = b"\x00" * 8
    label: int = 0
    subclass: str | None = None
    frame_index: int | None = None
    is_extended_id: bool = False
    raw_payload_len: int | None = None
    dataset_name: str | None = None
    vehicle_id: str | None = None
    trace_name: str | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        max_id = 0x1FFFFFFF if self.is_extended_id else 0x7FF
        if not (0 <= self.arbitration_id <= max_id):
            id_type = "29-bit extended" if self.is_extended_id else "11-bit standard"
            raise ValueError(
                f"Arbitration ID 0x{self.arbitration_id:X} exceeds "
                f"{id_type} range (0x000-0x{max_id:X})."
            )
        if not (0 <= self.dlc <= 8):
            raise ValueError(f"DLC {self.dlc} invalid; must be between 0 and 8.")
        if len(self.payload) != 8:
            raise ValueError(
                f"Payload must be normalized to exactly 8 bytes; got {len(self.payload)}."
            )
        if self.label not in (0, 1):
            raise ValueError(f"Label must be binary (0 or 1); got {self.label}.")
        if self.raw_payload_len is not None and not (0 <= self.raw_payload_len <= 8):
            raise ValueError(
                f"raw_payload_len must be between 0 and 8; got {self.raw_payload_len}."
            )

    @property
    def id_hex(self) -> str:
        """Returns the arbitration ID formatted as an uppercase hex string."""
        if self.is_extended_id:
            return f"{self.arbitration_id:08X}"
        return f"{self.arbitration_id:03X}"

    @property
    def payload_hex(self) -> str:
        """Returns the payload formatted as a 16-character uppercase hex string."""
        return self.payload.hex().upper()


def parse_can_id(id_val: str | int, allow_extended: bool = False) -> int:
    """Parses and validates an arbitration ID from string (hex/dec) or integer.

    Args:
        id_val: String or integer representation of CAN arbitration ID.
        allow_extended: If True, permits 29-bit extended identifiers (up to 0x1FFFFFFF).
            If False, enforces 11-bit standard identifiers (up to 0x7FF).

    Returns:
        Integer arbitration ID.
    """
    if isinstance(id_val, int):
        arb_id = id_val
    elif isinstance(id_val, str):
        cleaned = id_val.strip()
        if cleaned.lower().startswith("0x"):
            arb_id = int(cleaned, 16)
        else:
            try:
                arb_id = int(cleaned, 16)
            except ValueError:
                arb_id = int(cleaned, 10)
    else:
        raise TypeError(f"Unsupported arbitration ID type: {type(id_val)}")

    max_id = 0x1FFFFFFF if allow_extended else 0x7FF
    if not (0 <= arb_id <= max_id):
        bound_desc = "29-bit extended (0x1FFFFFFF)" if allow_extended else "11-bit standard (0x7FF)"
        raise ValueError(f"Arbitration ID 0x{arb_id:X} exceeds {bound_desc} bounds.")
    return arb_id


def normalize_payload_bytes(payload_input: str | bytes | list[int], dlc: int = 8) -> bytes:
    """Normalizes payload input to an exact 8-byte bytearray, padding shorter payloads with zeros.

    Args:
        payload_input: Raw hex string, byte string, or list of byte integers.
        dlc: Data Length Code specifying expected number of payload bytes.

    Returns:
        Bytes object of length exactly 8.
    """
    if isinstance(payload_input, bytes):
        raw_bytes = payload_input
    elif isinstance(payload_input, list):
        raw_bytes = bytes(payload_input)
    elif isinstance(payload_input, str):
        cleaned = re.sub(r"[\s,\-#]", "", payload_input)
        if len(cleaned) % 2 != 0:
            cleaned = "0" + cleaned
        try:
            raw_bytes = bytes.fromhex(cleaned)
        except ValueError as err:
            raise ValueError(f"Invalid hexadecimal string for payload: {payload_input}") from err
    else:
        raise TypeError(f"Unsupported payload type: {type(payload_input)}")

    # Pad or truncate to exactly 8 bytes
    if len(raw_bytes) < 8:
        raw_bytes = raw_bytes + b"\x00" * (8 - len(raw_bytes))
    elif len(raw_bytes) > 8:
        raw_bytes = raw_bytes[:8]

    return raw_bytes
