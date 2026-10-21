"""CAN frame data schema and validation utilities."""

from dataclasses import dataclass
from typing import Optional, Tuple
import re


@dataclass(frozen=True)
class CANFrame:
    """Represents a validated Controller Area Network (CAN) frame.

    Attributes:
        timestamp: Time of frame transmission (seconds from epoch).
        arbitration_id: 11-bit standard CAN identifier (integer 0x000 to 0x7FF).
        dlc: Data Length Code (0 to 8 bytes).
        payload: Exactly 8 bytes representing frame payload.
        label: Binary ground truth (0 = benign, 1 = attack).
        subclass: Optional detailed attack label (e.g., 'DoS', 'Fuzzing', 'RPM_Spoof').
    """
    timestamp: float
    arbitration_id: int
    dlc: int
    payload: bytes
    label: int = 0
    subclass: Optional[str] = None

    def __post_init__(self) -> None:
        if not (0 <= self.arbitration_id <= 0x7FF):
            raise ValueError(f"Arbitration ID 0x{self.arbitration_id:X} exceeds 11-bit range (0x000-0x7FF).")
        if not (0 <= self.dlc <= 8):
            raise ValueError(f"DLC {self.dlc} invalid; must be between 0 and 8.")
        if len(self.payload) != 8:
            raise ValueError(f"Payload must be normalized to exactly 8 bytes; got {len(self.payload)}.")
        if self.label not in (0, 1):
            raise ValueError(f"Label must be binary (0 or 1); got {self.label}.")

    @property
    def id_hex(self) -> str:
        """Returns the arbitration ID formatted as a 3-character uppercase hex string."""
        return f"{self.arbitration_id:03X}"

    @property
    def payload_hex(self) -> str:
        """Returns the payload formatted as a 16-character uppercase hex string."""
        return self.payload.hex().upper()


def parse_can_id(id_val: str | int) -> int:
    """Parses and validates an arbitration ID from string (hex/dec) or integer."""
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

    if not (0 <= arb_id <= 0x7FF):
        raise ValueError(f"Arbitration ID 0x{arb_id:X} out of standard 11-bit bounds.")
    return arb_id


def normalize_payload_bytes(payload_input: str | bytes | list[int], dlc: int = 8) -> bytes:
    """Normalizes payload input to an exact 8-byte bytearray, padding shorter payloads with zeros."""
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

    # Pad or truncate to exactly 8 bytes (padding leading/trailing per CAN standard)
    if len(raw_bytes) < 8:
        # Pad with zeros up to 8 bytes
        raw_bytes = raw_bytes + b"\x00" * (8 - len(raw_bytes))
    elif len(raw_bytes) > 8:
        raw_bytes = raw_bytes[:8]

    return raw_bytes
