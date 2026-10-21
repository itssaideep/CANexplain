"""DBC signal schema and CAN bitfield unpacking for physical signal decoding."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CANSignalDefinition:
    """Defines a physical signal packed inside a CAN frame payload.

    Attributes:
        name: Standardized signal identifier (e.g. 'vehicle_speed', 'steering_angle').
        start_bit: Zero-indexed start bit in the payload (0 to 63).
        bit_length: Bit length of the signal (1 to 32).
        byte_order: 'little_endian' (Intel) or 'big_endian' (Motorola).
        is_signed: True if signed two's complement integer.
        scale: Factor multiplied by raw integer value.
        offset: Physical offset added to scaled value.
        min_physical: Theoretical physical minimum for kinematic clamping.
        max_physical: Theoretical physical maximum for kinematic clamping.
        unit: Engineering unit string (e.g. 'km/h', 'deg', 'rpm').
    """

    name: str
    start_bit: int
    bit_length: int
    byte_order: str = "little_endian"
    is_signed: bool = False
    scale: float = 1.0
    offset: float = 0.0
    min_physical: float = 0.0
    max_physical: float = 100.0
    unit: str = ""

    def decode(self, payload: bytes) -> float:
        """Extracts and scales the physical signal from raw payload bytes.

        Args:
            payload: CAN payload bytes (up to 8 bytes).

        Returns:
            Physical engineering value (float) clamped to [min_physical, max_physical].
        """
        if len(payload) == 0:
            return self.offset

        # Convert payload bytes to a 64-bit unsigned integer (little-endian)
        payload_padded = payload.ljust(8, b"\x00")[:8]
        raw_uint64 = int.from_bytes(payload_padded, byteorder="little", signed=False)

        if self.byte_order == "little_endian":
            raw_val = (raw_uint64 >> self.start_bit) & ((1 << self.bit_length) - 1)
        else:
            # Motorola / Big-endian bit extraction
            # In standard DBC Motorola format, start_bit denotes MSB
            byte_idx = self.start_bit // 8
            bit_in_byte = self.start_bit % 8
            motorola_uint64 = int.from_bytes(payload_padded, byteorder="big", signed=False)
            shift = (7 - byte_idx) * 8 + bit_in_byte - self.bit_length + 1
            raw_val = (
                (motorola_uint64 >> shift) & ((1 << self.bit_length) - 1)
                if shift >= 0
                else 0
            )

        # Sign extension
        if self.is_signed and (raw_val & (1 << (self.bit_length - 1))):
            raw_val -= 1 << self.bit_length

        physical_val = float(raw_val) * self.scale + self.offset
        return float(min(max(physical_val, self.min_physical), self.max_physical))


@dataclass
class MessageDefinition:
    """Represents a CAN message with its associated signal definitions."""

    arbitration_id: int
    name: str
    dlc: int = 8
    signals: list[CANSignalDefinition] = field(default_factory=list)

    def decode(self, payload: bytes) -> dict[str, float]:
        """Decodes all defined signals from a payload."""
        results = {}
        for sig in self.signals:
            results[sig.name] = sig.decode(payload)
        return results


class DBCDatabase:
    """In-memory database of CAN messages and physical signal decoders."""

    def __init__(self, name: str = "generic_can"):
        self.name = name
        self.messages: dict[int, MessageDefinition] = {}

    def add_message(self, message: MessageDefinition) -> None:
        """Registers a message definition by arbitration ID."""
        self.messages[message.arbitration_id] = message

    def decode_frame(self, arbitration_id: int, payload: bytes) -> dict[str, float]:
        """Decodes physical signals if arbitration ID is registered."""
        if arbitration_id in self.messages:
            return self.messages[arbitration_id].decode(payload)
        return {}
