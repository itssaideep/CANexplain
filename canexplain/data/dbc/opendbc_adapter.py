"""Open automotive CAN DBC profiles and manufacturer message definitions."""

from canexplain.data.dbc.signal_schema import (
    CANSignalDefinition,
    DBCDatabase,
    MessageDefinition,
)

# Standardized universal vehicle dynamics signal names
UNIVERSAL_DYNAMICS_SIGNALS = [
    "vehicle_speed",  # km/h [0, 250]
    "steering_angle",  # deg [-500, +500]
    "engine_rpm",  # rpm [0, 8000]
    "throttle_pct",  # % [0, 100]
    "brake_active",  # binary [0, 1]
    "gear_num",  # integer [0, 8]
]


def create_hyundai_sonata_dbc() -> DBCDatabase:
    """Constructs DBC definitions for Hyundai YF Sonata / Kia Soul CAN bus."""
    db = DBCDatabase(name="hyundai_yf_sonata")

    # 1. Engine & Throttle (0x316 / 790 decimal)
    msg_engine = MessageDefinition(
        arbitration_id=0x316,
        name="EMS_11",
        signals=[
            CANSignalDefinition(
                name="engine_rpm",
                start_bit=16,
                bit_length=16,
                scale=0.25,
                offset=0.0,
                min_physical=0.0,
                max_physical=8000.0,
                unit="rpm",
            ),
            CANSignalDefinition(
                name="throttle_pct",
                start_bit=32,
                bit_length=8,
                scale=0.392,
                offset=0.0,
                min_physical=0.0,
                max_physical=100.0,
                unit="%",
            ),
        ],
    )
    db.add_message(msg_engine)

    # 2. Steering Angle Sensor (0x164 / 356 decimal)
    msg_sas = MessageDefinition(
        arbitration_id=0x164,
        name="SAS11",
        signals=[
            CANSignalDefinition(
                name="steering_angle",
                start_bit=0,
                bit_length=16,
                is_signed=True,
                scale=0.1,
                offset=0.0,
                min_physical=-500.0,
                max_physical=500.0,
                unit="deg",
            ),
        ],
    )
    db.add_message(msg_sas)

    # 3. Wheel Speed & Vehicle Speed (0x220 / 544 decimal)
    msg_speed = MessageDefinition(
        arbitration_id=0x220,
        name="WHL_SPD11",
        signals=[
            CANSignalDefinition(
                name="vehicle_speed",
                start_bit=0,
                bit_length=14,
                scale=0.03125,
                offset=0.0,
                min_physical=0.0,
                max_physical=250.0,
                unit="km/h",
            ),
            CANSignalDefinition(
                name="brake_active",
                start_bit=15,
                bit_length=1,
                scale=1.0,
                offset=0.0,
                min_physical=0.0,
                max_physical=1.0,
                unit="flag",
            ),
        ],
    )
    db.add_message(msg_speed)

    # 4. Gear Position (0x43F / 1087 decimal)
    msg_gear = MessageDefinition(
        arbitration_id=0x43F,
        name="CLU11",
        signals=[
            CANSignalDefinition(
                name="gear_num",
                start_bit=0,
                bit_length=4,
                scale=1.0,
                offset=0.0,
                min_physical=0.0,
                max_physical=8.0,
                unit="gear",
            ),
        ],
    )
    db.add_message(msg_gear)

    return db


def create_chevrolet_impala_dbc() -> DBCDatabase:
    """Constructs DBC definitions for GM / 2011 Chevrolet Impala CAN bus."""
    db = DBCDatabase(name="chevrolet_impala")

    # 1. Engine RPM (0x0C9 / 201 decimal)
    msg_engine = MessageDefinition(
        arbitration_id=0x0C9,
        name="ECM_Engine",
        signals=[
            CANSignalDefinition(
                name="engine_rpm",
                start_bit=0,
                bit_length=16,
                scale=0.25,
                offset=0.0,
                min_physical=0.0,
                max_physical=8000.0,
                unit="rpm",
            ),
            CANSignalDefinition(
                name="throttle_pct",
                start_bit=24,
                bit_length=8,
                scale=0.4,
                offset=0.0,
                min_physical=0.0,
                max_physical=100.0,
                unit="%",
            ),
        ],
    )
    db.add_message(msg_engine)

    # 2. Steering Angle (0x1E5 / 485 decimal)
    msg_steering = MessageDefinition(
        arbitration_id=0x1E5,
        name="PSCM_Steering",
        signals=[
            CANSignalDefinition(
                name="steering_angle",
                start_bit=0,
                bit_length=16,
                is_signed=True,
                scale=0.0625,
                offset=0.0,
                min_physical=-500.0,
                max_physical=500.0,
                unit="deg",
            ),
        ],
    )
    db.add_message(msg_steering)

    # 3. Vehicle Speed & Brake (0x1F1 / 497 decimal)
    msg_speed = MessageDefinition(
        arbitration_id=0x1F1,
        name="EBCM_BrakeSpeed",
        signals=[
            CANSignalDefinition(
                name="vehicle_speed",
                start_bit=0,
                bit_length=16,
                scale=0.015625,
                offset=0.0,
                min_physical=0.0,
                max_physical=250.0,
                unit="km/h",
            ),
            CANSignalDefinition(
                name="brake_active",
                start_bit=16,
                bit_length=1,
                scale=1.0,
                offset=0.0,
                min_physical=0.0,
                max_physical=1.0,
                unit="flag",
            ),
            CANSignalDefinition(
                name="gear_num",
                start_bit=24,
                bit_length=4,
                scale=1.0,
                offset=0.0,
                min_physical=0.0,
                max_physical=8.0,
                unit="gear",
            ),
        ],
    )
    db.add_message(msg_speed)

    return db
