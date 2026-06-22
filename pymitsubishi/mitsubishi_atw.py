#!/usr/bin/env python3
"""
Mitsubishi Ecodan air-to-WATER (ATW) protocol parser & command builder.

Counterpart to ``mitsubishi_parser.py`` (air-to-air). ATW units share the same
MAC-577IF /smart transport, AES key and 0xfc framing, but are distinguished by
the ``0x02 0x7a`` preamble (ATA uses ``0x01 0x30``) and carry a different set of
"group" payloads (the FTC / CN105 protocol).

Field decoding follows the community Ecodan CN105 maps (F1p / m000c400) and has
been verified live against an FTC6 + PUZ-WM112 over the MELCloud adapter's local
/smart API. Command framing (SET 0x32) is verified — the generated cool-flow
packet is byte-identical to one accepted by the unit.
"""

from __future__ import annotations

import dataclasses
import enum
import logging

from .mitsubishi_parser import calc_fcc

logger = logging.getLogger(__name__)

ATW_PREAMBLE = b"\x02\x7a"
SET_REQUEST = 0x41


class EcodanOperationMode(enum.Enum):
    OFF = 0
    HOT_WATER = 1
    HEATING = 2
    COOLING = 3
    FROST_PROTECT = 5
    LEGIONELLA = 6
    HEATING_ECO = 7


class ZoneMode(enum.Enum):
    HEAT_TARGET = 0
    HEAT_FLOW = 1
    HEAT_COMPENSATION = 2
    COOL_TARGET = 3
    COOL_FLOW = 4
    DRY_UP = 5
    COOL_COMPENSATION = 6


class FTCVersion(enum.Enum):
    FTC2B = 0
    FTC4 = 1
    FTC5 = 2
    FTC6 = 3
    FTC7 = 5


# --- SET 0x32 (Set Options) flags ---------------------------------------------
SET_SYSTEM_POWER = 0x0001
SET_HOT_WATER_MODE = 0x0004
SET_HEATING_MODE_Z1 = 0x0008
SET_HEATING_MODE_Z2 = 0x0010
SET_HOT_WATER_SETPOINT = 0x0020
SET_ZONE1_SETPOINT = 0x0080
SET_ZONE2_SETPOINT = 0x0200


def _enum(enum_cls, value):
    try:
        return enum_cls(value)
    except ValueError:
        return value


def _u16(p, i):
    return (p[i] << 8) | p[i + 1]


def _s16(p, i):
    v = _u16(p, i)
    return v - 0x10000 if v >= 0x8000 else v


def _t100(p, i):
    return _u16(p, i) / 100.0


def _s100(p, i):
    return _s16(p, i) / 100.0


def _energy(p, i):
    return _u16(p, i) + p[i + 2] / 100.0


@dataclasses.dataclass
class EcodanStatus:
    """Complete parsed Ecodan (air-to-water) state."""

    # identity / capability
    ftc_version: FTCVersion | int | None = None
    ftc_software: str = ""
    has_cooling: bool | None = None
    # operation
    power_on: bool | None = None
    operation_mode: EcodanOperationMode | int | None = None
    zone1_mode: ZoneMode | int | None = None
    zone2_mode: ZoneMode | int | None = None
    dhw_eco: bool | None = None
    heat_cool: str | None = None
    defrost: int | None = None
    dhw_active: bool | None = None
    # temperatures
    flow_temperature: float | None = None
    return_temperature: float | None = None
    delta_t: float | None = None
    outdoor_temperature: float | None = None
    zone1_room_temperature: float | None = None
    zone2_room_temperature: float | None = None
    dhw_tank_temperature: float | None = None
    mixing_tank_temperature: float | None = None
    condensing_temperature: float | None = None
    refrigerant_temperature: float | None = None
    # setpoints
    zone1_flow_setpoint: float | None = None
    zone2_flow_setpoint: float | None = None
    dhw_setpoint: float | None = None
    legionella_setpoint: float | None = None
    flow_temp_min: float | None = None
    flow_temp_max: float | None = None
    # plant
    compressor_frequency: int | None = None
    compressor_running: bool | None = None
    run_hours: int | None = None
    input_power_kw: int | None = None
    output_power_kw: int | None = None
    primary_flow_rate_lmin: int | None = None
    primary_pump: bool | None = None
    booster_heater: bool | None = None
    immersion_heater: bool | None = None
    # prohibits / flags
    forced_dhw: bool | None = None
    holiday_mode: bool | None = None
    prohibit_dhw: bool | None = None
    prohibit_heating_zone1: bool | None = None
    prohibit_cooling_zone1: bool | None = None
    prohibit_heating_zone2: bool | None = None
    prohibit_cooling_zone2: bool | None = None
    server_control_mode: bool | None = None
    # demand
    zone1_demand: int | None = None
    zone2_demand: int | None = None
    # energy (today's running totals, kWh)
    consumed_heating_kwh: float | None = None
    consumed_cooling_kwh: float | None = None
    consumed_dhw_kwh: float | None = None
    delivered_heating_kwh: float | None = None
    delivered_cooling_kwh: float | None = None
    delivered_dhw_kwh: float | None = None
    # transport identity (from /smart wrapper)
    mac: str = ""
    serial: str = ""
    rssi: str = ""

    @property
    def cop_today(self) -> float | None:
        cons = sum(v for v in (self.consumed_heating_kwh, self.consumed_cooling_kwh, self.consumed_dhw_kwh) if v)
        deliv = sum(v for v in (self.delivered_heating_kwh, self.delivered_cooling_kwh, self.delivered_dhw_kwh) if v)
        return round(deliv / cons, 2) if cons else None


def is_atw_payload(data: bytes) -> bool:
    """True if a 0xfc-framed packet is an ATW (Ecodan) get-response."""
    return len(data) >= 6 and data[1] in (0x62, 0x7B) and data[2:4] == ATW_PREAMBLE


def parse_atw_code_values(code_values: list[str], profile_code: str | None = None) -> EcodanStatus:
    """Parse the CODE/VALUE groups from a /smart status response into EcodanStatus."""
    st = EcodanStatus()
    for hexv in code_values:
        try:
            data = bytes.fromhex(hexv)
            if not is_atw_payload(data):
                continue
            if calc_fcc(data[1:-1]) != data[-1]:
                logger.warning("ATW checksum mismatch, ignoring: %s", hexv)
                continue
            _parse_group(data, st)
        except (ValueError, IndexError) as e:
            logger.warning("Failed to parse ATW code value %s: %s", hexv, e)
    if profile_code:
        try:
            _parse_profile(bytes.fromhex(profile_code), st)
        except (ValueError, IndexError) as e:
            logger.warning("Failed to parse ATW profile code: %s", e)
    return st


def _parse_group(data: bytes, st: EcodanStatus) -> None:
    p = data[5:21]  # group byte + 15 data bytes (== F1p "Buffer")
    g = p[0]
    if g == 0x01:
        st.ftc_software = f"{p[7]:02X}.{p[8]:02X}"
    elif g == 0x02:
        st.defrost = p[3]
    elif g == 0x04:
        st.compressor_frequency = p[1]
    elif g == 0x05:
        st.dhw_active = bool(p[7])
    elif g == 0x07:
        st.input_power_kw = p[4]
        st.output_power_kw = p[6]
    elif g == 0x09:
        st.zone1_flow_setpoint = _t100(p, 5)
        st.zone2_flow_setpoint = _t100(p, 7)
        st.legionella_setpoint = _t100(p, 9)
        st.flow_temp_max = (p[12] - 40) / 2
        st.flow_temp_min = (p[13] - 40) / 2
    elif g == 0x0B:
        st.zone1_room_temperature = _t100(p, 1) if p[1] != 0xF0 else None
        st.zone2_room_temperature = _t100(p, 3) if p[3] != 0xF0 else None
        st.refrigerant_temperature = _s100(p, 8)
        st.outdoor_temperature = p[11] / 2 - 40.0
    elif g == 0x0C:
        flow = _t100(p, 1)
        ret = _t100(p, 4)
        st.flow_temperature = flow
        st.return_temperature = ret
        st.delta_t = round(flow - ret, 2)
        st.dhw_tank_temperature = _t100(p, 7)
    elif g == 0x0F:
        st.mixing_tank_temperature = _t100(p, 1)
        st.condensing_temperature = _s100(p, 4)
    elif g == 0x10:
        st.zone1_demand = p[1]
        st.zone2_demand = p[2]
    elif g == 0x11:
        st.has_cooling = bool(p[3] & 0x08)  # SW2-4
    elif g == 0x13:
        st.compressor_running = bool(p[1])
        st.run_hours = (p[4] << 8 | p[5]) * 100 + p[3]
    elif g == 0x14:
        st.booster_heater = bool(p[2] or p[3])
        st.immersion_heater = bool(p[5])
        st.primary_flow_rate_lmin = p[12]
    elif g == 0x15:
        st.primary_pump = bool(p[1])
    elif g == 0x26:
        st.power_on = p[3] == 1
        st.operation_mode = _enum(EcodanOperationMode, p[4])
        st.dhw_eco = p[5] == 1
        st.zone1_mode = _enum(ZoneMode, p[6])
        st.zone2_mode = _enum(ZoneMode, p[7])
        st.dhw_setpoint = _t100(p, 8)
    elif g == 0x28:
        st.forced_dhw = bool(p[3])
        st.holiday_mode = bool(p[4])
        st.prohibit_dhw = bool(p[5])
        st.prohibit_heating_zone1 = bool(p[6])
        st.prohibit_cooling_zone1 = bool(p[7])
        st.prohibit_heating_zone2 = bool(p[8])
        st.prohibit_cooling_zone2 = bool(p[9])
        st.server_control_mode = bool(p[10])
    elif g == 0x29:
        st.heat_cool = "Cool" if p[3] == 1 else "Heat"
    elif g == 0xA1:
        st.consumed_heating_kwh = _energy(p, 4)
        st.consumed_cooling_kwh = _energy(p, 7)
        st.consumed_dhw_kwh = _energy(p, 10)
    elif g == 0xA2:
        st.delivered_heating_kwh = _energy(p, 4)
        st.delivered_cooling_kwh = _energy(p, 7)
        st.delivered_dhw_kwh = _energy(p, 10)


def _parse_profile(data: bytes, st: EcodanStatus) -> None:
    """0xC9 extended-connect response: FTC version / refrigerant type."""
    if len(data) < 12 or data[2:4] != ATW_PREAMBLE:
        return
    p = data[5:21]
    if p[0] == 0xC9:
        st.ftc_version = _enum(FTCVersion, p[6])


# --- command construction -----------------------------------------------------
def _atw_set_packet(payload: bytes) -> bytes:
    """Frame a 16-byte SET payload (payload[0]=sub-command) into a full packet."""
    if len(payload) != 16:
        raise ValueError("ATW SET payload must be 16 bytes")
    body = bytes([SET_REQUEST]) + ATW_PREAMBLE + b"\x10" + payload
    return b"\xfc" + body + bytes([calc_fcc(body)])


def generate_set_flow_setpoint_command(
    zone1_setpoint: float,
    zone2_setpoint: float,
    dhw_setpoint: float,
    *,
    power_on: bool = True,
    dhw_eco: bool = True,
    zone_mode: ZoneMode | int = ZoneMode.COOL_FLOW,
) -> bytes:
    """SET 0x32 — change ONLY the Z1/Z2 flow setpoints (flags 0x80|0x02).

    All other bytes carry the unit's CURRENT values so nothing else moves even if
    the FTC acts on an unflagged field. DHW setpoint MUST be the current value
    (anti-zero). VERIFIED on FTC6: matches a packet the unit accepted.
    """
    z1 = int(round(zone1_setpoint * 100))
    z2 = int(round(zone2_setpoint * 100))
    d = int(round(dhw_setpoint * 100))
    m = zone_mode.value if isinstance(zone_mode, ZoneMode) else zone_mode
    payload = bytes(
        [
            0x32,
            0x80,
            0x02,
            1 if power_on else 0,
            0x00,
            1 if dhw_eco else 0,
            m,
            m,
            (d >> 8) & 0xFF,
            d & 0xFF,
            (z1 >> 8) & 0xFF,
            z1 & 0xFF,
            (z2 >> 8) & 0xFF,
            z2 & 0xFF,
            0,
            0,
        ]
    )
    return _atw_set_packet(payload)


def generate_set_dhw_setpoint_command(dhw_setpoint: float) -> bytes:
    """SET 0x32 — hot-water tank setpoint (flag 0x20). Spec-derived (untested)."""
    d = int(round(dhw_setpoint * 100))
    flags = SET_HOT_WATER_SETPOINT
    payload = bytes(
        [0x32, flags & 0xFF, (flags >> 8) & 0xFF, 0, 0, 0, 0, 0, (d >> 8) & 0xFF, d & 0xFF, 0, 0, 0, 0, 0, 0]
    )
    return _atw_set_packet(payload)


def generate_set_zone_mode_command(zone1_mode: ZoneMode | int, zone2_mode: ZoneMode | int) -> bytes:
    """SET 0x32 — heating/cooling control mode per zone (flags 0x08|0x10). Spec-derived (untested)."""
    m1 = zone1_mode.value if isinstance(zone1_mode, ZoneMode) else zone1_mode
    m2 = zone2_mode.value if isinstance(zone2_mode, ZoneMode) else zone2_mode
    flags = SET_HEATING_MODE_Z1 | SET_HEATING_MODE_Z2
    payload = bytes([0x32, flags & 0xFF, (flags >> 8) & 0xFF, 0, 0, 0, m1, m2, 0, 0, 0, 0, 0, 0, 0, 0])
    return _atw_set_packet(payload)


def generate_set_power_command(power_on: bool) -> bytes:
    """SET 0x32 — system power (flag 0x01). Spec-derived (untested)."""
    payload = bytes([0x32, SET_SYSTEM_POWER, 0, 1 if power_on else 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return _atw_set_packet(payload)


def generate_forced_dhw_command(on: bool) -> bytes:
    """SET 0x34 — force DHW (boost). Flag 0x01, byte[3]. Spec-derived (untested)."""
    payload = bytes([0x34, 0x01, 0, 1 if on else 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return _atw_set_packet(payload)
