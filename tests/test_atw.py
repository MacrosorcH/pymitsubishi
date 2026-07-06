"""
Unit tests for the air-to-WATER (Mitsubishi Ecodan) protocol codec in
``pymitsubishi.mitsubishi_atw``.

All packets used here are REAL captures verified live against an FTC6 + PUZ-WM112
over the MELCloud adapter's local /smart API. No network access is performed.

Importing the full ``pymitsubishi`` package pulls in ``mitsubishi_api`` which
requires ``requests`` and ``pycryptodome`` (``Crypto``). ``mitsubishi_atw`` itself
only needs ``calc_fcc`` from the stdlib-only ``mitsubishi_parser``. To keep these
tests runnable with ``python3 -m pytest`` whether or not those heavy deps are
installed, we import the package normally when possible and otherwise load the two
modules directly via importlib, registering a stub ``pymitsubishi`` package in
``sys.modules`` so the ``from .mitsubishi_parser import calc_fcc`` relative import
inside ``mitsubishi_atw`` still resolves.
"""

import importlib
import importlib.util
import pathlib
import sys
import types

import pytest

# pymitsubishi/pymitsubishi/  (the inner package dir holding the modules)
_PKG_DIR = pathlib.Path(__file__).resolve().parent.parent / "pymitsubishi"


def _load_atw_module():
    """Return the mitsubishi_atw module, avoiding the package __init__ if its
    optional heavy dependencies (requests / pycryptodome) are unavailable."""
    # Fast path: real package import (works when deps are present).
    try:
        return importlib.import_module("pymitsubishi.mitsubishi_atw")
    except Exception:  # pragma: no cover - depends on the test environment
        pass

    # Fallback: register a stub package and exec the two stdlib-only modules
    # directly, bypassing pymitsubishi/__init__.py (which imports Crypto).
    if "pymitsubishi" not in sys.modules or not getattr(sys.modules["pymitsubishi"], "__path__", None):
        stub = types.ModuleType("pymitsubishi")
        stub.__path__ = [str(_PKG_DIR)]
        sys.modules["pymitsubishi"] = stub

    def _exec(name):
        full = f"pymitsubishi.{name}"
        spec = importlib.util.spec_from_file_location(full, _PKG_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    _exec("mitsubishi_parser")  # provides calc_fcc for the relative import
    return _exec("mitsubishi_atw")


atw = _load_atw_module()

EcodanStatus = atw.EcodanStatus
EcodanOperationMode = atw.EcodanOperationMode
ZoneMode = atw.ZoneMode
FTCVersion = atw.FTCVersion
is_atw_payload = atw.is_atw_payload
parse_atw_code_values = atw.parse_atw_code_values
generate_set_flow_setpoint_command = atw.generate_set_flow_setpoint_command


# --- Real captured packets (FTC6 + PUZ-WM112) ---------------------------------
CODE_SETPOINTS = "fc62027a100907d007d007080708177032965000009e"  # group 0x09
CODE_TEMPS_ROOM = "fc62027a100b0a280a28f0c40b09c40b7e000000008e"  # group 0x0B
CODE_TEMPS_FLOW = "fc62027a100c079e700802731162a909c40b00000080"  # group 0x0C
CODE_OPERATION = "fc62027a10260000000001040413880708070800002a"  # group 0x26
PROFILE_CODE = "fc7b027a10c903000100140300010100000000000013"  # 0xC9 profile

ALL_CODES = [CODE_SETPOINTS, CODE_TEMPS_ROOM, CODE_TEMPS_FLOW, CODE_OPERATION]

# An ATA (air-to-air) style packet: preamble 01 30, not 02 7a.
ATA_PACKET_HEX = "fc6201301002000000000000000000000000000000"


@pytest.fixture(scope="module")
def status() -> EcodanStatus:
    """Parse the full set of captured codes + profile once for the module."""
    return parse_atw_code_values(ALL_CODES, profile_code=PROFILE_CODE)


# --- Test 1: parse_atw_code_values decodes the real packets -------------------
class TestParseAtwCodeValues:
    def test_returns_ecodan_status(self, status):
        assert isinstance(status, EcodanStatus)

    def test_flow_setpoints_and_limits(self, status):
        assert status.zone1_flow_setpoint == 18.0
        assert status.zone2_flow_setpoint == 18.0
        assert status.legionella_setpoint == 60.0
        assert status.flow_temp_min == 20.0
        assert status.flow_temp_max == 55.0

    def test_room_and_outdoor_temperatures(self, status):
        assert status.zone1_room_temperature == 26.0
        assert status.zone2_room_temperature == 26.0
        assert status.outdoor_temperature == 23.0

    def test_flow_return_delta_and_dhw_tank(self, status):
        assert status.flow_temperature == 19.5
        assert status.return_temperature == 20.5
        assert status.delta_t == -1.0
        assert status.dhw_tank_temperature == 44.5

    def test_operation_group(self, status):
        assert status.power_on is False
        assert status.operation_mode == EcodanOperationMode.OFF
        assert status.dhw_eco is True
        assert status.zone1_mode == ZoneMode.COOL_FLOW
        assert status.zone2_mode == ZoneMode.COOL_FLOW
        assert status.dhw_setpoint == 50.0

    def test_profile_ftc_version(self, status):
        assert status.ftc_version == FTCVersion.FTC6

    def test_order_independent(self):
        """Decoding must not depend on the order codes arrive in."""
        st = parse_atw_code_values(list(reversed(ALL_CODES)), profile_code=PROFILE_CODE)
        assert st.zone1_flow_setpoint == 18.0
        assert st.operation_mode == EcodanOperationMode.OFF
        assert st.ftc_version == FTCVersion.FTC6

    def test_profile_optional(self):
        """Without a profile code, ftc_version stays unset; other fields parse."""
        st = parse_atw_code_values(ALL_CODES)
        assert st.ftc_version is None
        assert st.flow_temperature == 19.5


# --- Test 2: is_atw_payload distinguishes ATW from ATA ------------------------
class TestIsAtwPayload:
    def test_atw_packet_is_true(self):
        assert is_atw_payload(bytes.fromhex(CODE_TEMPS_FLOW)) is True

    def test_ata_packet_is_false(self):
        assert is_atw_payload(bytes.fromhex(ATA_PACKET_HEX)) is False

    def test_too_short_is_false(self):
        assert is_atw_payload(b"\xfc\x62") is False


# --- Test 3: command builder is byte-exact ------------------------------------
class TestGenerateSetFlowSetpointCommand:
    def test_byte_exact_cool_flow(self):
        cmd = generate_set_flow_setpoint_command(19.0, 19.0, 50.0, zone_mode=ZoneMode.COOL_FLOW)
        assert cmd.hex() == "fc41027a1032800201000104041388076c076c0000f4"

    def test_returns_22_byte_packet(self):
        cmd = generate_set_flow_setpoint_command(19.0, 19.0, 50.0)
        assert isinstance(cmd, bytes | bytearray)
        assert len(cmd) == 22
        # 0xfc framing + ATW 0x41 02 7a 10 header
        assert cmd[0] == 0xFC
        assert cmd[1] == 0x41
        assert cmd[2:4] == atw.ATW_PREAMBLE


# --- Test 4: checksum / round-trip invariant for every command builder --------
def _all_generated_commands():
    """Exercise every public command builder with representative arguments."""
    cmds = [
        atw.generate_set_flow_setpoint_command(19.0, 19.0, 50.0),
        atw.generate_set_flow_setpoint_command(
            18.5, 21.0, 48.0, power_on=False, dhw_eco=False, zone_mode=ZoneMode.HEAT_FLOW
        ),
        atw.generate_set_flow_setpoint_command(20.0, 20.0, 50.0, zone_mode=4),
        atw.generate_set_dhw_setpoint_command(50.0),
        atw.generate_set_zone_mode_command(ZoneMode.HEAT_FLOW, ZoneMode.COOL_FLOW),
        atw.generate_set_zone_mode_command(1, 4),
        atw.generate_set_power_command(True),
        atw.generate_set_power_command(False),
        atw.generate_forced_dhw_command(True),
        atw.generate_forced_dhw_command(False),
    ]
    return cmds


@pytest.mark.parametrize("cmd", _all_generated_commands())
def test_command_checksum_invariant(cmd):
    """Last byte must be the Mitsubishi FCC over bytes[1:-1]."""
    expected = (0x100 - sum(cmd[1:-1]) % 0x100) % 0x100
    assert cmd[-1] == expected


@pytest.mark.parametrize("cmd", _all_generated_commands())
def test_command_well_formed(cmd):
    """Every builder yields a 22-byte 0xfc-framed ATW SET packet."""
    assert len(cmd) == 22
    assert cmd[0] == 0xFC
    assert cmd[1] == atw.SET_REQUEST
    assert cmd[2:4] == atw.ATW_PREAMBLE
