"""Shared fixtures for the PALEOS test suite."""

import os
from pathlib import Path

import pytest

from paleos import mgsio3_eos


@pytest.fixture(scope="session")
def mgsio3_wrapper():
    """One MgSiO3EoS per session: building the RTpress liquid takes a few seconds."""
    return mgsio3_eos.MgSiO3EoS()


@pytest.fixture(scope="session")
def mgsio3_phases(mgsio3_wrapper):
    """Per-phase MgSiO3 EoS instances, as the MgSiO3EoS wrapper dispatches them."""
    return mgsio3_wrapper._phase_eos_map


@pytest.fixture(scope="session")
def iron_wrapper():
    from paleos import iron_eos

    return iron_eos.IronEoS()


@pytest.fixture(scope="session")
def iron_phases(iron_wrapper):
    """Per-phase iron EoS instances, as the IronEoS wrapper dispatches them."""
    return iron_wrapper._phase_eos_map


@pytest.fixture(scope="session")
def water():
    """WaterEoS on the AQUA table named by PALEOS_AQUA_TABLE, or in an AQUA checkout next to this repository. Skips if neither exists."""
    from paleos.water_eos import WaterEoS

    default = Path(__file__).resolve().parents[2] / "AQUA" / "Tables" / "aqua_eos_pt_v1_0.dat"
    path = Path(os.environ.get("PALEOS_AQUA_TABLE", default))
    if not path.is_file():
        pytest.skip(f"AQUA table not found at {path}; set PALEOS_AQUA_TABLE")
    return WaterEoS(str(path))
