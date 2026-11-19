"""Shared fixtures for the PALEOS test suite."""

import pytest

from paleos import mgsio3_eos


@pytest.fixture(scope="session")
def mgsio3_phases():
    """Per-phase MgSiO3 EoS instances, as the MgSiO3EoS wrapper dispatches them."""
    # One wrapper per session: building the RTpress liquid takes a few seconds.
    return mgsio3_eos.MgSiO3EoS()._phase_eos_map
