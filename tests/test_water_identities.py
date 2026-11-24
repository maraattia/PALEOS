"""
Thermodynamic identities for the water EoS.

Along an isobar inside one phase, C_P must be positive, the entropy must rise with temperature, C_P must equal T (ds/dT)_P, and the internal energy must follow the first law, du = T ds - P dv. PALEOS derives C_P from the AQUA density and adiabatic gradient, so the second and third rules tie the corrected entropy to those columns. The points cover ice Ih, liquid water, vapor, the Brown (2018) fluid, and Region 7 of AQUA (Mazevet et al. 2019), where the entropy carries the correction to the published fit, including the bands where AQUA blends Region 7 with ice X and with the fluid. In the fluid bands the blend weight changes along the isobar, which is where the constant part of the correction shows. The differences span about two AQUA grid cells on each side, so the tolerance is the one of the table check in paleos/checks.py.
"""

import numpy as np
import pytest

H = 0.05  # half-width of the stencil in ln T
REL = 0.1

# (label, P [Pa], T [K]): every stencil point lies in the labeled phase
POINTS = [
    ("ice-Ih", 1e5, 250.0), ("ice-Ih", 1e8, 200.0), ("liquid", 1e5, 330.0), ("vapor", 1e3, 500.0),
    ("supercritical", 2e9, 2000.0), ("ice-X", 150e9, 1800.0),
    ("ice-X-blend", 340e9, 500.0), ("ice-X-blend", 500e9, 400.0), ("ice-X-blend", 500e9, 1500.0), ("ice-X-blend", 620e9, 500.0),
    ("ice-X-M19", 1e12, 400.0), ("ice-X-M19", 1e12, 1000.0), ("ice-X-M19", 2.551e12, 600.0),
    ("ice-X-M19", 10e12, 500.0), ("ice-X-M19", 10e12, 2000.0), ("ice-X-M19", 50e12, 800.0),
    ("supercritical-blend", 45e9, 3000.0), ("supercritical-blend", 55e9, 2500.0), ("supercritical-blend", 1e9, 8000.0),
    ("supercritical-M19", 100e9, 5000.0), ("supercritical-M19", 1e12, 1e4), ("supercritical-M19", 10e12, 3e4),
]

BLEND57 = "the 5 to 7 band of AQUA moves with temperature, so its blended C_P misses T ds/dT by T (s_7 - s_5) d(theta)/dT"
CLAMP = "AQUA holds the M19 fit at its 300 K values below 300 K, so rho does not change with T there"

# point id -> (reason, tests that fail there)
KNOWN = {
    "supercritical-blend-45GPa-3000K": (BLEND57, ("cp",)),
    "supercritical-blend-55GPa-2500K": (BLEND57, ("cp",)),
}


def cases(test):
    out = []
    for label, P, T in POINTS:
        pid = f"{label}-{P / 1e9:g}GPa-{T:g}K"
        reason, tests = KNOWN.get(pid, (None, ()))
        marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)] if test in tests else []
        out.append(pytest.param(P, T, id=pid, marks=marks))
    return out


def stencil(water, P, T, f):
    return f(P, T * np.exp(H)) - f(P, T * np.exp(-H))


@pytest.mark.parametrize("P, T", cases("phase"))
def test_points_stay_in_one_phase(water, P, T):
    assert len({water.phase(P, T * np.exp(k * H)) for k in (-1, 0, 1)}) == 1


@pytest.mark.parametrize("P, T", cases("cp_positive") + [pytest.param(1e12, 200.0, id="ice-X-M19-1000GPa-200K", marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=CLAMP))])
def test_heat_capacity_is_positive(water, P, T):
    assert water.isobaric_heat_capacity(P, T) > 0


@pytest.mark.parametrize("P, T", cases("entropy"))
def test_entropy_rises_along_isobar(water, P, T):
    assert stencil(water, P, T, water.specific_entropy) > 0


@pytest.mark.parametrize("P", [1e12, 10e12, 50e12])
def test_entropy_rises_from_ice_x_into_the_fluid(water, P):
    # Dense in T so that queries fall between AQUA nodes, across the top of ice X near 2250 K
    s = [water.specific_entropy(P, T) for T in np.geomspace(300.0, 1e4, 600)]
    assert np.all(np.diff(s) > 0)


@pytest.mark.parametrize("P, T", cases("cp"))
def test_cp_matches_entropy(water, P, T):
    cp_s = stencil(water, P, T, water.specific_entropy) / (2 * H)
    assert water.isobaric_heat_capacity(P, T) == pytest.approx(cp_s, rel=REL)


@pytest.mark.parametrize("P, T", cases("first_law"))
def test_first_law(water, P, T):
    du = stencil(water, P, T, water.specific_internal_energy)
    T_ds = T * stencil(water, P, T, water.specific_entropy)
    P_dv = P * stencil(water, P, T, lambda p, t: 1 / water.density(p, t))
    assert abs(du - (T_ds - P_dv)) <= REL * (abs(T_ds) + abs(P_dv))
