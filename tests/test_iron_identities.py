"""
Thermodynamic identities for every iron EoS.

Each per-phase EoS must describe one material: alpha must equal -(d ln rho/dT)_P, the Maxwell relation (ds/dP)_T = -alpha/rho must hold with the tabulated alpha, c_p must equal T (ds/dT)_P, nabla_ad must equal P alpha/(rho c_p), and u, s, and rho must come from one Gibbs energy (dG/dP = V, dG/dT = -S). The points cover every class behind the IronEoS wrapper, including the Miozzi20 to Hakim18 blend and the liquid below its melting curve, which two-phase tables need. The strict expected failures record known inconsistencies.
"""

import numpy as np
import pytest

from paleos import iron_eos as fe

M = fe.ATOMIC_MASS_FE  # kg/mol

# (phase whose EoS is evaluated, P [Pa], T [K]). The phase need not be stable there.
POINTS = [
    ("solid-alpha", 1e5, 300.0), ("solid-alpha", 1e9, 800.0), ("solid-delta", 1e9, 1750.0),
    ("solid-gamma", 5e9, 1500.0), ("solid-gamma", 20e9, 2000.0),
    ("solid-epsilon", 50e9, 1500.0), ("solid-epsilon", 100e9, 2500.0),
    ("solid-epsilon", 250e9, 5000.0), ("solid-epsilon", 292.86e9, 5868.0), ("solid-epsilon", 370e9, 4000.0),
    ("solid-epsilon", 500e9, 5000.0), ("solid-epsilon", 3e12, 5000.0),
    ("liquid", 10e9, 3000.0), ("liquid", 100e9, 5000.0), ("liquid", 1e12, 12000.0),
    ("liquid", 3e12, 25000.0), ("liquid", 5e12, 30000.0),
    ("liquid", 3e12, 5000.0), ("liquid", 5e12, 5000.0), ("liquid", 10e12, 5000.0), ("liquid", 50e12, 1000.0),
]


def point_id(phase, P, T):
    return f"{phase}-{P / 1e9:g}GPa-{T:g}K"


BLEND = "the Miozzi20 to Hakim18 blend mixes the seven quantities directly"

# point id -> (reason, tests that fail there)
KNOWN = {
    "solid-epsilon-250GPa-5000K": (BLEND, ("nabla_ad", "alpha", "maxwell", "gibbs")),
    "solid-epsilon-292.86GPa-5868K": (BLEND, ("nabla_ad", "alpha", "maxwell", "gibbs")),
    "solid-epsilon-370GPa-4000K": (BLEND, ("nabla_ad", "alpha", "maxwell", "gibbs")),
}


def cases(test):
    out = []
    for phase, P, T in POINTS:
        pid = point_id(phase, P, T)
        reason, tests = KNOWN.get(pid, (None, ()))
        marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)] if test in tests else []
        out.append(pytest.param(phase, P, T, id=pid, marks=marks))
    return out


def gibbs(eos, P, T):
    return eos.specific_internal_energy(P, T) - T * eos.specific_entropy(P, T) + P / eos.density(P, T)


def d_dT(f, P, T, h=1e-3):
    dT = h * T
    return (f(P, T + dT) - f(P, T - dT)) / (2.0 * dT)


def d_dP(f, P, T, h=1e-4):
    dP = h * P
    return (f(P + dP, T) - f(P - dP, T)) / (2.0 * dP)


@pytest.mark.parametrize("phase, P, T", cases("nabla_ad"))
def test_adiabatic_gradient_matches_alpha_rho_cp(phase, P, T, iron_phases):
    eos = iron_phases[phase]
    ref = P * eos.thermal_expansion(P, T) / (eos.density(P, T) * eos.isobaric_heat_capacity(P, T))
    assert eos.adiabatic_gradient(P, T) == pytest.approx(ref, rel=1e-6, abs=0.0)


@pytest.mark.parametrize("phase, P, T", cases("alpha"))
def test_alpha_matches_density(phase, P, T, iron_phases):
    eos = iron_phases[phase]
    alpha_rho = -d_dT(lambda p, t: np.log(eos.density(p, t)), P, T)
    assert eos.thermal_expansion(P, T) == pytest.approx(alpha_rho, rel=1e-5, abs=0.0)


@pytest.mark.parametrize("phase, P, T", cases("maxwell"))
def test_maxwell_relation_with_tabulated_alpha(phase, P, T, iron_phases):
    eos = iron_phases[phase]
    dsdP = d_dP(eos.specific_entropy, P, T)
    assert dsdP == pytest.approx(-eos.thermal_expansion(P, T) / eos.density(P, T), rel=1e-5, abs=0.0)


@pytest.mark.parametrize("phase, P, T", cases("cp"))
def test_cp_matches_entropy(phase, P, T, iron_phases):
    eos = iron_phases[phase]
    assert T * d_dT(eos.specific_entropy, P, T) == pytest.approx(eos.isobaric_heat_capacity(P, T), rel=1e-5, abs=0.0)


@pytest.mark.parametrize("phase, P, T", cases("gibbs"))
def test_gibbs_derivatives_match_volume_and_entropy(phase, P, T, iron_phases):
    eos = iron_phases[phase]
    V = 1.0 / eos.density(P, T)
    S = eos.specific_entropy(P, T)
    g = lambda p, t: gibbs(eos, p, t)
    assert d_dP(g, P, T) == pytest.approx(V, rel=1e-6, abs=0.0), "dG/dP != V"
    assert abs(d_dT(g, P, T, h=1e-4) + S) * M < 1e-3, "dG/dT != -S"


def test_liquid_extrapolation_joins_the_fit_smoothly():
    """beta, its volume derivative, the entropy, and the energy are continuous at the 1.3 TPa calibration edge."""
    liquid = fe.Luo24()
    V_above, V_below = liquid.V_edge * (1 + 1e-10), liquid.V_edge * (1 - 1e-10)
    assert liquid._beta_and_derivative(V_below) == pytest.approx(liquid._beta_and_derivative(V_above), rel=1e-6, abs=0.0)
    assert liquid._thermal_entropy(V_below) == pytest.approx(liquid._thermal_entropy(V_above), abs=1e-6)
    for T in (3000.0, 20000.0):
        u_above, u_below = liquid._internal_energy(V_above, T), liquid._internal_energy(V_below, T)
        assert u_below == pytest.approx(u_above, rel=1e-9, abs=0.0)


def test_liquid_alpha_is_positive_over_the_table_box():
    """Two-phase tables need the liquid at every (P, T) of the iron tables, also below melting."""
    liquid = fe.Luo24()
    for P in np.logspace(5, 14, 19):
        for T in np.logspace(np.log10(300.0), 5, 9):
            assert liquid.thermal_expansion(P, T) > 0.0, (P, T)
