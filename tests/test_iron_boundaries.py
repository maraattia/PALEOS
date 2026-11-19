"""
Entropy and Gibbs energy across the iron phase boundaries.

The reference constants (U0, S0) are set so that entropy rises on heating across every solid boundary and on melting, and so that fcc, hcp, and liquid share one Gibbs energy at their triple point (98.5 GPa, 3712 K). Hakim18 is tied to Miozzi20 at the center of their blend, 310 GPa on the gamma-epsilon line, where s and u must match. The analytic boundaries and the EoS of each phase come from separate fits, so the Clausius-Clapeyron test allows a factor of 2.5 either way. The strict expected failures record known inconsistencies.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from scipy.optimize import brentq

from paleos import iron_eos as fe

M = fe.ATOMIC_MASS_FE  # kg/mol
P_TRIPLE, T_TRIPLE = 98.5e9, 3712.0
# The delta field closes where the delta-gamma line meets the melting curve.
P_DELTA_MAX = brentq(lambda P: fe.T_delta_gamma(P) - fe.T_melt_Fe(P), 1e9, 5.2e9)
# Melting pressures up to the 1e5 K top of the tables.
MELT_PRESSURES = np.logspace(5, np.log10(4e13), 161)


@dataclass(frozen=True)
class Boundary:
    """A solid-solid boundary T(P), with the phases below and above it in T."""

    name: str
    low: str
    high: str
    T_of_P: Callable[[float], float]
    P_min: float
    P_max: float

    def pressures(self, n=60):
        return np.linspace(self.P_min, self.P_max, n + 2)[1:-1]


SOLID_BOUNDARIES = [
    Boundary("alpha-gamma", "solid-alpha", "solid-gamma", fe.T_alpha_gamma, 1e5, 7.3e9),
    Boundary("gamma-delta", "solid-gamma", "solid-delta", fe.T_delta_gamma, 1e5, P_DELTA_MAX),
    Boundary("alpha-epsilon", "solid-alpha", "solid-epsilon", fe.T_alpha_epsilon, 7.3e9, 15.8e9),
    Boundary("epsilon-gamma", "solid-epsilon", "solid-gamma", fe.T_gamma_epsilon, 7.3e9, P_TRIPLE),
]
IDS = [b.name for b in SOLID_BOUNDARIES]


def molar_entropy(eos, P, T):
    return eos.specific_entropy(P, T) * M


def molar_volume(eos, P, T):
    return M / eos.density(P, T)


def molar_gibbs(eos, P, T):
    return (eos.specific_internal_energy(P, T) - T * eos.specific_entropy(P, T)
            + P / eos.density(P, T)) * M


def clapeyron_ratio(lo, hi, T_of_P, P):
    """(dS/dV) times dT/dP of the boundary, which Clausius-Clapeyron sets to 1."""
    T = T_of_P(P)
    dS = molar_entropy(hi, P, T) - molar_entropy(lo, P, T)
    dV = molar_volume(hi, P, T) - molar_volume(lo, P, T)
    dTdP = (T_of_P(P * 1.001) - T_of_P(P * 0.999)) / (0.002 * P)
    return dS / dV * dTdP


@pytest.mark.parametrize("b", SOLID_BOUNDARIES, ids=IDS)
def test_phase_map_heating_order(b):
    """Heating at fixed P crosses each boundary from the low-T to the high-T phase."""
    for P in b.pressures(25):
        T = b.T_of_P(P)
        assert fe.get_iron_phase(P, T - 1.0) == b.low, (P, T)
        assert fe.get_iron_phase(P, T + 1.0) == b.high, (P, T)


def with_xfails(xfails):
    return [pytest.param(b, id=b.name, marks=[pytest.mark.xfail(strict=True, raises=AssertionError, reason=xfails[b.name])]
                         if b.name in xfails else []) for b in SOLID_BOUNDARIES]


@pytest.mark.parametrize("b", SOLID_BOUNDARIES, ids=IDS)
def test_entropy_rises_on_heating(b, iron_phases):
    lo, hi = iron_phases[b.low], iron_phases[b.high]
    for P in b.pressures():
        T = b.T_of_P(P)
        dS = molar_entropy(hi, P, T) - molar_entropy(lo, P, T)
        assert dS > 0.0, f"{b.name}: S falls by {-dS:.3f} J/(mol K) at {P:.4e} Pa, {T:.1f} K"


CLAPEYRON_XFAIL = {
    "alpha-epsilon": "dS/dV is 0.28 to 1.54 times the slope of the alpha-epsilon line",
}


@pytest.mark.parametrize("b", with_xfails(CLAPEYRON_XFAIL))
def test_clapeyron_slope(b, iron_phases):
    lo, hi = iron_phases[b.low], iron_phases[b.high]
    for P in b.pressures():
        ratio = clapeyron_ratio(lo, hi, b.T_of_P, P)
        assert 1.0 / 2.5 < ratio < 2.5, f"{b.name}: dS/dV is {ratio:.2f} times the boundary slope at {P:.4e} Pa"


def test_entropy_rises_on_melting(iron_phases):
    liquid = iron_phases["liquid"]
    for P in MELT_PRESSURES:
        T = fe.T_melt_Fe(P)
        solid = fe.get_iron_phase(P, T - 1.0)
        dS = molar_entropy(liquid, P, T) - molar_entropy(iron_phases[solid], P, T)
        assert dS > 0.0, f"{solid} to liquid: S falls by {-dS:.3f} J/(mol K) at {P:.4e} Pa"


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="the Anzellini et al. (2013) curve is not the G_liq = G_solid line of the EoS; liquid Luo24 is denser than bcc at 1 bar")
def test_clapeyron_on_melting_curve(iron_phases):
    liquid = iron_phases["liquid"]
    for P in MELT_PRESSURES:
        solid = iron_phases[fe.get_iron_phase(P, fe.T_melt_Fe(P) - 1.0)]
        ratio = clapeyron_ratio(solid, liquid, fe.T_melt_Fe, P)
        assert 1.0 / 2.5 < ratio < 2.5, f"melting: dS/dV is {ratio:.2f} times dP/dT at {P:.4e} Pa"


def test_gibbs_continuity_at_triple_point(iron_phases):
    G = [molar_gibbs(iron_phases[p], P_TRIPLE, T_TRIPLE) for p in ("solid-gamma", "solid-epsilon", "liquid")]
    assert max(G) - min(G) < 1e-2, f"Gibbs spread {max(G) - min(G):.3e} J/mol"


def test_hakim18_matches_miozzi20_at_blend_center(iron_phases):
    hcp = iron_phases["solid-epsilon"]
    P = hcp.P_center
    T = fe.T_gamma_epsilon(P)
    dS = molar_entropy(hcp.eos_high, P, T) - molar_entropy(hcp.eos_low, P, T)
    dU = (hcp.eos_high.specific_internal_energy(P, T) - hcp.eos_low.specific_internal_energy(P, T)) * M
    assert abs(dS) < 1e-2, f"S_H18 - S_M20 = {dS:.3f} J/(mol K)"
    assert abs(dU) < 1.0, f"U_H18 - U_M20 = {dU:.1f} J/mol"


@pytest.mark.parametrize("phase, P", [
    ("solid-alpha", 1e5), ("solid-gamma", 1e5),
    pytest.param("solid-epsilon", 50e9, marks=pytest.mark.xfail(
        strict=True, raises=AssertionError, reason="hcp S0 is lifted by +8 J/(mol K) to keep dS > 0 on the alpha-epsilon line")),
])
def test_third_law(phase, P, iron_phases):
    assert abs(molar_entropy(iron_phases[phase], P, 1.0)) < 1e-1
