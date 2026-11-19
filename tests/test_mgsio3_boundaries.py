"""
Entropy and Gibbs energy across the MgSiO3 phase boundaries.

The reference constants (U0, S0) of each phase are set so that entropy rises on
heating across every boundary, the Clausius-Clapeyron relation dP/dT = dS/dV
holds against the analytic boundary slopes, the orthoenstatite melting entropy
matches the Stebbins et al. (1984) calorimetric value, and brg, ppv, and liquid
share one Gibbs energy at their triple point. The analytic boundaries and the
EoS of each phase come from separate fits, so the Clapeyron test allows a
factor of 2.5 either way; a sign error or an offset of the wrong size fails it.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from scipy.optimize import brentq

from paleos import mgsio3_eos as mg

M = mg._M_MgSiO3  # kg/mol
CLAPEYRON_FACTOR = 2.5


@dataclass(frozen=True)
class Boundary:
    """A solid-solid boundary P(T), with the phases below and above it in T."""

    name: str
    low: str
    high: str
    P_of_T: Callable[[float], float]
    T_min: float
    T_max: float

    def temperatures(self, n=200):
        return np.linspace(self.T_min, self.T_max, n)

    def slope(self, T, dT=0.5):
        return (self.P_of_T(T + dT) - self.P_of_T(T - dT)) / (2.0 * dT)


# The en-hpcen line ends where it meets the melting curve.
_T_EN_HPCEN_MELT = brentq(lambda T: mg.T_melt_MgSiO3(mg.P_en_hpcen(T)) - T, 1500.0, 4000.0)

SOLID_BOUNDARIES = [
    Boundary("lpcen-en", "solid-lpcen", "solid-en", mg.P_lpcen_en, 770.0, 1095.0),
    Boundary("lpcen-hpcen", "solid-lpcen", "solid-hpcen", mg.P_lpcen_hpcen, 300.0, 1095.0),
    Boundary("hpcen-en", "solid-hpcen", "solid-en", mg.P_en_hpcen, 1105.0, _T_EN_HPCEN_MELT - 5.0),
    Boundary("ppv-brg", "solid-ppv", "solid-brg", mg.P_brg_ppv, 300.0, mg._T_TRIPLE_MGSIO3 - 5.0),
]
IDS = [b.name for b in SOLID_BOUNDARIES]


def molar_entropy(eos, P, T):
    return eos.specific_entropy(P, T) * M


def molar_volume(eos, P, T):
    return M / eos.density(P, T)


def molar_gibbs(eos, P, T):
    return (eos.specific_internal_energy(P, T) - T * eos.specific_entropy(P, T)
            + P / eos.density(P, T)) * M


@pytest.mark.parametrize("b", SOLID_BOUNDARIES, ids=IDS)
def test_phase_map_heating_order(b):
    """Heating at fixed P crosses each boundary from the low-T to the high-T phase."""
    for T in b.temperatures(25):
        P = b.P_of_T(T)
        assert mg.get_mgsio3_phase(P, T - 1.0) == b.low, (P, T)
        assert mg.get_mgsio3_phase(P, T + 1.0) == b.high, (P, T)


@pytest.mark.parametrize("b", SOLID_BOUNDARIES, ids=IDS)
def test_entropy_rises_on_heating(b, mgsio3_phases):
    lo, hi = mgsio3_phases[b.low], mgsio3_phases[b.high]
    for T in b.temperatures():
        P = b.P_of_T(T)
        dS = molar_entropy(hi, P, T) - molar_entropy(lo, P, T)
        assert dS > 0.0, f"{b.name}: S falls by {-dS:.3f} J/(mol K) at {P:.4e} Pa, {T:.1f} K"


@pytest.mark.parametrize("b", SOLID_BOUNDARIES, ids=IDS)
def test_clapeyron_slope(b, mgsio3_phases):
    lo, hi = mgsio3_phases[b.low], mgsio3_phases[b.high]
    for T in b.temperatures():
        P = b.P_of_T(T)
        dS = molar_entropy(hi, P, T) - molar_entropy(lo, P, T)
        dV = molar_volume(hi, P, T) - molar_volume(lo, P, T)
        ratio = (dS / dV) / b.slope(T)
        assert 1.0 / CLAPEYRON_FACTOR < ratio < CLAPEYRON_FACTOR, (
            f"{b.name}: dS/dV is {ratio:.2f} times the boundary slope at {P:.4e} Pa, {T:.1f} K")


@pytest.mark.parametrize("a, b", [("solid-lpcen", "solid-en"), ("solid-lpcen", "solid-hpcen"),
                                  ("solid-en", "solid-hpcen")])
def test_pyroxene_entropies_meet_at_zero_kelvin(a, b, mgsio3_phases):
    """The three pyroxenes share one Sokolova et al. (2022) model, so the third law applies."""
    for P in (1e5, 6.5e9, 12e9):
        dS = molar_entropy(mgsio3_phases[a], P, 1.0) - molar_entropy(mgsio3_phases[b], P, 1.0)
        assert abs(dS) < 1e-2, f"S({a}) - S({b}) = {dS:.4f} J/(mol K) at {P:.1e} Pa, 1 K"


def test_entropy_rises_on_melting(mgsio3_phases):
    liquid = mgsio3_phases["liquid"]
    for P in np.logspace(5, 12, 141):
        T = mg.T_melt_MgSiO3(P)
        solid = mg.get_mgsio3_phase(P, T - 1.0)
        dS = molar_entropy(liquid, P, T) - molar_entropy(mgsio3_phases[solid], P, T)
        assert dS > 0.0, f"{solid} to liquid: S falls by {-dS:.3f} J/(mol K) at {P:.4e} Pa"


def test_stebbins_fusion_entropy(mgsio3_phases):
    P, T = 1e5, 1831.0
    dS = (molar_entropy(mgsio3_phases["liquid"], P, T)
          - molar_entropy(mgsio3_phases["solid-en"], P, T))
    assert dS == pytest.approx(41.99, abs=0.05)


def test_gibbs_continuity_at_deep_triple_point(mgsio3_phases):
    P, T = mg._P_TRIPLE_MGSIO3, mg._T_TRIPLE_MGSIO3
    G = [molar_gibbs(mgsio3_phases[p], P, T) for p in ("solid-brg", "solid-ppv", "liquid")]
    assert max(G) - min(G) < 1e-2, f"Gibbs spread {max(G) - min(G):.3e} J/mol"
