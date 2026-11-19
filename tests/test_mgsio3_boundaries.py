"""
Entropy and Gibbs energy across the MgSiO3 phase boundaries.

The reference constants (U0, S0) are set so that entropy rises on heating across every solid boundary and along the melting curve, and so that the Clausius-Clapeyron ratio dS/dV agrees with the slope of the three pyroxene lines and the brg-ppv line. The analytic boundaries and the EoS of each phase come from separate fits, so the Clapeyron test allows a factor of 1.7 either way, and 2.5 on the OrthoEn to HP-CEn line above 2000 K, where Sokolova et al. (2022) did not compute it. The orthoenstatite melting entropy must match Stebbins et al. (1984). Brg, ppv, and liquid must share one Gibbs energy at their triple point, and G_liq = G_en at the Stebbins point ties the pyroxene energies to the liquid. Every phase must satisfy dG/dP = V and dG/dT = -S. The strict expected failures record known inconsistencies.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from scipy.optimize import brentq

from paleos import mgsio3_eos as mg

M = mg._M_MgSiO3  # kg/mol


@dataclass(frozen=True)
class Boundary:
    """A solid-solid boundary P(T), with the phases below and above it in T."""

    name: str
    low: str
    high: str
    P_of_T: Callable[[float], float]
    T_min: float
    T_max: float
    clapeyron_factor: float = 1.7

    def temperatures(self, n=200):
        return np.linspace(self.T_min, self.T_max, n)

    def slope(self, T, dT=0.5):
        return (self.P_of_T(T + dT) - self.P_of_T(T - dT)) / (2.0 * dT)


# The en-hpcen line ends where it meets the melting curve.
_T_EN_HPCEN_MELT = brentq(lambda T: mg.T_melt_MgSiO3(mg.P_en_hpcen(T)) - T, 1500.0, 4000.0)

SOLID_BOUNDARIES = [
    Boundary("lpcen-en", "solid-lpcen", "solid-en", mg.P_lpcen_en, 770.0, 1095.0),
    Boundary("lpcen-hpcen", "solid-lpcen", "solid-hpcen", mg.P_lpcen_hpcen, 300.0, 1095.0),
    Boundary("hpcen-en", "solid-hpcen", "solid-en", mg.P_en_hpcen, 1105.0, 2000.0),
    # Extrapolated past the 2000 K limit of Sokolova et al. (2022), where dS/dV drifts to 2.3 times the slope.
    Boundary("hpcen-en-extrapolated", "solid-hpcen", "solid-en", mg.P_en_hpcen, 2000.0,
             _T_EN_HPCEN_MELT - 5.0, clapeyron_factor=2.5),
    Boundary("ppv-brg", "solid-ppv", "solid-brg", mg.P_brg_ppv, 300.0, mg._T_TRIPLE_MGSIO3 - 5.0),
]
IDS = [b.name for b in SOLID_BOUNDARIES]
MELT_PRESSURES = np.logspace(5, 14, 181)


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
        assert 1.0 / b.clapeyron_factor < ratio < b.clapeyron_factor, (
            f"{b.name}: dS/dV is {ratio:.2f} times the boundary slope at {P:.4e} Pa, {T:.1f} K")


@pytest.mark.parametrize("a, b", [("solid-lpcen", "solid-en"), ("solid-lpcen", "solid-hpcen"),
                                  ("solid-en", "solid-hpcen")])
def test_pyroxenes_share_the_sokolova_entropy_zero(a, b, mgsio3_phases):
    """The three pyroxenes share one Sokolova et al. (2022) model, so their entropies coincide at 0 K."""
    for P in (1e5, 6.5e9, 12e9):
        dS = molar_entropy(mgsio3_phases[a], P, 1.0) - molar_entropy(mgsio3_phases[b], P, 1.0)
        assert abs(dS) < 1e-2, f"S({a}) - S({b}) = {dS:.4f} J/(mol K) at {P:.1e} Pa, 1 K"


def test_entropy_rises_on_melting(mgsio3_phases):
    liquid = mgsio3_phases["liquid"]
    for P in MELT_PRESSURES:
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


def test_gibbs_continuity_at_one_bar_melting(mgsio3_phases):
    """The common pyroxene U0 shift sets G_liq = G_en at the Stebbins et al. (1984) point."""
    P, T = 1e5, 1831.0
    dG = molar_gibbs(mgsio3_phases["liquid"], P, T) - molar_gibbs(mgsio3_phases["solid-en"], P, T)
    assert abs(dG) < 1.0, f"G_liq - G_en = {dG:.1f} J/mol at 1 bar, 1831 K"


# Measured maxima up to 2000 K are 70, 41, and 124 J/mol; a common U0 shift cancels on these lines.
PYROXENE_GIBBS_TOL = {"lpcen-en": 150.0, "lpcen-hpcen": 100.0, "hpcen-en": 250.0}
PYROXENE_BOUNDARIES = [b for b in SOLID_BOUNDARIES if b.name in PYROXENE_GIBBS_TOL]


@pytest.mark.parametrize("b", PYROXENE_BOUNDARIES, ids=[b.name for b in PYROXENE_BOUNDARIES])
def test_gibbs_continuity_across_pyroxene_boundaries(b, mgsio3_phases):
    """The pyroxene EoS and boundaries both come from Sokolova et al. (2022), so G matches on the boundaries up to 2000 K."""
    lo, hi = mgsio3_phases[b.low], mgsio3_phases[b.high]
    for T in b.temperatures(25):
        P = b.P_of_T(T)
        dG = molar_gibbs(hi, P, T) - molar_gibbs(lo, P, T)
        assert abs(dG) < PYROXENE_GIBBS_TOL[b.name], f"{b.name}: G jumps by {dG:.0f} J/mol at {P:.4e} Pa, {T:.1f} K"


MAXWELL_POINTS = [("solid-lpcen", 1e9, 600.0), ("solid-en", 1e9, 1500.0), ("solid-hpcen", 10e9, 1500.0),
                  ("solid-brg", 12.5e9, 2000.0), ("solid-brg", 50e9, 2500.0), ("solid-ppv", 130e9, 2500.0),
                  ("liquid", 5e9, 2500.0), ("liquid", 50e9, 4500.0)]


@pytest.mark.parametrize("phase, P, T", MAXWELL_POINTS)
def test_gibbs_derivatives_match_volume_and_entropy(phase, P, T, mgsio3_phases):
    """u, s, and rho of each phase must come from one Gibbs energy: dG/dP = V and dG/dT = -S."""
    eos = mgsio3_phases[phase]
    dP, dT = 1e-4 * P, 0.5
    dGdP = (molar_gibbs(eos, P + dP, T) - molar_gibbs(eos, P - dP, T)) / (2.0 * dP)
    dGdT = (molar_gibbs(eos, P, T + dT) - molar_gibbs(eos, P, T - dT)) / (2.0 * dT)
    V = molar_volume(eos, P, T)
    assert abs(dGdP - V) / V < 1e-6, f"{phase}: (dG/dP - V)/V = {(dGdP - V) / V:.2e}"
    assert abs(dGdT + molar_entropy(eos, P, T)) < 1e-3, f"{phase}: dG/dT + S = {dGdT + molar_entropy(eos, P, T):.4f} J/(mol K)"


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="liquid and brg Gibbs energies disagree below the triple point, so brg lies 120 to 220 kJ/mol below the pyroxenes (#13)")
def test_bridgmanite_is_metastable_at_one_bar(mgsio3_phases):
    brg, en = mgsio3_phases["solid-brg"], mgsio3_phases["solid-en"]
    for T in np.linspace(300.0, 1800.0, 6):
        dG = molar_gibbs(brg, 1e5, T) - molar_gibbs(en, 1e5, T)
        assert dG > 0.0, f"G_brg - G_en = {dG / 1e3:.1f} kJ/mol at 1 bar, {T:.0f} K"


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="G_brg - G_hpcen is -170 to -220 kJ/mol on the empirical 12 GPa line (#13)")
def test_gibbs_continuity_on_hpcen_brg_line(mgsio3_phases):
    hp, brg = mgsio3_phases["solid-hpcen"], mgsio3_phases["solid-brg"]
    for T in np.linspace(300.0, 3000.0, 10):
        dG = molar_gibbs(brg, mg._P_HPCEN_BRG, T) - molar_gibbs(hp, mg._P_HPCEN_BRG, T)
        assert abs(dG) < 1e4, f"G_brg - G_hpcen = {dG / 1e3:.1f} kJ/mol at 12 GPa, {T:.0f} K"


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="G_liq - G_solid falls to -75 kJ/mol on the pyroxene branch of the melting curve (#13)")
def test_gibbs_continuity_on_pyroxene_melting_curve(mgsio3_phases):
    liquid = mgsio3_phases["liquid"]
    for P in np.linspace(1e5, 11.99e9, 13):
        T = mg.T_melt_MgSiO3(P)
        solid = mg.get_mgsio3_phase(P, T - 1.0)
        dG = molar_gibbs(liquid, P, T) - molar_gibbs(mgsio3_phases[solid], P, T)
        assert abs(dG) < 1e4, f"G_liq - G_{solid} = {dG / 1e3:.1f} kJ/mol at {P:.4e} Pa, {T:.0f} K"


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="the Fei et al. (2021) melting curve is not the G_liq = G_brg line of the EoS (#13)")
def test_gibbs_continuity_on_brg_melting_curve(mgsio3_phases):
    liquid, brg = mgsio3_phases["liquid"], mgsio3_phases["solid-brg"]
    for P in np.logspace(np.log10(13e9), np.log10(150e9), 20):
        T = mg.T_melt_MgSiO3(P)
        dG = molar_gibbs(liquid, P, T) - molar_gibbs(brg, P, T)
        assert abs(dG) < 1e4, f"G_liq - G_brg = {dG / 1e3:.1f} kJ/mol at {P:.4e} Pa, {T:.0f} K"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="the HP-CEn to brg line at 12 GPa is an empirical cutoff")
def test_clapeyron_on_hpcen_brg_line(mgsio3_phases):
    """A constant-pressure boundary needs dS = 0 by Clausius-Clapeyron."""
    hp, brg = mgsio3_phases["solid-hpcen"], mgsio3_phases["solid-brg"]
    for T in np.linspace(300.0, mg.T_melt_MgSiO3(mg._P_HPCEN_BRG) - 5.0, 50):
        dS = molar_entropy(brg, mg._P_HPCEN_BRG, T) - molar_entropy(hp, mg._P_HPCEN_BRG, T)
        assert abs(dS) < 1.0, f"dS(brg - hpcen) = {dS:.2f} J/(mol K) at 12 GPa, {T:.0f} K"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="melting curve disagrees with Clapeyron on the pyroxene branches and above 0.5 TPa")
def test_clapeyron_on_melting_curve(mgsio3_phases):
    liquid = mgsio3_phases["liquid"]
    for P in MELT_PRESSURES:
        T = mg.T_melt_MgSiO3(P)
        solid = mgsio3_phases[mg.get_mgsio3_phase(P, T - 1.0)]
        dS = molar_entropy(liquid, P, T) - molar_entropy(solid, P, T)
        dV = molar_volume(liquid, P, T) - molar_volume(solid, P, T)
        dTdP = (mg.T_melt_MgSiO3(P * 1.001) - mg.T_melt_MgSiO3(P * 0.999)) / (0.002 * P)
        ratio = (dS / dV) * dTdP
        assert 1.0 / 2.5 < ratio < 2.5, f"melting: dS/dV is {ratio:.2f} times dP/dT at {P:.4e} Pa"


@pytest.mark.parametrize("phase", ["lp-cen", "orthoen", "hp-cen"])
def test_pyroxene_cold_energy_is_continuous_at_v0(phase):
    eos = mg.Sokolova22(phase)
    V0 = eos.params["V0"]
    assert eos._cold_energy(V0) == pytest.approx(eos._cold_energy(V0 * (1.0 + 1e-9)), abs=1e-3)
