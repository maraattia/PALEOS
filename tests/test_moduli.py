"""
Elastic moduli of every EoS.

K_T must equal rho (dP/drho)_T of the EoS density, K_S / K_T must equal C_P / C_V, and the unrelaxed shear modulus must be positive (Born stability) and fall with temperature in the solids. The points cover every class behind the IronEoS and MgSiO3EoS wrappers, the Miozzi20 to Hakim18 blend zone, and metastable states on both sides of the melting curves, which two-phase tables need. The wrappers must return the moduli of the stable phase, and water, which has no shear model, must return NaN.
"""

import numpy as np
import pytest

from paleos import shear

# (material, phase whose EoS is evaluated, P [Pa], T [K]). The phase need not be stable there.
POINTS = [
    ("iron", "solid-alpha", 1e5, 300.0), ("iron", "solid-delta", 1e9, 1750.0),
    ("iron", "solid-gamma", 5e9, 1500.0), ("iron", "solid-gamma", 50e9, 2500.0),
    ("iron", "solid-epsilon", 100e9, 2500.0), ("iron", "solid-epsilon", 250e9, 3000.0),
    ("iron", "solid-epsilon", 310e9, 4000.0), ("iron", "solid-epsilon", 380e9, 6500.0),
    ("iron", "solid-epsilon", 1e12, 8000.0),
    ("iron", "liquid", 10e9, 2000.0), ("iron", "liquid", 330e9, 7000.0), ("iron", "liquid", 3e12, 5000.0),
    ("mgsio3", "solid-lpcen", 1e5, 300.0), ("mgsio3", "solid-en", 5e9, 1500.0),
    ("mgsio3", "solid-hpcen", 10e9, 1500.0), ("mgsio3", "solid-brg", 25e9, 2000.0),
    ("mgsio3", "solid-brg", 120e9, 6000.0), ("mgsio3", "solid-ppv", 500e9, 6000.0),
    ("mgsio3", "liquid", 1e5, 2200.0), ("mgsio3", "liquid", 25e9, 2500.0), ("mgsio3", "liquid", 1e12, 15000.0),
]


@pytest.fixture(scope="module")
def phases(iron_phases, mgsio3_phases):
    return {"iron": iron_phases, "mgsio3": mgsio3_phases}


def cases():
    return [pytest.param(*p, id=f"{p[0]}-{p[1]}-{p[2] / 1e9:g}GPa-{p[3]:g}K") for p in POINTS]


@pytest.mark.parametrize("material, phase, P, T", cases())
def test_isothermal_bulk_modulus_matches_density(phases, material, phase, P, T):
    eos = phases[material][phase]
    h = 1e-4 * P
    K_fd = 2 * h / np.log(eos.density(P + h, T) / eos.density(P - h, T))
    assert eos.isothermal_bulk_modulus(P, T) == pytest.approx(K_fd, rel=1e-4)


@pytest.mark.parametrize("material, phase, P, T", cases())
def test_bulk_modulus_ratio_is_heat_capacity_ratio(phases, material, phase, P, T):
    eos = phases[material][phase]
    ratio = eos.isobaric_heat_capacity(P, T) / eos.isochoric_heat_capacity(P, T)
    assert eos.adiabatic_bulk_modulus(P, T) / eos.isothermal_bulk_modulus(P, T) == pytest.approx(ratio, rel=1e-9)


@pytest.mark.parametrize("material, phase, P, T", cases())
def test_shear_modulus_is_positive(phases, material, phase, P, T):
    assert phases[material][phase].shear_modulus(P, T) > 0


@pytest.mark.parametrize("material, phase, P, T", [c for c in cases() if c.values[1] != "liquid"])
def test_solid_shear_modulus_falls_with_temperature(phases, material, phase, P, T):
    eos = phases[material][phase]
    assert eos.shear_modulus(P, 1.1 * T) < eos.shear_modulus(P, T)


@pytest.mark.parametrize("key", sorted(shear.SLB))
def test_slb_shear_modulus_is_g0_at_the_reference_state(key):
    """At V0 and 300 K the finite strain and the thermal term vanish, so G = G0."""
    rho0 = shear.FORMULA_MASS[key] / shear.SLB[key][0]
    assert shear.slb_shear_modulus(key, rho0, shear.T_REF) == pytest.approx(shear.SLB[key][6], rel=1e-12)


MODULI = ("isothermal_bulk_modulus", "adiabatic_bulk_modulus", "shear_modulus")
WRAPPER_POINTS = [("iron", 1e5, 300.0), ("iron", 1e9, 1750.0), ("iron", 5e9, 1500.0), ("iron", 300e9, 4000.0),
                  ("iron", 300e9, 7000.0), ("mgsio3", 1e5, 300.0), ("mgsio3", 5e9, 1500.0), ("mgsio3", 10e9, 1200.0),
                  ("mgsio3", 50e9, 2000.0), ("mgsio3", 200e9, 3000.0), ("mgsio3", 25e9, 4000.0)]


@pytest.mark.parametrize("material, P, T", WRAPPER_POINTS)
def test_wrappers_return_the_moduli_of_the_stable_phase(iron_wrapper, mgsio3_wrapper, material, P, T):
    wrapper = {"iron": iron_wrapper, "mgsio3": mgsio3_wrapper}[material]
    eos = wrapper._phase_eos_map[wrapper.phase(P, T)]
    for method in MODULI:
        assert getattr(wrapper, method)(P, T) == getattr(eos, method)(P, T)


WATER_POINTS = [(1e5, 300.0), (1e5, 200.0), (1e8, 266.7), (5e9, 300.0), (30e9, 1000.0), (0.56e9, 2993.0),
                (398e9, 300.0), (1e12, 1e4), (1e3, 500.0)]


@pytest.mark.parametrize("P, T", WATER_POINTS)
def test_water_bulk_moduli(water, P, T):
    h = 1e-2 * P
    K_fd = 2 * h / np.log(water.density(P + h, T) / water.density(P - h, T))
    K_T = water.isothermal_bulk_modulus(P, T)
    assert K_T == pytest.approx(K_fd, rel=2e-2)
    K_S = water.adiabatic_bulk_modulus(P, T)
    if np.isfinite(K_S):
        assert K_S * water.isochoric_heat_capacity(P, T) == pytest.approx(K_T * water.isobaric_heat_capacity(P, T), rel=1e-9)
    assert np.isnan(water.shear_modulus(P, T))


def test_water_vapor_is_an_ideal_gas(water):
    assert water.isothermal_bulk_modulus(1e3, 500.0) == pytest.approx(1e3, rel=1e-2)


@pytest.mark.parametrize("phase, key, P", [("solid-alpha", "fea", 1e5), ("solid-gamma", "feg", 10e9),
                                           ("solid-epsilon", "fee", 100e9), ("solid-epsilon", "fee", 360e9)])
def test_solid_iron_returns_its_slb_modulus_at_300_k(iron_phases, phase, key, P):
    """The Nadal & Le Poac softening is anchored on the SLB 300 K isotherm of each iron phase."""
    assert iron_phases[phase].shear_modulus(P, 300.0) == pytest.approx(shear._isotherm_300(key, P)[0], rel=5e-3)
