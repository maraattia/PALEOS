"""
Elastic moduli of every iron and MgSiO3 EoS.

K_T must equal rho (dP/drho)_T of the EoS density, K_S / K_T must equal C_P / C_V, and the unrelaxed shear modulus must be positive (Born stability). The points cover every class behind the IronEoS and MgSiO3EoS wrappers, the Miozzi20 to Hakim18 blend zone, and metastable states on both sides of the melting curves, which two-phase tables need.
"""

import numpy as np
import pytest

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
