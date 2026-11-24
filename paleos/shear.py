"""
Unrelaxed (infinite-frequency) shear modulus of the iron and MgSiO3 phases.

This is the shear modulus of a Maxwell or Andrade body. In liquids the relaxed modulus is zero, and the unrelaxed one below is what a viscoelastic model needs. Every function takes the density of the PALEOS phase, so the modulus follows the PALEOS volume, and returns Pa.

- MgSiO3 solids: finite-strain cold term and Debye thermal term of Stixrude & Lithgow-Bertelloni (2005, Eqs. 33, 41, 46, and 47), with the Stixrude & Lithgow-Bertelloni (2024) parameters of each phase (their own V0, K0, and K0') at the PALEOS volume.
- Solid iron: the Stixrude & Lithgow-Bertelloni (2024) cold curve at the given pressure, softened toward melting with the Nadal & Le Poac (2003) model. Its constants f = 0.112 and zeta = 0.1048 reproduce the ab initio hcp moduli of Martorell et al. (2013, Table S1) at 360 GPa, with their melting point of 7350 K at 360 GPa carried to other pressures in proportion to the PALEOS melting curve (Anzellini et al. 2013).
- Liquid iron: 0.712 times the hcp cold-curve modulus at the liquid density plus 6.02 MPa/K, a fit to the infinite-frequency moduli of Xian et al. (2019, Table S2) over 9.5 to 13.3 g/cm3 and 4300 to 10000 K (rms 7.4 GPa).
- Liquid MgSiO3: the shear modulus of MgSiO3 glass at the same density, 31.94 GPa (rho / 3000 kg/m3)^1.904, a fit to the first-principles glass moduli of Ghosh et al. (2014, Table S2) over 3.0 to 7.0 g/cm3. The glass stands in for the unrelaxed melt (Dingwell & Webb 1989). Below 3.0 g/cm3 the modulus stays at its 3.0 g/cm3 value, which lies nearer the 41.1 GPa measured for the glass at ambient conditions (Sanchez-Valle & Bass 2010) than the extrapolated fit.

Above the density ranges of the liquid fits, the fits are extrapolated.
"""

from functools import lru_cache

import numpy as np

R_GAS = 8.314462618  # J/(mol K)
K_B = 1.380649e-23  # J/K
N_A = 6.02214076e23  # 1/mol
T_REF = 300.0  # K, reference temperature of the SLB thermal term

_M_FE = 55.845e-3  # kg/mol
_M_MGSIO3 = (24.305 + 28.0855 + 3 * 15.9994) * 1e-3  # kg/mol

# HeFESTo parameter set 010123 (Stixrude & Lithgow-Bertelloni 2024), copied unmodified: V0 [m3/mol], K0 [Pa], K0', theta0 [K], gamma0, q0, G0 [Pa], G0', eta_S0, atoms per formula unit.
# The HeFESTo terms do not allow redistribution of modified values.
SLB = {
    'fea': (7.093e-06, 163.42114e9, 6.01732, 398.00486, 1.66729, 0.90658, 81.47e9, 1.9223, 5.96207, 1),
    'feg': (6.9883e-06, 165.0e9, 5.97, 285.19315, 1.84924, 1.02176, 91.0e9, 2.33869, 6.50014, 1),
    'fee': (6.7651e-06, 165.0e9, 4.97, 379.74281, 2.13092, 1.02253, 91.0e9, 2.24804, 5.20603, 1),
    'cen': (6.25e-05, 113.7599e9, 4.8061, 805.59286, 1.00921, 0.60142, 76.81031e9, 1.71384, 1.42387, 10),
    'en': (6.2676e-05, 107.07681e9, 7.02751, 812.21227, 0.78477, 3.43847, 76.8e9, 1.54596, 2.5045, 10),
    'mgc2': (6.076e-05, 116.02556e9, 6.23685, 824.44051, 1.12478, 0.20409, 87.92717e9, 1.84119, 2.14193, 10),
    'mgpv': (2.4445e-05, 250.56535e9, 4.13438, 892.95164, 1.54466, 0.83352, 172.9e9, 1.73254, 1.65233, 5),
    'mppv': (2.41108e-05, 247.74e9, 4.0, 931.02549, 1.64015, 1.76454, 166.64493e9, 1.98845, 1.43815, 5),
}  # fmt: skip

# Molar mass of each SLB formula unit [kg/mol]: Fe, Mg2Si2O6, and MgSiO3
FORMULA_MASS = {'fea': _M_FE, 'feg': _M_FE, 'fee': _M_FE, 'cen': 2 * _M_MGSIO3, 'en': 2 * _M_MGSIO3, 'mgc2': 2 * _M_MGSIO3, 'mgpv': _M_MGSIO3, 'mppv': _M_MGSIO3}

NP_F = 0.112  # Lindemann constant of Martorell et al. (2013)
NP_ZETA = 0.1048
NP_T_MELT_360 = 7350.0  # K, melting point of the simulated hcp crystal at 360 GPa (Martorell et al. 2013)
_T_HAT_MAX = 0.999  # keeps the Nadal & Le Poac J finite up to the model melting point

LIQUID_FE = (0.712396, 6.02382e6)  # (c, b [Pa/K]) of c G_cold(rho) + b T
LIQUID_MGSIO3 = (31.9392e9, 1.90372)  # (a [Pa], n) of a (rho / 3000 kg/m3)^n

_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def _strain(key, rho):
    """Eulerian finite strain f and compression x = V0 / V of the SLB formula unit at density rho."""
    x = rho * SLB[key][0] / FORMULA_MASS[key]
    return 0.5 * (x ** (2.0 / 3.0) - 1.0), x


def _cold_coefficients(key):
    """G0 and the coefficients a, b of the finite-strain expansion G0 + a f + b f^2 (SLB 2005, Eq. 33)."""
    _, K0, Kp, _, _, _, G0, Gp, _, _ = SLB[key]
    return G0, 3.0 * K0 * Gp - 5.0 * G0, 6.0 * K0 * Gp - 24.0 * K0 - 14.0 * G0 + 4.5 * K0 * Kp


def _cold(key, f):
    """Third-order finite-strain shear modulus at strain f (SLB 2005, Eq. 33)."""
    G0, a, b = _cold_coefficients(key)
    return (1.0 + 2.0 * f) ** 2.5 * (G0 + a * f + b * f * f)


@lru_cache(maxsize=None)
def peak_strain(key):
    """
    Strain f beyond which the cold shear modulus of SLB phase `key` falls with compression, infinite if it never does.

    With b < 0 (mgpv, mppv, and cen), the third-order expansion peaks where 9 b f^2 + (7 a + 2 b) f + 5 G0 + a = 0 and turns negative further on, an artifact of the truncation far outside the fitted pressures.
    """
    G0, a, b = _cold_coefficients(key)
    if b >= 0:
        return np.inf
    return max(np.roots([9.0 * b, 7.0 * a + 2.0 * b, 5.0 * G0 + a]))


def _debye_energy(T, theta, n):
    """Debye thermal energy 9 n R T (T/theta)^3 int_0^(theta/T) t^3 / (e^t - 1) dt [J/mol]."""
    x = theta / T
    t = 0.5 * x * (_GL_X + 1.0)
    integral = 0.5 * x * np.sum(_GL_W * t**3 / np.expm1(t))
    return 9.0 * n * R_GAS * T * integral / x**3


def slb_shear_modulus(key, rho, T):
    """
    Shear modulus [Pa] of SLB phase `key` at density rho [kg/m3] and temperature T [K].

    G = G_cold(f) - eta_S [E_th(T) - E_th(300 K)] / V (SLB 2005, Eqs. 33, 41, 46, and 47), with the sign of eta_S of BurnMan (`eos/slb.py`), which corrects a typo in Eq. 46. NaN in three cases: beyond the peak strain of the cold term, where nu^2 <= 0 (outside the Debye temperature expansion), and where G <= 0 (no mechanically stable crystal).
    """
    _, _, _, theta0, g0, q0, _, _, eta0, n = SLB[key]
    f, x = _strain(key, rho)
    if f > peak_strain(key):
        return np.nan
    a1 = 6.0 * g0
    a2 = -12.0 * g0 + 36.0 * g0**2 - 18.0 * q0 * g0
    nu2 = 1.0 + a1 * f + 0.5 * a2 * f * f
    if not nu2 > 0:
        return np.nan
    gamma = (2.0 * f + 1.0) * (a1 + a2 * f) / (6.0 * nu2)
    eta_s = -gamma + (2.0 * f + 1.0) ** 2 * (g0 + eta0) / nu2
    theta = theta0 * np.sqrt(nu2)
    V = FORMULA_MASS[key] / rho
    G = _cold(key, f) - eta_s * (_debye_energy(T, theta, n) - _debye_energy(T_REF, theta, n)) / V
    return G if G > 0 else np.nan


def _cold_at_pressure(key, P):
    """Cold-curve shear modulus [Pa] on the 300 K isotherm of the SLB equation of state at pressure P [Pa]."""
    _, K0, Kp, *_ = SLB[key]
    lo, hi = 0.0, 2.0
    for _ in range(60):
        f = 0.5 * (lo + hi)
        if 3.0 * K0 * f * (1.0 + 2.0 * f) ** 2.5 * (1.0 + 1.5 * (Kp - 4.0) * f) < P:
            lo = f
        else:
            hi = f
    return _cold(key, 0.5 * (lo + hi))


@lru_cache(maxsize=1)
def _melting_ratio():
    """Ratio of the Martorell et al. (2013) melting point to the PALEOS melting curve at 360 GPa."""
    from .iron_eos import T_melt_Fe

    return NP_T_MELT_360 / T_melt_Fe(360e9)


def solid_iron_shear_modulus(key, P, T, rho):
    """
    Shear modulus [Pa] of solid iron phase `key` ('fea', 'feg', or 'fee') at pressure P [Pa], temperature T [K], and density rho [kg/m3].

    Nadal & Le Poac (2003), as written by Banerjee (2005, arXiv:cond-mat/0512466, Eqs. 10 and 11): G = [G_cold(P) (1 - T/T_m) + rho k_B T / (C m)] / J(T/T_m), with C = (6 pi^2)^(2/3) f^2 / 3, m the atomic mass, and T_m the PALEOS melting curve scaled to 7350 K at 360 GPa, the melting point of the simulated crystal.
    """
    from .iron_eos import T_melt_Fe

    t = min(T / (_melting_ratio() * T_melt_Fe(P)), _T_HAT_MAX)
    C = (6.0 * np.pi**2) ** (2.0 / 3.0) * NP_F**2 / 3.0
    J = 1.0 + np.exp(-(1.0 + 1.0 / NP_ZETA) / (1.0 + NP_ZETA / (1.0 - t)))
    m = _M_FE / N_A
    return (_cold_at_pressure(key, P) * (1.0 - t) + rho * K_B * T / (C * m)) / J


def liquid_iron_shear_modulus(rho, T):
    """Unrelaxed shear modulus [Pa] of liquid iron at density rho [kg/m3] and temperature T [K]."""
    c, b = LIQUID_FE
    return c * _cold('fee', _strain('fee', rho)[0]) + b * T


def liquid_mgsio3_shear_modulus(rho):
    """Unrelaxed shear modulus [Pa] of liquid MgSiO3 at density rho [kg/m3], held at its 3.0 g/cm3 value below the fitted range."""
    a, n = LIQUID_MGSIO3
    return a * (max(rho, 3000.0) / 3000.0) ** n
