"""
PALEOS Equations of State for MgSiO₃

This module contains implementations of various equations of state (EoS) for 
MgSiO₃ phases relevant to planetary interiors and high-pressure physics.

Each EoS is implemented as a separate class with consistent method signatures
to ensure interoperability across different implementations.

EoS Classes
-----------
- Wolf15: Bridgmanite (Mg,Fe)SiO₃ from Wolf et al. (2015)
- Sakai16: Post-perovskite MgSiO₃ from Sakai et al. (2016)
- Sokolova22: (high/low-pressure) clino- and orthoenstatite MgSiO₃ from Sokolova et al. (2022)
- Wolf18: Liquid MgSiO₃ from Wolf & Bower (2018)
- MgSiO3EoS: Wrapper class with pre-instantiated phases for efficient repeated
             evaluation with automatic phase selection

Phase Determination
-------------------
The module provides functions to determine the stable MgSiO₃ phase at given
P-T conditions:

- get_mgsio3_phase(P, T): Returns the stable phase ('solid-lpcen', 'solid-en',
                          'solid-hpcen', 'solid-brg', 'solid-ppv', or 'liquid')
- get_mgsio3_eos(phase): Returns EoS instance for a given phase name
- get_mgsio3_eos_for_PT(P, T): Returns (EoS instance, phase) for given conditions

Phase boundary functions:
- P_lpcen_hpcen(T): LP-CEn ↔ HP-CEn boundary (Sokolova et al. 2022)
- P_lpcen_en(T): LP-CEn ↔ OrthoEn boundary (Sokolova et al. 2022)
- P_en_hpcen(T): OrthoEn ↔ HP-CEn boundary (Sokolova et al. 2022)
- P_brg_ppv(T): Bridgmanite ↔ post-perovskite boundary (Ono & Oganov 2005)
- T_melt_MgSiO3(P): Melting curve (Belonoshko et al. 2005 / Fei et al. 2021)

Author: Mara Attia
Date: February 2026
"""

import numpy as np
from scipy.optimize import brentq

# Physical constants
R_GAS = 8.314462618  # J/(mol·K) - Universal gas constant
N_AVOGADRO = 6.02214076e23  # mol^-1 - Avogadro number

# Atomic masses [g/mol]
_M_Mg = 24.305
_M_Fe = 55.845
_M_Si = 28.085
_M_O = 15.999

# MgSiO₃ base molar mass [kg/mol]
_M_MgSiO3 = (_M_Mg + _M_Si + 3 * _M_O) * 1e-3  # 0.100387 kg/mol


# =============================================================================
# Equations of State Classes
# =============================================================================
#
# Each EoS class implements the same public interface for thermodynamic 
# properties as functions of pressure P [Pa] and temperature T [K]:
#
#   density(P, T)                  -> kg/m³
#   specific_internal_energy(P, T) -> J/kg
#   specific_entropy(P, T)         -> J/(kg·K)
#   isobaric_heat_capacity(P, T)   -> J/(kg·K)
#   isochoric_heat_capacity(P, T)  -> J/(kg·K)
#   thermal_expansion(P, T)        -> K⁻¹
#   adiabatic_gradient(P, T)       -> dimensionless
#
# =============================================================================


class Wolf15:
    """
    Equation of state for (Mg,Fe)SiO₃ bridgmanite from Wolf et al. (2015).
    
    Reference:
    Wolf, A.S., Jackson, J.M., Dera, P., Prakapenka, V.B. (2015)
    "The thermal equation of state of (Mg, Fe)SiO₃ bridgmanite (perovskite)
    and implications for lower mantle structures"
    Journal of Geophysical Research: Solid Earth, 120, 7460-7489,
    DOI: 10.1002/2015JB012108
    
    This implementation uses:
    - Vinet (Morse-Rydberg) EoS for cold compression (Eq. 2)
    - Mie-Grüneisen-Debye formalism for thermal pressure (Eq. 3)
    - Power-law Grüneisen parameter (Eq. 5): γ(V) = γ_0(V/V_0)^q
    - Debye temperature (Eq. 6): Θ(V) = Θ_0 exp[-(γ - γ_0)/q]
    - Ideal lattice mixing model for intermediate Fe compositions (Section 5.1)
    
    The EoS is calibrated using laser-heated diamond anvil cell data with
    neon pressure medium at quasi-hydrostatic conditions, covering pressures
    up to ~120 GPa and temperatures up to ~2500 K. Parameters are provided
    for both pure MgSiO₃ (0% Fe) and (Mg₀.₈₇Fe₀.₁₃)SiO₃ (13% Fe) end-members.
    
    Thermodynamic model:
    P(V,T) = P_ref(V) + ΔP_th(V,T)
    
    where:
    - P_ref(V): Vinet pressure along the 300 K reference isotherm
    - ΔP_th(V,T) = (γ/V)[E_th(V,T) - E_th(V,T_0)]: thermal pressure
    - E_th(V,T) = 3nRT D_3(Θ/T): Debye thermal energy (including zero-point)
    
    For intermediate compositions (0 < x_Fe < 0.13), pressures and all
    V-T-dependent thermodynamic quantities are linearly interpolated between
    the two end-members at constant volume and temperature following the
    ideal lattice mixing model.
    
    All methods take pressure P [Pa] and temperature T [K] as inputs and
    return quantities in SI units unless otherwise specified.
    
    Parameters
    ----------
    x_Fe : float, optional
        Iron mole fraction on the A-site: (Mg_{1-x}Fe_x)SiO₃.
        Must satisfy 0 ≤ x_Fe ≤ 0.13. Default is 0.0 (pure MgSiO₃).
    
    Attributes
    ----------
    x_Fe : float
        Iron mole fraction
    T0 : float
        Reference temperature (300 K)
    molar_mass : float
        Composition-dependent molar mass [kg/mol]
    params_0Fe : dict
        Parameters for pure MgSiO₃ end-member
    params_13Fe : dict
        Parameters for 13% Fe end-member
    
    Examples
    --------
    >>> # Pure MgSiO₃ bridgmanite
    >>> eos = Wolf15(x_Fe=0.0)
    >>> rho = eos.density(P=50e9, T=2000)
    >>> print(f"Density: {rho:.1f} kg/m³")
    
    >>> # Iron-bearing bridgmanite
    >>> eos_fe = Wolf15(x_Fe=0.10)
    >>> rho_fe = eos_fe.density(P=50e9, T=2000)
    
    Notes
    -----
    The bridgmanite structure has Z = 4 formula units per unit cell, with
    n = 5 atoms per formula unit (one A-site cation + Si + 3 O). The
    Dulong-Petit heat capacity limit is therefore C_V^max = 3nR = 15R
    per mole of formula units.
    
    The paper notes that pure MgSiO₃ bridgmanite is inconsistent with
    PREM at any Fe content, highlighting the importance of Fe incorporation
    for realistic lower mantle models.
    
    Synchrotron Mössbauer spectroscopy confirms high-spin Fe²⁺ is
    maintained to at least 120 GPa, validating the ideal mixing approach
    across the full pressure range studied.
    """
    
    def __init__(self, x_Fe: float = 0.0):
        """
        Initialize the Wolf15 EoS for (Mg,Fe)SiO₃ bridgmanite.
        
        Parameters
        ----------
        x_Fe : float, optional
            Iron mole fraction: (Mg_{1-x}Fe_x)SiO₃.
            Must be in range [0, 0.13]. Default is 0.0.
            
        Raises
        ------
        ValueError
            If x_Fe is outside the valid range [0, 0.13]
        """
        if not (0.0 <= x_Fe <= 0.13):
            raise ValueError(
                f"x_Fe = {x_Fe} is outside valid range [0, 0.13]. "
                f"The ideal lattice mixing model is calibrated for "
                f"compositions between pure MgSiO₃ and (Mg₀.₈₇Fe₀.₁₃)SiO₃."
            )
        
        self.x_Fe = x_Fe
        
        # Reference temperature
        self.T0 = 300.0  # K
        
        # Composition-dependent molar mass [kg/mol]
        # (Mg_{1-x}Fe_x)SiO₃: replace Mg with Fe on A-site
        self.molar_mass = (_M_Mg * (1 - x_Fe) + _M_Fe * x_Fe + _M_Si + 3 * _M_O) * 1e-3
        
        # Number of atoms per formula unit
        self.n_atoms = 5  # (Mg/Fe) + Si + 3×O
        
        # Number of formula units per unit cell (Pbnm space structure)
        self.Z = 4
        
        # Mixing weight: fraction of 13% Fe end-member
        if x_Fe <= 0.0:
            self.w_Fe = 0.0
        else:
            self.w_Fe = x_Fe / 0.13
        
        # Parameters for 0% Fe (pure MgSiO₃) end-member
        # From Table 3 of Wolf et al. (2015)
        V0_cell_0Fe = 162.12e-30  # m³
        self.params_0Fe = {
            'U0': -86826.,                                      # J/mol
            'S0': -180.300,                                     # J/(mol·K)
            'V0': V0_cell_0Fe * N_AVOGADRO / self.Z,            # m³/mol
            'K0': 262.3e9,                                      # Pa
            'K0_prime': 4.044,                                  # dimensionless
            'Theta0': 1000.0,                                   # K
            'gamma0': 1.675,                                    # dimensionless
            'q': 1.39,                                          # dimensionless
            'n': self.n_atoms,                                  # dimensionless
        }
        
        # Parameters for 13% Fe end-member: (Mg₀.₈₇Fe₀.₁₃)SiO₃
        # From Table 3 of Wolf et al. (2015)
        V0_cell_13Fe = 163.16e-30  # m³
        self.params_13Fe = {
            'U0': -86826.,                                      # J/mol
            'S0': -180.300,                                     # J/(mol·K)
            'V0': V0_cell_13Fe * N_AVOGADRO / self.Z,           # m³/mol
            'K0': 243.8e9,                                      # Pa
            'K0_prime': 4.160,                                  # dimensionless
            'Theta0': 1000.0,                                   # K
            'gamma0': 1.400,                                    # dimensionless
            'q': 0.56,                                          # dimensionless
            'n': self.n_atoms,                                  # dimensionless
        }
    
    # =========================================================================
    # Core thermodynamic helper functions (single end-member)
    # =========================================================================
    
    def _gruneisen_parameter(self, V: float, params: dict) -> float:
        """
        Calculate Grüneisen parameter as a function of volume.
        
        Equation (5): γ(V) = γ_0 (V/V_0)^q
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Grüneisen parameter (dimensionless)
        """
        x = V / params['V0']
        return params['gamma0'] * x**params['q']
    
    def _debye_temperature(self, V: float, params: dict) -> float:
        """
        Calculate Debye temperature as a function of volume.
        
        Equation (6): Θ(V) = Θ_0 exp[(γ_0 - γ(V))/q]
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Debye temperature [K]
        """
        gamma_V = self._gruneisen_parameter(V, params)
        gamma0 = params['gamma0']
        q = params['q']
        Theta0 = params['Theta0']
        
        if abs(q) < 1e-10:
            return Theta0
        
        exponent = (gamma0 - gamma_V) / q
        return Theta0 * np.exp(exponent)
    
    def _debye_integral(self, x_max: float) -> float:
        """
        Calculate the Debye integral: ∫_0^x_max x^3/(exp(x) - 1) dx
        
        Parameters
        ----------
        x_max : float
            Upper limit of integration (Θ/T)
            
        Returns
        -------
        float
            Value of the Debye integral
        """
        # For very small x_max, use series expansion
        if x_max < 0.01:
            # ∫_0^x x^3/(e^x - 1) dx ≈ x^3/3 for small x
            return x_max**3 / 3
        
        # For very large x_max, integral approaches π^4/15
        if x_max > 100:
            return np.pi**4 / 15
        
        # For intermediate values, use numerical integration
        # Split into fine grid near origin where integrand varies rapidly
        n_points = 1000
        x = np.linspace(0, x_max, n_points)

        # Avoid division by zero at x=0
        x[0] = 1e-10

         # Integrand: x^3/(exp(x) - 1)
        integrand = x**3 / (np.exp(x) - 1)

        # Trapezoidal rule integration
        return np.trapezoid(integrand, x)
    
    def _debye_function_D3(self, x: float) -> float:
        """
        Calculate the third Debye function: D_3(x) = (3/x^3) ∫_0^x t^3/(e^x - 1) dx
        
        Parameters
        ----------
        x : float
            Θ/T ratio
            
        Returns
        -------
        float
            D_3(x) value
        """
        integral = self._debye_integral(x)
        return (3 / x**3) * integral
    
    def _thermal_energy(self, V: float, T: float, params: dict) -> float:
        """
        Calculate Debye thermal energy.
        
        E_th(V,T) = 3nR [3Θ/8 + T·D_3(Θ/T)]
        
        where Θ/8 is the zero-point energy per mode and D_3 is the 
        third Debye function.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Thermal energy [J/mol]
        """
        Theta = self._debye_temperature(V, params)
        n = params['n']
        
        if T < 1e-6:
            # Only zero-point energy at T = 0
            return 9 * n * R_GAS * Theta / 8
        
        x = Theta / T
        D3 = self._debye_function_D3(x)
        
        return 3 * n * R_GAS * (3 * Theta / 8 + T * D3)
    
    def _cold_pressure(self, V: float, params: dict) -> float:
        """
        Calculate cold compression pressure using Vinet EoS.
        
        Equation (2):
        P_ref(x) = 3K_0(1 - x)x^{-2} exp[ν(1 - x)]
        
        where x = (V/V_0)^(1/3) and ν = (3/2)(K'_0 - 1)
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Cold compression pressure [Pa]
        """
        V0 = params['V0']
        K0 = params['K0']
        K0_prime = params['K0_prime']
        
        x = (V / V0)**(1 / 3)
        nu = 1.5 * (K0_prime - 1)
        
        return 3 * K0 * (1 - x) * x**(-2) * np.exp(nu * (1 - x))
    
    def _thermal_pressure(self, V: float, T: float, params: dict) -> float:
        """
        Calculate thermal pressure contribution.
        
        Equation (3): ΔP_th = (γ/V)[E_th(V,T) - E_th(V,T_0)]
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Thermal pressure [Pa]
        """
        if abs(T - self.T0) < 1e-6:
            return 0.0
        
        gamma_V = self._gruneisen_parameter(V, params)
        E_T = self._thermal_energy(V, T, params)
        E_T0 = self._thermal_energy(V, self.T0, params)
        
        return (gamma_V / V) * (E_T - E_T0)
    
    def _total_pressure_single(self, V: float, T: float, params: dict) -> float:
        """
        Calculate total pressure for a single end-member.
        
        P(V,T) = P_ref(V) + ΔP_th(V,T)
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Total pressure [Pa]
        """
        return self._cold_pressure(V, params) + self._thermal_pressure(V, T, params)
    
    def _total_pressure(self, V: float, T: float) -> float:
        """
        Calculate total pressure with ideal lattice mixing.
        
        For compositions between the two end-members:
        P(V,T) = (1 - w) P_0Fe(V,T) + w P_13Fe(V,T)
        
        where w = x_Fe / 0.13
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Total pressure [Pa]
        """
        if self.w_Fe <= 0.0:
            return self._total_pressure_single(V, T, self.params_0Fe)
        elif self.w_Fe >= 1.0:
            return self._total_pressure_single(V, T, self.params_13Fe)
        else:
            P_0 = self._total_pressure_single(V, T, self.params_0Fe)
            P_13 = self._total_pressure_single(V, T, self.params_13Fe)
            return (1 - self.w_Fe) * P_0 + self.w_Fe * P_13
    
    def _find_volume(self, P: float, T: float) -> float:
        """
        Find molar volume for given pressure and temperature.
        
        Solves P(V,T) = P_target using Brent's method.
        
        Parameters
        ----------
        P : float
            Target pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Molar volume [m³/mol]
            
        Raises
        ------
        RuntimeError
            If root finding fails to converge
        """
        # Use the 0% Fe V0 as reference scale (both end-members are similar)
        V0 = self.params_0Fe['V0']
        
        V_min = 0.3 * V0
        V_max = 1.5 * V0
        
        def pressure_residual(V):
            return self._total_pressure(V, T) - P
        
        try:
            V = brentq(pressure_residual, V_min, V_max, xtol=1e-12)
            return V
        except ValueError:
            raise RuntimeError(
                f"Failed to find volume at P = {P/1e9:.2f} GPa, T = {T:.2f} K "
                f"(x_Fe = {self.x_Fe:.3f}). "
                f"Pressure may be outside valid range for Wolf15 bridgmanite EoS."
            )
    
    # =========================================================================
    # Thermodynamic property helpers (single end-member, molar quantities)
    # =========================================================================
    
    def _isochoric_heat_capacity_single(self, V: float, T: float, params: dict) -> float:
        """
        Calculate molar isochoric heat capacity for a single end-member.

        C_V = (∂E/∂T)_V
        
        For Debye model:
        C_V = 9nR (T/Θ)^3 ∫_0^(Θ/T) x^4 exp(x)/(exp(x) - 1)^2 dx
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Isochoric heat capacity [J/(mol·K)]
        """
        if T < 1e-6:
            return 0.0
        
        Theta = self._debye_temperature(V, params)
        n = params['n']
        
        x_max = Theta / T
        
        # High temperature limit: C_V → 3nR (Dulong-Petit)
        if x_max < 0.01:
            return 3 * n * R_GAS
        
        # Low temperature limit: C_V → 0
        if x_max > 100:
            return 0.0
        
        # Numerical integration
        n_points = 1000
        x = np.linspace(1e-10, x_max, n_points)
        exp_x = np.exp(x)
        integrand = x**4 * exp_x / (exp_x - 1)**2
        integral = np.trapezoid(integrand, x)
        
        return 9 * n * R_GAS * (T / Theta)**3 * integral
    
    def _isothermal_bulk_modulus_single(self, V: float, T: float, params: dict) -> float:
        """
        Calculate isothermal bulk modulus for a single end-member.
        
        K_T = -V(∂P/∂V)_T = K_cold + K_th
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Isothermal bulk modulus [Pa]
        """
        V0 = params['V0']
        K0 = params['K0']
        K0_prime = params['K0_prime']
        q = params['q']
        
        gamma = self._gruneisen_parameter(V, params)
        
        # Cold bulk modulus from Vinet EoS
        # K_cold = K0 y^{-2} exp(η(1-y)) [2 - y + η y(1-y)]
        # where y = (V/V0)^{1/3}, η = 3/2(K'0 - 1)
        y = (V / V0)**(1 / 3)
        eta = 1.5 * (K0_prime - 1)
        
        K_cold = K0 * y**(-2) * np.exp(eta * (1 - y)) * (2 - y + eta * y * (1 - y))
        
        # Thermal contribution to bulk modulus
        # dP_th/dV = (q - 1 - γ) P_th/V + (γ/V)^2 (T·Cv_T - T0·Cv_T0)
        P_th = self._thermal_pressure(V, T, params)
        Cv_T = self._isochoric_heat_capacity_single(V, T, params)
        Cv_T0 = self._isochoric_heat_capacity_single(V, self.T0, params)
        
        dP_th_dV = (q - 1 - gamma) * P_th / V + (gamma / V)**2 * (T * Cv_T - self.T0 * Cv_T0)
        
        # K_th = -V · dP_th/dV
        K_th = -V * dP_th_dV
        
        return K_cold + K_th
    
    def _thermal_expansion_single(self, V: float, T: float, params: dict) -> float:
        """
        Calculate thermal expansion coefficient for a single end-member.
        
        α = (∂P/∂T)_V / K_T = (γ/V) C_V / K_T
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Thermal expansion coefficient [K⁻¹]
        """
        if T < 1e-6:
            return 0.0
        
        gamma = self._gruneisen_parameter(V, params)
        Cv = self._isochoric_heat_capacity_single(V, T, params)
        KT = self._isothermal_bulk_modulus_single(V, T, params)
        
        if abs(KT) < 1e-6:
            return 0.0
        
        dP_dT = (gamma / V) * Cv
        return dP_dT / KT
    
    def _entropy_single(self, V: float, T: float, params: dict) -> float:
        """
        Calculate molar entropy for a single end-member.
        
        For Debye model (e.g., Gopal 1966, equation 2.16b):
        S = nR[4D_3(θ_D/T) - 3ln(1 - exp(-θ_D/T))] + S_0
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Molar entropy [J/(mol·K)]
        """
        S0 = params['S0']
        
        if T < 1e-6:
            return S0
        
        Theta = self._debye_temperature(V, params)
        n = params['n']
        
        x = Theta / T
        
        if x > 100:
            return S0
        
        # Debye entropy
        D3 = self._debye_function_D3(x)
        
        S_Debye = n * R_GAS * (4 * D3 - 3 * np.log(1 - np.exp(-x)))
        
        return S_Debye + S0
    
    def _internal_energy_single(self, V: float, T: float, params: dict) -> float:
        """
        Calculate molar internal energy for a single end-member.
        
        E = E_cold(V) + E_th(V,T) - E_th(V,T_0) + U_0
        
        For the Vinet cold energy:
        E_cold = 9 K_0 V_0 [1 - (1 - ν(1-y)) exp(ν(1-y))] / ν²
        
        where y = (V/V_0)^{1/3}, ν = 3/2(K'_0 - 1)
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
        params : dict
            End-member parameter set
            
        Returns
        -------
        float
            Molar internal energy [J/mol]
        """
        U0 = params['U0']
        V0 = params['V0']
        K0 = params['K0']
        K0_prime = params['K0_prime']
        
        # Cold energy from Vinet EoS
        y = (V / V0)**(1 / 3)
        nu = 1.5 * (K0_prime - 1)
        
        E_cold = 9 * K0 * V0 * (1 - (1 - nu * (1 - y)) * np.exp(nu * (1 - y))) / nu**2
        
        # Thermal energy (Debye)
        E_th = self._thermal_energy(V, T, params)
        E_th_ref = self._thermal_energy(V, self.T0, params)
        
        return U0 + E_cold + (E_th - E_th_ref)
    
    # =========================================================================
    # Mixing helpers (blend end-member properties at constant V, T)
    # =========================================================================
    
    def _blend(self, val_0Fe: float, val_13Fe: float) -> float:
        """
        Linearly blend two end-member values using composition weight.
        
        Parameters
        ----------
        val_0Fe : float
            Value from 0% Fe end-member
        val_13Fe : float
            Value from 13% Fe end-member
            
        Returns
        -------
        float
            Blended value
        """
        return (1 - self.w_Fe) * val_0Fe + self.w_Fe * val_13Fe
    
    def _isochoric_heat_capacity(self, V: float, T: float) -> float:
        """
        Calculate molar C_V with ideal lattice mixing.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Isochoric heat capacity [J/(mol·K)]
        """
        if self.w_Fe <= 0.0:
            return self._isochoric_heat_capacity_single(V, T, self.params_0Fe)
        elif self.w_Fe >= 1.0:
            return self._isochoric_heat_capacity_single(V, T, self.params_13Fe)
        else:
            Cv_0 = self._isochoric_heat_capacity_single(V, T, self.params_0Fe)
            Cv_13 = self._isochoric_heat_capacity_single(V, T, self.params_13Fe)
            return self._blend(Cv_0, Cv_13)
    
    def _isothermal_bulk_modulus(self, V: float, T: float) -> float:
        """
        Calculate isothermal bulk modulus with ideal lattice mixing.
        
        Since pressure mixes linearly at constant V,T, so does
        K_T = -V(∂P/∂V)_T.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Isothermal bulk modulus [Pa]
        """
        if self.w_Fe <= 0.0:
            return self._isothermal_bulk_modulus_single(V, T, self.params_0Fe)
        elif self.w_Fe >= 1.0:
            return self._isothermal_bulk_modulus_single(V, T, self.params_13Fe)
        else:
            KT_0 = self._isothermal_bulk_modulus_single(V, T, self.params_0Fe)
            KT_13 = self._isothermal_bulk_modulus_single(V, T, self.params_13Fe)
            return self._blend(KT_0, KT_13)
    
    def _thermal_expansion_coeff(self, V: float, T: float) -> float:
        """
        Calculate thermal expansion coefficient with ideal lattice mixing.
        
        Since α = (∂P/∂T)_V / K_T, and both numerator and denominator
        mix linearly, we compute the mixed α from the mixed (∂P/∂T)_V
        and mixed K_T.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Thermal expansion coefficient [K⁻¹]
        """
        if self.w_Fe <= 0.0:
            return self._thermal_expansion_single(V, T, self.params_0Fe)
        elif self.w_Fe >= 1.0:
            return self._thermal_expansion_single(V, T, self.params_13Fe)
        else:
            # Compute mixed (∂P/∂T)_V and mixed K_T
            gamma_0 = self._gruneisen_parameter(V, self.params_0Fe)
            Cv_0 = self._isochoric_heat_capacity_single(V, T, self.params_0Fe)
            dPdT_0 = (gamma_0 / V) * Cv_0
            
            gamma_13 = self._gruneisen_parameter(V, self.params_13Fe)
            Cv_13 = self._isochoric_heat_capacity_single(V, T, self.params_13Fe)
            dPdT_13 = (gamma_13 / V) * Cv_13
            
            dPdT = self._blend(dPdT_0, dPdT_13)
            KT = self._isothermal_bulk_modulus(V, T)
            
            if abs(KT) < 1e-6:
                return 0.0
            return dPdT / KT
    
    def _entropy_molar(self, V: float, T: float) -> float:
        """
        Calculate molar entropy with ideal lattice mixing.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Molar entropy [J/(mol·K)]
        """
        if self.w_Fe <= 0.0:
            return self._entropy_single(V, T, self.params_0Fe)
        elif self.w_Fe >= 1.0:
            return self._entropy_single(V, T, self.params_13Fe)
        else:
            S_0 = self._entropy_single(V, T, self.params_0Fe)
            S_13 = self._entropy_single(V, T, self.params_13Fe)
            return self._blend(S_0, S_13)
    
    def _internal_energy_molar(self, V: float, T: float) -> float:
        """
        Calculate molar internal energy with ideal lattice mixing.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Molar internal energy [J/mol]
        """
        if self.w_Fe <= 0.0:
            return self._internal_energy_single(V, T, self.params_0Fe)
        elif self.w_Fe >= 1.0:
            return self._internal_energy_single(V, T, self.params_13Fe)
        else:
            E_0 = self._internal_energy_single(V, T, self.params_0Fe)
            E_13 = self._internal_energy_single(V, T, self.params_13Fe)
            return self._blend(E_0, E_13)
    
    # =========================================================================
    # Public interface
    # =========================================================================
    
    def density(self, P: float, T: float) -> float:
        """
        Calculate density at given pressure and temperature.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Density [kg/m³]
        """
        V = self._find_volume(P, T)
        return self.molar_mass / V
    
    def specific_internal_energy(self, P: float, T: float) -> float:
        """
        Calculate specific internal energy.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific internal energy [J/kg]
        """
        V = self._find_volume(P, T)
        E_molar = self._internal_energy_molar(V, T)
        return E_molar / self.molar_mass
    
    def specific_entropy(self, P: float, T: float) -> float:
        """
        Calculate specific entropy.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific entropy [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        S_molar = self._entropy_molar(V, T)
        return S_molar / self.molar_mass
    
    def isobaric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isobaric heat capacity.
        
        C_P = C_V + α² T V K_T
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific isobaric heat capacity [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        Cv = self._isochoric_heat_capacity(V, T)
        alpha = self._thermal_expansion_coeff(V, T)
        KT = self._isothermal_bulk_modulus(V, T)
        
        Cp_molar = Cv + alpha**2 * T * V * KT
        return Cp_molar / self.molar_mass
    
    def isochoric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isochoric heat capacity.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific isochoric heat capacity [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        Cv_molar = self._isochoric_heat_capacity(V, T)
        return Cv_molar / self.molar_mass
    
    def thermal_expansion(self, P: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Thermal expansion coefficient [K⁻¹]
        """
        V = self._find_volume(P, T)
        return self._thermal_expansion_coeff(V, T)
    
    def adiabatic_gradient(self, P: float, T: float) -> float:
        """
        Calculate dimensionless adiabatic temperature gradient.
        
        (∂ln T/∂ln P)_S = α P / (ρ C_P)
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Adiabatic gradient (dimensionless)
        """
        alpha = self.thermal_expansion(P, T)
        Cp = self.isobaric_heat_capacity(P, T)
        rho = self.density(P, T)
        
        return alpha * P / (Cp * rho)


class Sakai16:
    """
    Equation of state for MgSiO₃ post-perovskite from Sakai et al. (2016).
    
    Reference:
    Sakai, T., Dekura, H., Hirao, N. (2016)
    "Experimental and theoretical thermal equations of state of MgSiO₃
    post-perovskite at multi-megabar pressures"
    Scientific Reports, 6:22652, DOI: 10.1038/srep22652
    
    This implementation uses the ab initio parameters (fit 8, Table 1):
    - Keane EoS for cold compression (Supplementary Information A)
    - Mie-Grüneisen-Debye (MGD) formalism for thermal pressure
    - Al'tshuler form for the Grüneisen parameter
    
    The Keane EoS is chosen because it includes the finite-pressure derivative
    K_∞' and yields pressures intermediate between 3BM and Vinet, making it
    well suited for multi-megabar extrapolation to super-Earth mantles.
    
    The ab initio (fit 8) parameters are used because they cover the widest
    P-T range (up to 1.2 TPa and 5000 K) and satisfy both thermodynamic
    constraints and Keane's rule.
    
    Thermodynamic model:
    P(V,T) = P_cold(V) + ΔP_th(V,T)
    
    where:
    - P_cold(V): Keane EoS along the 300 K reference isotherm
    - ΔP_th(V,T) = (γ/V)[E_th(V,T) - E_th(V,T_0)]: MGD thermal pressure
    - γ(V) = γ_∞ + (γ_0 - γ_∞)(V/V_0)^β: Al'tshuler Grüneisen parameter
    
    All methods take pressure P [Pa] and temperature T [K] as inputs and
    return quantities in SI units.
    
    Attributes
    ----------
    T0 : float
        Reference temperature (300 K)
    molar_mass : float
        Molar mass of MgSiO₃ [kg/mol]
    params : dict
        EoS parameters from fit 8
    
    Examples
    --------
    >>> eos = Sakai16()
    >>> rho = eos.density(P=150e9, T=3000)
    >>> print(f"Density: {rho:.1f} kg/m³")
    
    Notes
    -----
    The post-perovskite structure has Z = 4 formula units per unit cell
    (Cmcm space group), with n = 5 atoms per formula unit.
    
    PPv is predicted to decompose above ~900-1000 GPa. The ab initio EoS
    covers 0 to 1200 GPa; experimental validation extends to ~300 GPa.
    The quasi-harmonic approximation is valid up to ~5000 K.
    """
    
    def __init__(self):
        """
        Initialize the Sakai16 EoS for MgSiO₃ post-perovskite.
        
        Parameters are from Table 1, fit 8 (ab initio, LDA, Keane model):
        - Cold EoS: Keane equation (Supplementary Information A)
        - Thermal model: MGD with Al'tshuler Grüneisen parameter
        - Debye temperature from Supplementary Information A
        """
        # Reference temperature
        self.T0 = 300.0  # K
        
        # Molar mass of MgSiO₃
        self.molar_mass = _M_MgSiO3
        
        # Number of atoms per formula unit
        self.n_atoms = 5  # Mg + Si + 3×O
        
        # Number of formula units per unit cell (Cmcm space group)
        self.Z = 4
        
        # EoS parameters from fit 8 (ab initio, Keane model)
        # Table 1 of Sakai et al. (2016)
        V0_cell = 164.22e-30  # m³/cell
        
        self.params = {
            'U0': -86894.,                                      # J/mol
            'S0': -180.248,                                     # J/(mol·K)
            'V0': V0_cell * N_AVOGADRO / self.Z,                # m³/mol
            'K0': 205.4e9,                                      # Pa
            'K0_prime': 5.069,                                  # dimensionless
            'K_inf_prime': 2.627,                               # dimensionless
            'gamma0': 1.495,                                    # dimensionless
            'gamma_inf': 0.818,                                 # dimensionless
            'beta': 1.97,                                       # dimensionless
            'Theta0': 995.0,                                    # K
            'n': self.n_atoms,                                  # atoms per formula unit
        }
    
    # =========================================================================
    # Core thermodynamic helper functions
    # =========================================================================
    
    def _gruneisen_parameter(self, V: float) -> float:
        """
        Calculate Grüneisen parameter using Al'tshuler et al. form.
        
        γ(V) = γ_∞ + (γ_0 - γ_∞)(V/V_0)^β
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Grüneisen parameter (dimensionless)
        """
        p = self.params
        x = V / p['V0']
        return p['gamma_inf'] + (p['gamma0'] - p['gamma_inf']) * x**p['beta']
    
    def _q_parameter(self, V: float) -> float:
        """
        Calculate q = dln(γ)/dln(V) = V/γ dγ/dV.
        
        Analytical result:
        q = β(γ_0 - γ_∞)(V/V_0)^β / γ
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            q parameter (dimensionless)
        """
        p = self.params
        x = V / p['V0']
        g = self._gruneisen_parameter(V)
        return p['beta'] * (p['gamma0'] - p['gamma_inf']) * x**p['beta'] / g
    
    def _debye_temperature(self, V: float) -> float:
        """
        Calculate Debye temperature using Al'tshuler et al. form.
        
        θ_D(V) = θ_0 (V/V_0)^(-γ_∞) exp{(γ_0 - γ_∞)/β [1 - (V/V_0)^β]}
        
        From Supplementary Information A of Sakai et al. (2016).
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Debye temperature [K]
        """
        p = self.params
        x = V / p['V0']
        
        exponent = (p['gamma0'] - p['gamma_inf']) / p['beta'] * (1 - x**p['beta'])
        
        return p['Theta0'] * x**(-p['gamma_inf']) * np.exp(exponent)
    
    def _debye_integral(self, x_max: float) -> float:
        """
        Calculate the Debye integral: ∫_0^x_max z³/(exp(z) - 1) dz
        
        Parameters
        ----------
        x_max : float
            Upper limit of integration (θ/T)
            
        Returns
        -------
        float
            Value of the Debye integral
        """
        if x_max < 0.01:
            return x_max**3 / 3
        
        if x_max > 100:
            return np.pi**4 / 15
        
        n_points = 1000
        z = np.linspace(0, x_max, n_points)
        z[0] = 1e-10
        integrand = z**3 / (np.exp(z) - 1)
        return np.trapezoid(integrand, z)
    
    def _debye_function_D3(self, x: float) -> float:
        """
        Calculate the third Debye function: D_3(x) = (3/x^3) ∫_0^x z^3/(e^z - 1) dz
        
        Parameters
        ----------
        x : float
            θ/T ratio
            
        Returns
        -------
        float
            D_3(x) value
        """
        integral = self._debye_integral(x)
        return (3 / x**3) * integral
    
    def _thermal_energy(self, V: float, T: float) -> float:
        """
        Calculate Debye thermal energy.
        
        E_th(V,T) = 3nR [3Θ/8 + T·D_3(Θ/T)]
        
        where Θ/8 is the zero-point energy per mode and D_3 is the 
        third Debye function.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Thermal energy [J/mol]
        """
        Theta = self._debye_temperature(V)
        n = self.params['n']
        
        if T < 1e-6:
            return 9 * n * R_GAS * Theta / 8
        
        x = Theta / T
        D3 = self._debye_function_D3(x)
        
        return 3 * n * R_GAS * (3 * Theta / 8 + T * D3)
    
    def _cold_pressure(self, V: float) -> float:
        """
        Calculate cold compression pressure using Keane EoS.
        
        P = K_0 {(K_0'/K_∞'^2)[(V_0/V)^K_∞' - 1] - (K_0'/K_∞' - 1) ln(V_0/V)}
        
        From Supplementary Information A.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Cold compression pressure [Pa]
        """
        p = self.params
        eta = p['V0'] / V  # compression ratio
        K0 = p['K0']
        K0p = p['K0_prime']
        Kinf = p['K_inf_prime']
        
        return K0 * (
            (K0p / Kinf**2) * (eta**Kinf - 1)
            - (K0p / Kinf - 1) * np.log(eta)
        )
    
    def _cold_bulk_modulus(self, V: float) -> float:
        """
        Calculate isothermal bulk modulus from Keane EoS at reference T.
        
        K_cold(V) = -V dP_cold/dV 

        Analytical result with η = V_0/V:
        K_cold(V) = K_0 {(K_0'/K_∞')(η^K_∞' - 1) + 1}
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Cold bulk modulus [Pa]
        """
        p = self.params
        eta = p['V0'] / V
        K0 = p['K0']
        K0p = p['K0_prime']
        Kinf = p['K_inf_prime']
        
        return K0 * ((K0p/Kinf)*(eta**Kinf - 1) + 1)
    
    def _cold_energy(self, V: float) -> float:
        """
        Calculate cold compression energy by integrating Keane pressure.
        
        E_cold(V) = -∫_{V_0}^{V} P_cold(V') dV'
        
        Analytical result with η = V_0/V:
        E_cold = K_0 V_0 {(K_0'/K_∞'^2)[(η^(K_∞'-1) - 1)/(K_∞' - 1) + (1/η - 1)]
                        + (K_0'/K_∞' - 1)[(ln(η) + 1)/η - 1]}
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Cold compression energy [J/mol]
        """
        p = self.params
        eta = p['V0'] / V
        V0 = p['V0']
        K0 = p['K0']
        K0p = p['K0_prime']
        Kinf = p['K_inf_prime']
        
        # Term 1: from K_0'/K_∞'^2 × (η^K_∞' - 1) part
        term1 = (K0p / Kinf**2) * (
            (eta**(Kinf - 1) - 1)/(Kinf - 1) + 1/eta - 1
        )
        
        # Term 2: from -(K_0'/K_∞' - 1) × ln(η) part
        term2 = (K0p / Kinf - 1) * ((np.log(eta) + 1)/eta - 1)
        
        return K0 * V0 * (term1 + term2)
    
    def _thermal_pressure(self, V: float, T: float) -> float:
        """
        Calculate thermal pressure from MGD model.
        
        ΔP_th(V,T) = (γ/V)[E_th(V,T) - E_th(V,T_0)]
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Thermal pressure [Pa]
        """
        if abs(T - self.T0) < 1e-6:
            return 0.0
        
        gamma = self._gruneisen_parameter(V)
        E_T = self._thermal_energy(V, T)
        E_T0 = self._thermal_energy(V, self.T0)
        
        return (gamma / V) * (E_T - E_T0)
    
    def _total_pressure(self, V: float, T: float) -> float:
        """
        Calculate total pressure: P(V,T) = P_cold(V) + ΔP_th(V,T).
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Total pressure [Pa]
        """
        return self._cold_pressure(V) + self._thermal_pressure(V, T)
    
    def _find_volume(self, P: float, T: float) -> float:
        """
        Find molar volume for given pressure and temperature.
        
        Solves P(V,T) = P_target using Brent's method.
        
        Parameters
        ----------
        P : float
            Target pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Molar volume [m³/mol]
            
        Raises
        ------
        RuntimeError
            If root finding fails to converge
        """
        V0 = self.params['V0']
        
        # PPv is a high-pressure phase; search range spans from
        # highly compressed (~0.2 V_0 for TPa) to slightly expanded
        V_min = 0.08 * V0
        V_max = 1.20 * V0
        
        def pressure_residual(V):
            return self._total_pressure(V, T) - P
        
        try:
            V = brentq(pressure_residual, V_min, V_max, xtol=1e-12)
            return V
        except ValueError:
            raise RuntimeError(
                f"Failed to find volume at P = {P/1e9:.2f} GPa, T = {T:.2f} K. "
                f"Pressure may be outside valid range for Sakai16 PPv EoS."
            )
    
    # =========================================================================
    # Thermodynamic property helpers (molar quantities)
    # =========================================================================
    
    def _isochoric_heat_capacity_molar(self, V: float, T: float) -> float:
        """
        Calculate molar isochoric heat capacity.
        
        C_V = (∂E_th/∂T)_V
        
        For Debye model:
        C_V = 9nR (T/Θ)^3 ∫_0^(Θ/T) x^4 exp(x)/(exp(x) - 1)^2 dx
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Isochoric heat capacity [J/(mol·K)]
        """
        if T < 1e-6:
            return 0.0
        
        Theta = self._debye_temperature(V)
        n = self.params['n']
        
        x_max = Theta / T
        
        # High temperature limit: C_V → 3nR (Dulong-Petit)
        if x_max < 0.01:
            return 3 * n * R_GAS
        
        # Low temperature limit: C_V → 0
        if x_max > 100:
            return 0.0
        
        n_points = 1000
        z = np.linspace(1e-10, x_max, n_points)
        exp_z = np.exp(z)
        integrand = z**4 * exp_z / (exp_z - 1)**2
        integral = np.trapezoid(integrand, z)
        
        return 9 * n * R_GAS * (T / Theta)**3 * integral
    
    def _isothermal_bulk_modulus(self, V: float, T: float) -> float:
        """
        Calculate isothermal bulk modulus: K_T = K_cold + K_th.
        
        The thermal correction uses the Al'tshuler Grüneisen parameter
        and its volume derivative.
        
        K_th = -V (∂P_th/∂V)_T, where P_th = (γ/V)(E_th - E_th0)
        
        dP_th/dV = (dγ/dV · 1/V - γ/V^2)(E_th - E_th0)
                   + (γ/V)(dE_th/dV - dE_th0/dV)
        
        Using dE_th/dV = γ C_V T / V - P_th (from the Debye model):
        K_th = (1 + γ - q)P_th - γ^2/V (T·C_V(T) - T_0·C_V(T_0))
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Isothermal bulk modulus [Pa]
        """
        K_cold = self._cold_bulk_modulus(V)
        
        gamma = self._gruneisen_parameter(V)
        q = self._q_parameter(V)
        
        P_th = self._thermal_pressure(V, T)
        Cv_T = self._isochoric_heat_capacity_molar(V, T)
        Cv_T0 = self._isochoric_heat_capacity_molar(V, self.T0)
        
        K_th = (1 + gamma - q)*P_th - (gamma**2/V)*(T*Cv_T - self.T0*Cv_T0)
        
        return K_cold + K_th
    
    def _thermal_expansion_coeff(self, V: float, T: float) -> float:
        """
        Calculate thermal expansion coefficient.
        
        α = (∂P/∂T)_V / K_T = (γ/V) C_V / K_T
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Thermal expansion coefficient [K⁻¹]
        """
        if T < 1e-6:
            return 0.0
        
        gamma = self._gruneisen_parameter(V)
        Cv = self._isochoric_heat_capacity_molar(V, T)
        KT = self._isothermal_bulk_modulus(V, T)
        
        if abs(KT) < 1e-6:
            return 0.0
        
        dP_dT = (gamma / V) * Cv
        return dP_dT / KT
    
    def _entropy_molar(self, V: float, T: float) -> float:
        """
        Calculate molar entropy.
        
        For Debye model (e.g., Gopal 1966, equation 2.16b):
        S = nR[4D_3(θ_D/T) - 3ln(1 - exp(-θ_D/T))] + S_0
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Molar entropy [J/(mol·K)]
        """
        S0 = self.params['S0']
        
        if T < 1e-6:
            return S0
        
        Theta = self._debye_temperature(V)
        n = self.params['n']
        
        x = Theta / T
        
        if x > 100:
            return S0
        
        D3 = self._debye_function_D3(x)
        S_Debye = n * R_GAS * (4 * D3 - 3 * np.log(1 - np.exp(-x)))
        
        return S_Debye + S0
    
    def _internal_energy_molar(self, V: float, T: float) -> float:
        """
        Calculate molar internal energy.
        
        E = E_cold(V) + E_th(V,T) - E_th(V,T_0) + U_0
        
        where E_cold is obtained from the analytical integral of the Keane
        pressure (see _cold_energy).
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Molar internal energy [J/mol]
        """
        U0 = self.params['U0']
        
        E_cold = self._cold_energy(V)
        E_th = self._thermal_energy(V, T)
        E_th_ref = self._thermal_energy(V, self.T0)
        
        return U0 + E_cold + (E_th - E_th_ref)
    
    # =========================================================================
    # Public interface
    # =========================================================================
    
    def density(self, P: float, T: float) -> float:
        """
        Calculate density at given pressure and temperature.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Density [kg/m³]
        """
        V = self._find_volume(P, T)
        return self.molar_mass / V
    
    def specific_internal_energy(self, P: float, T: float) -> float:
        """
        Calculate specific internal energy.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific internal energy [J/kg]
        """
        V = self._find_volume(P, T)
        E_molar = self._internal_energy_molar(V, T)
        return E_molar / self.molar_mass
    
    def specific_entropy(self, P: float, T: float) -> float:
        """
        Calculate specific entropy.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific entropy [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        S_molar = self._entropy_molar(V, T)
        return S_molar / self.molar_mass
    
    def isobaric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isobaric heat capacity.
        
        C_P = C_V + α² T V K_T
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific isobaric heat capacity [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        Cv = self._isochoric_heat_capacity_molar(V, T)
        alpha = self._thermal_expansion_coeff(V, T)
        KT = self._isothermal_bulk_modulus(V, T)
        
        Cp_molar = Cv + alpha**2 * T * V * KT
        return Cp_molar / self.molar_mass
    
    def isochoric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isochoric heat capacity.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific isochoric heat capacity [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        Cv_molar = self._isochoric_heat_capacity_molar(V, T)
        return Cv_molar / self.molar_mass
    
    def thermal_expansion(self, P: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Thermal expansion coefficient [K⁻¹]
        """
        V = self._find_volume(P, T)
        return self._thermal_expansion_coeff(V, T)
    
    def adiabatic_gradient(self, P: float, T: float) -> float:
        """
        Calculate dimensionless adiabatic temperature gradient.
        
        (∂ln T/∂ln P)_S = α P / (ρ C_P)
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Adiabatic gradient (dimensionless)
        """
        alpha = self.thermal_expansion(P, T)
        Cp = self.isobaric_heat_capacity(P, T)
        rho = self.density(P, T)
        
        return alpha * P / (Cp * rho)


class Sokolova22:
    """
    Equation of state for MgSiO₃ pyroxene phases from Sokolova et al. (2022).

    Reference:
    Sokolova, T.S., Dorogokupets, P.I., Filippova, A.I., Dymshits, A.M.,
    Danilov, B.S., Litasov, K.D. (2022)
    "Equations of state for MgSiO₃ phases at pressures up to 12 GPa"
    Physics and Chemistry of Minerals 49:37,
    DOI: 10.1007/s00269-022-01212-7

    This implementation covers three pyroxene phases:
    - LP-CEn: low-pressure clinoenstatite (P2₁/c), stable at low T and low P
    - OrthoEn: orthoenstatite (Pbca), stable at T > ~600 K and P < ~6 GPa
    - HP-CEn: high-pressure clinoenstatite (C2/c), stable at P > ~6 GPa

    The thermodynamic model is based on the Helmholtz free energy decomposition:
    F(V,T) = U₀ + F_T₀(V) + F_th(V,T) + F_anh(V,T)

    where:
    - U₀: reference energy
    - F_T₀(V): potential part at reference isotherm T₀ = 298.15 K
      (Kunc equation with k = 5, i.e. HO2 form)
    - F_th(V,T): thermal contribution (Einstein model with two characteristic
      temperatures, volume-dependent via Al'tshuler formulation)
    - F_anh(V,T): intrinsic anharmonicity contribution

    All methods take pressure P [Pa] and temperature T [K] as inputs and
    return quantities in SI units.

    Parameters
    ----------
    phase : str
        Phase of MgSiO₃ pyroxene: 'lp-cen', 'orthoen', or 'hp-cen'

    Attributes
    ----------
    phase : str
        Selected phase
    T0 : float
        Reference temperature (298.15 K)
    params : dict
        Active parameters for the selected phase
    n_atoms : int
        Number of atoms per formula unit (5 for MgSiO₃)

    Examples
    --------
    >>> eos = Sokolova22(phase='orthoen')
    >>> rho = eos.density(P=1e5, T=300)
    >>> print(f"Density: {rho:.1f} kg/m³")

    >>> eos_hp = Sokolova22(phase='hp-cen')
    >>> cp = eos_hp.isobaric_heat_capacity(P=8e9, T=1500)

    Notes
    -----
    The reference isotherm uses the Kunc equation with k = 5 (HO2 form):
        P_T₀(V) = 3 K₀ X^(-k) (1 - X) exp[η(1 - X)]
    where X = (V/V₀)^(1/3) and η = 1.5 K' - k + 0.5.

    The volume dependence of characteristic temperatures follows the
    Al'tshuler formulation:
        Θᵢ(V) = Θ₀ᵢ x^(-γ∞) exp[(γ₀ - γ∞)/β (1 - x^β)]
    where x = V/V₀.

    The methodology and thermodynamic framework are described in detail
    in Sokolova & Dorogokupets (2021), Minerals 11:322.
    """

    def __init__(self, phase: str = 'orthoen'):
        """
        Initialize the Sokolova22 EoS for a specific MgSiO₃ pyroxene phase.

        Parameters
        ----------
        phase : str, optional
            Phase of MgSiO₃: 'lp-cen', 'orthoen', or 'hp-cen'.
            Default is 'orthoen'.

        Raises
        ------
        ValueError
            If phase is not one of the valid options.
        """
        phase = phase.lower()
        if phase not in ['lp-cen', 'orthoen', 'hp-cen']:
            raise ValueError(
                f"Invalid phase '{phase}'. "
                f"Must be 'lp-cen', 'orthoen', or 'hp-cen'."
            )

        self.phase = phase
        self.T0 = 298.15   # K, reference temperature
        self.n_atoms = 5    # atoms per formula unit in MgSiO₃
        self.molar_mass = _M_MgSiO3  # kg/mol

        # =====================================================================
        # EoS parameters from Table 2 of Sokolova et al. (2022)
        # =====================================================================

        if phase == 'lp-cen':
            self.params = {
                'U0': -99915.,        # J/mol
                'S0': -191.087,       # J/(mol·K)
                'V0': 31.473e-6,      # m³/mol
                'K0': 102.8e9,        # Pa
                'K0_prime': 8.40,     # dimensionless
                'theta01': 368.0,     # K
                'theta02': 960.0,     # K
                'm1': 6.0,            # dimensionless
                'm2': 9.0,            # dimensionless
                'gamma0': 0.85,       # dimensionless
                'gamma_inf': 0.0,     # dimensionless
                'beta': 1.0,          # dimensionless
                'a0': 17.0e-6,        # K⁻¹
                'm_anh': 1.0,         # dimensionless
            }
        elif phase == 'orthoen':
            self.params = {
                'U0': -101463.,       # J/mol
                'S0': -202.356,       # J/(mol·K)
                'V0': 31.347e-6,      # m³/mol
                'K0': 106.2e9,        # Pa
                'K0_prime': 7.80,     # dimensionless
                'theta01': 327.0,     # K
                'theta02': 973.0,     # K
                'm1': 6.785,          # dimensionless
                'm2': 8.215,          # dimensionless
                'gamma0': 1.0,        # dimensionless
                'gamma_inf': 0.0,     # dimensionless
                'beta': 1.0,          # dimensionless
                'a0': 13.0e-6,        # K⁻¹
                'm_anh': 1.0,         # dimensionless
            }
        elif phase == 'hp-cen':
            self.params = {
                'U0': -99705.,        # J/mol
                'S0': -192.88,       # J/(mol·K)
                'V0': 30.310e-6,      # m³/mol
                'K0': 112.0e9,        # Pa
                'K0_prime': 6.20,     # dimensionless
                'theta01': 373.0,     # K
                'theta02': 985.0,     # K
                'm1': 6.785,          # dimensionless
                'm2': 8.215,          # dimensionless
                'gamma0': 0.745,      # dimensionless
                'gamma_inf': 0.0,     # dimensionless
                'beta': 1.0,          # dimensionless
                'a0': 0.0,            # K⁻¹
                'm_anh': 1.0,         # dimensionless
            }

        # Derived constant: Kunc equation parameter
        # η = 1.5 K' - k + 0.5, where k = 5 for HO2 form
        self.params['eta'] = 1.5 * self.params['K0_prime'] - 5.0 + 0.5
        self.params['k'] = 5  # Kunc equation order

    # =========================================================================
    # Helper functions for EoS components
    # =========================================================================

    def _gruneisen_parameter(self, V: float) -> float:
        """
        Calculate Grüneisen parameter as a function of volume.

        Al'tshuler formulation:
            γ(V) = γ∞ + (γ₀ - γ∞) x^β

        where x = V/V₀.

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]

        Returns
        -------
        float
            Grüneisen parameter (dimensionless)
        """
        x = V / self.params['V0']
        g0 = self.params['gamma0']
        ginf = self.params['gamma_inf']
        beta = self.params['beta']

        return ginf + (g0 - ginf) * x**beta

    def _einstein_temperature(self, V: float, i: int) -> float:
        """
        Calculate Einstein characteristic temperature as a function of volume.

        Al'tshuler formulation:
            Θᵢ(V) = Θ₀ᵢ x^(-γ∞) exp[(γ₀ - γ∞)/β (1 - x^β)]

        where x = V/V₀.

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        i : int
            Index of Einstein mode (1 or 2)

        Returns
        -------
        float
            Einstein characteristic temperature [K]
        """
        x = V / self.params['V0']
        g0 = self.params['gamma0']
        ginf = self.params['gamma_inf']
        beta = self.params['beta']

        theta0 = self.params[f'theta0{i}']

        return theta0 * x**(-ginf) * np.exp((g0 - ginf) / beta * (1.0 - x**beta))

    def _q_parameter(self, V: float) -> float:
        """
        Calculate q = dln(γ)/dln(V) = V/γ dγ/dV.

        Analytical result:
            q = β (γ₀ - γ∞) (V/V₀)^β / γ

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]

        Returns
        -------
        float
            q parameter (dimensionless)
        """
        p = self.params
        x = V / p['V0']
        g = self._gruneisen_parameter(V)

        if abs(g) < 1e-10:
            return 0.0

        return p['beta'] * (p['gamma0'] - p['gamma_inf']) * x**p['beta'] / g

    def _anharmonic_parameter(self, V: float) -> float:
        """
        Calculate the volume-dependent anharmonicity parameter a(V).

            a(V) = a₀ × (V/V₀)^m

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]

        Returns
        -------
        float
            Anharmonicity parameter [K⁻¹]
        """
        x = V / self.params['V0']
        return self.params['a0'] * x**self.params['m_anh']

    # =========================================================================
    # Pressure components
    # =========================================================================

    def _cold_pressure(self, V: float) -> float:
        """
        Calculate cold (reference isotherm) pressure using the Kunc equation.

        HO2 form (k = 5):
            P_T₀(V) = 3 K₀ X^(-k) (1 - X) exp[η(1 - X)]

        where X = (V/V₀)^(1/3) and η = 1.5 K' - k + 0.5.

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]

        Returns
        -------
        float
            Cold pressure at T₀ [Pa]
        """
        K0 = self.params['K0']
        eta = self.params['eta']
        k = self.params['k']

        X = (V / self.params['V0'])**(1.0 / 3.0)

        return 3.0 * K0 * X**(-k) * (1.0 - X) * np.exp(eta * (1.0 - X))

    def _cold_bulk_modulus(self, V: float) -> float:
        """
        Calculate isothermal bulk modulus at the reference isotherm.

        Sokolova & Dorogokupets (2021), Eq. 3:
            K_T₀(V) = K₀ X^(-k) exp[η(1-X)] [X + (1-X)(ηX + k)]

        where X = (V/V₀)^(1/3), η = 1.5K' - k + 0.5, and k = 5.

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]

        Returns
        -------
        float
            Cold bulk modulus [Pa]
        """
        K0 = self.params['K0']
        eta = self.params['eta']
        k = self.params['k']

        X = (V / self.params['V0'])**(1.0 / 3.0)

        return K0 * X**(-k) * np.exp(eta * (1.0 - X)) * (
            X + (1.0 - X) * (eta * X + k)
        )

    def _cold_energy(self, V: float) -> float:
        """
        Calculate cold compression energy by integrating Kunc pressure.

            E_cold(V) = -∫_{V₀}^{V} P_cold(V') dV'

        This is the potential part of the Helmholtz free energy at the
        reference isotherm, positive for compression (V < V₀).

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]

        Returns
        -------
        float
            Cold compression energy [J/mol]
        """
        from scipy.integrate import quad

        V0 = self.params['V0']

        if abs(V - V0) / V0 < 1e-12:
            return 0.0

        result, _ = quad(self._cold_pressure, V0, V)
        return -result + self.params['U0']

    def _thermal_energy(self, V: float, T: float) -> float:
        """
        Calculate thermal energy from the two-mode Einstein model.

            E_th = Σᵢ mᵢ R Θᵢ / [exp(Θᵢ/T) - 1]

        Note: mᵢ already includes the factor of n atoms (m₁ + m₂ = 3n = 15).

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Thermal energy [J/mol]
        """
        E_th = 0.0
        for i in [1, 2]:
            mi = self.params[f'm{i}']
            theta_i = self._einstein_temperature(V, i)
            y = theta_i / T

            if y > 500:
                continue  # negligible contribution
            else:
                E_th += mi * R_GAS * theta_i / (np.exp(y) - 1.0)

        return E_th

    def _thermal_pressure(self, V: float, T: float) -> float:
        """
        Calculate thermal pressure.

            P_th = (γ/V) × E_th(V,T)

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Thermal pressure [Pa]
        """
        gamma = self._gruneisen_parameter(V)
        E_th = self._thermal_energy(V, T)
        return gamma * E_th / V

    def _anharmonic_energy(self, V: float, T: float) -> float:
        """
        Calculate anharmonic internal energy.

            E_anh = (3/2) n R a(V) T²

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Anharmonic energy [J/mol]
        """
        a = self._anharmonic_parameter(V)
        return 1.5 * self.n_atoms * R_GAS * a * T**2

    def _anharmonic_pressure(self, V: float, T: float) -> float:
        """
        Calculate anharmonic pressure contribution.

            P_anh = (m_anh / V) × E_anh

        This follows from -(∂F_anh/∂V)_T where F_anh = -(3/2)nRa(V)T²
        and a(V) = a₀(V/V₀)^m, giving da/dV = m·a/V.

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Anharmonic pressure [Pa]
        """
        m = self.params['m_anh']
        E_anh = self._anharmonic_energy(V, T)
        return m * E_anh / V

    def _total_pressure(self, V: float, T: float) -> float:
        """
        Calculate total pressure at given volume and temperature.

            P(V,T) = P_cold(V) + P_th(V,T) + P_anh(V,T)

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Total pressure [Pa]
        """
        return (self._cold_pressure(V)
                + self._thermal_pressure(V, T)
                + self._anharmonic_pressure(V, T))

    def _find_volume(self, P: float, T: float) -> float:
        """
        Find molar volume for given pressure and temperature by root finding.

        Uses Brent's method to solve P(V,T) = P_target.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Molar volume [m³/mol]

        Raises
        ------
        RuntimeError
            If root finding fails to converge.
        """
        V0 = self.params['V0']

        # Bounds: allow compression to 0.5 V₀ and expansion to 1.5 V₀
        # Pyroxenes are low-pressure phases (P < ~12 GPa)
        V_min = 0.5 * V0
        V_max = 1.5 * V0

        def pressure_residual(V):
            return self._total_pressure(V, T) - P

        try:
            V = brentq(pressure_residual, V_min, V_max, xtol=1e-15, rtol=1e-12)
            return V
        except ValueError:
            raise RuntimeError(
                f"Could not find volume for P = {P / 1e9:.2f} GPa, T = {T:.1f} K. "
                f"Pressure may be outside valid range for {self.phase} phase."
            )

    # =========================================================================
    # Thermodynamic property calculations (V, T)
    # =========================================================================

    def _isochoric_heat_capacity(self, V: float, T: float) -> float:
        """
        Calculate molar isochoric heat capacity.

            C_V = Σᵢ mᵢ R (Θᵢ/T)² exp(Θᵢ/T) / [exp(Θᵢ/T) - 1]²
                  + 3 n R a(V) T

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Molar C_V [J/(mol·K)]
        """
        Cv = 0.0
        for i in [1, 2]:
            mi = self.params[f'm{i}']
            theta_i = self._einstein_temperature(V, i)
            y = theta_i / T

            if y > 500:
                continue
            else:
                ey = np.exp(y)
                Cv += mi * R_GAS * y**2 * ey / (ey - 1.0)**2

        # Anharmonic contribution
        a = self._anharmonic_parameter(V)
        Cv += 3.0 * self.n_atoms * R_GAS * a * T

        return Cv

    def _entropy(self, V: float, T: float) -> float:
        """
        Calculate molar entropy.

        Sokolova & Dorogokupets (2021), Eq. 8 + anharmonic (Eq. 15):
            S = S₀ + Σᵢ mᵢ R { -ln[1 - exp(-Θᵢ/T)] + (Θᵢ/T) / [exp(Θᵢ/T) - 1] }
                + 3 n R a(V) T

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Molar entropy [J/(mol·K)]
        """
        S = self.params['S0']

        for i in [1, 2]:
            mi = self.params[f'm{i}']
            theta_i = self._einstein_temperature(V, i)
            y = theta_i / T

            if y > 500:
                continue
            else:
                S += mi * R_GAS * (-np.log(1.0 - np.exp(-y)) + y / (np.exp(y) - 1.0))

        # Anharmonic entropy
        a = self._anharmonic_parameter(V)
        S += 3.0 * self.n_atoms * R_GAS * a * T

        return S

    def _internal_energy(self, V: float, T: float) -> float:
        """
        Calculate molar internal energy.

            U = E_cold(V) + E_th(V,T) + E_anh(V,T)

        where E_cold = -∫_{V₀}^{V} P_cold dV' is the cold compression
        energy from the Kunc reference isotherm.

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Internal energy [J/mol]
        """
        return (self._cold_energy(V)
                + self._thermal_energy(V, T)
                + self._anharmonic_energy(V, T))

    def _thermal_pressure_dT(self, V: float, T: float) -> float:
        """
        Calculate (∂P/∂T)_V for thermal expansion computation.

            (∂P/∂T)_V = (γ/V) C_V,harm + (m/V) C_V,anh

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            (∂P/∂T)_V [Pa/K]
        """
        gamma = self._gruneisen_parameter(V)
        m = self.params['m_anh']

        # Harmonic Cv
        Cv_harm = 0.0
        for i in [1, 2]:
            mi = self.params[f'm{i}']
            theta_i = self._einstein_temperature(V, i)
            y = theta_i / T

            if y > 500:
                continue
            else:
                ey = np.exp(y)
                Cv_harm += mi * R_GAS * y**2 * ey / (ey - 1.0)**2

        # Anharmonic Cv
        a = self._anharmonic_parameter(V)
        Cv_anh = 3.0 * self.n_atoms * R_GAS * a * T

        return gamma * Cv_harm / V + m * Cv_anh / V

    def _isothermal_bulk_modulus(self, V: float, T: float) -> float:
        """
        Calculate isothermal bulk modulus K_T = K_cold + K_th + K_anh.

        Uses analytical expressions from Sokolova & Dorogokupets (2021):
        - K_cold: Eq. 3
        - K_th:   Eq. 12
        - K_anh:  Eq. 15

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Isothermal bulk modulus [Pa]
        """
        K_cold = self._cold_bulk_modulus(V)
        K_th = self._thermal_bulk_modulus(V, T)
        K_anh = self._anharmonic_bulk_modulus(V, T)

        return K_cold + K_th + K_anh

    def _thermal_bulk_modulus(self, V: float, T: float) -> float:
        """
        Calculate thermal contribution to isothermal bulk modulus.

        Sokolova & Dorogokupets (2021), Eq. 12:
            K_T,th = P_th (1 + γ - q) - γ² C_V,harm T / V

        where C_V,harm is the harmonic (Einstein) heat capacity from Eq. 11.

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Thermal bulk modulus [Pa]
        """
        gamma = self._gruneisen_parameter(V)
        q = self._q_parameter(V)
        P_th = self._thermal_pressure(V, T)

        # Harmonic heat capacity only (Eq. 11)
        Cv_harm = 0.0
        for i in [1, 2]:
            mi = self.params[f'm{i}']
            theta_i = self._einstein_temperature(V, i)
            y = theta_i / T
            if y > 500:
                continue
            ey = np.exp(y)
            Cv_harm += mi * R_GAS * y**2 * ey / (ey - 1.0)**2

        return P_th * (1.0 + gamma - q) - gamma**2 * Cv_harm * T / V

    def _anharmonic_bulk_modulus(self, V: float, T: float) -> float:
        """
        Calculate anharmonic contribution to isothermal bulk modulus.

        Sokolova & Dorogokupets (2021), Eq. 15:
            K_T,anh = P_anh (1 - m)

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Anharmonic bulk modulus [Pa]
        """
        return self._anharmonic_pressure(V, T) * (1.0 - self.params['m_anh'])

    def _thermal_expansion_coeff(self, V: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient.

            α = (∂P/∂T)_V / K_T

        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Thermal expansion coefficient [K⁻¹]
        """
        dP_dT = self._thermal_pressure_dT(V, T)
        K_T = self._isothermal_bulk_modulus(V, T)

        return dP_dT / K_T

    # =========================================================================
    # Public methods: all take P [Pa] and T [K], return SI units
    # =========================================================================

    def density(self, P: float, T: float) -> float:
        """
        Calculate density.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Density [kg/m³]
        """
        V = self._find_volume(P, T)
        return self.molar_mass / V

    def specific_internal_energy(self, P: float, T: float) -> float:
        """
        Calculate specific internal energy.

        Referenced to (V₀, T₀): includes cold, thermal, and anharmonic
        contributions.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific internal energy [J/kg]
        """
        V = self._find_volume(P, T)
        V0 = self.params['V0']

        U_mol = self._internal_energy(V, T)

        return U_mol / self.molar_mass

    def specific_entropy(self, P: float, T: float) -> float:
        """
        Calculate specific entropy.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific entropy [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        V0 = self.params['V0']

        S_mol = self._entropy(V, T)

        return S_mol / self.molar_mass

    def isobaric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isobaric heat capacity.

            C_P = C_V + α² T V K_T

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific isobaric heat capacity [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        Cv_mol = self._isochoric_heat_capacity(V, T)
        alpha = self._thermal_expansion_coeff(V, T)
        K_T = self._isothermal_bulk_modulus(V, T)

        Cp_mol = Cv_mol + alpha**2 * T * V * K_T

        return Cp_mol / self.molar_mass

    def isochoric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isochoric heat capacity.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific isochoric heat capacity [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        Cv_mol = self._isochoric_heat_capacity(V, T)

        return Cv_mol / self.molar_mass

    def thermal_expansion(self, P: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Thermal expansion coefficient [K⁻¹]
        """
        V = self._find_volume(P, T)
        return self._thermal_expansion_coeff(V, T)

    def adiabatic_gradient(self, P: float, T: float) -> float:
        """
        Calculate dimensionless adiabatic temperature gradient.

            (∂ln T/∂ln P)_S = α P / (ρ C_P)

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Adiabatic gradient (dimensionless)
        """
        alpha = self.thermal_expansion(P, T)
        Cp = self.isobaric_heat_capacity(P, T)
        rho = self.density(P, T)

        return alpha * P / (Cp * rho)


class Wolf18:
    """
    Equation of state for liquid MgSiO₃ using the RTpress model.

    Reference:
    Wolf, A.S., Bower, D.J. (2018)
    "An equation of state for high pressure-temperature liquids (RTpress)
    with application to MgSiO₃ melt"
    Physics of the Earth and Planetary Interiors, 278, 59-74,
    DOI: 10.1016/j.pepi.2018.02.004

    This implementation wraps the RTpress package (Wolf & Bower 2018) as
    a backend, using the parameter set from Luo & Deng (2025) Table 1.
    Parameters are calibrated against ab initio molecular dynamics
    simulations (PBEsol) of MgSiO₃ liquid covering 0-1200 GPa and
    2200-14000 K.

    The RTpress model builds a complete liquid free energy from three
    components:
    - Vinet cold compression along the reference adiabat (Eq. 3-4)
    - Finite-strain Grüneisen model for the reference adiabat
      temperature profile (Eq. 5-7)
    - RT thermal model for off-adiabat excursions using a
      stretched-exponential heat capacity parameterization (Eq. 8-13)

    All thermodynamic quantities are derived analytically from the
    Helmholtz free energy via symbolic differentiation (sympy), then
    compiled to numpy functions for evaluation.

    The Luo & Deng (2025) parameterization captures the non-monotonic
    Grüneisen parameter behavior with compression, with average
    residuals of 3 GPa in pressure and 0.06 eV/atom in energy across
    the full P-T range up to 1200 GPa.

    All methods take pressure P [Pa] and temperature T [K] as inputs
    and return quantities in SI units unless otherwise specified.

    Attributes
    ----------
    T0 : float
        Reference temperature (3000 K)
    molar_mass : float
        MgSiO₃ molar mass [kg/mol]
    n_atoms : int
        Number of atoms per formula unit (5)

    Examples
    --------
    >>> eos = Wolf18()
    >>> rho = eos.density(P=50e9, T=3000)
    >>> print(f"Density: {rho:.1f} kg/m³")

    Notes
    -----
    Requires the RTpress package (Wolf & Bower 2018) and sympy.
    RTpress performs symbolic construction of all thermodynamic
    derivatives at initialization, so the first instantiation incurs
    a one-time setup cost (~seconds).

    The volume solver converts from the PALEOS (P, T) interface to
    the RTpress native (V, T) interface using Brent's method.

    Internal RTpress units (atomic basis): V in Å³/atom, P in GPa,
    E in GPa·Å³/atom, S in GPa·Å³/(K·atom). All conversions to SI
    are handled internally.
    """

    # Unit conversion constants
    _PV_UNIT = 160.21766208          # GPa·Å³/eV
    _AMU_TO_KG = 1.66053906660e-27   # kg/amu
    _ANG3_TO_M3 = 1e-30              # m³/Å³
    _GPA_TO_PA = 1e9                 # Pa/GPa
    _GPA_ANG3_TO_J = 1e-21           # J/(GPa·Å³)

    def __init__(self):
        """
        Initialize the Wolf18 EoS for liquid MgSiO₃.

        Creates an RTpress instance with the Luo & Deng (2025)
        parameter set (Table 1) and precompiles all thermodynamic
        derivative functions.

        Raises
        ------
        ImportError
            If RTpress or sympy is not available
        """
        from RTpress import RTpress as _RTpress

        # MgSiO₃: 5 atoms per formula unit
        self.n_atoms = 5
        self.molar_mass = _M_MgSiO3  # 0.100387 kg/mol

        # Average atomic mass [amu/atom]
        self._mavg = self.molar_mass * 1e3 / self.n_atoms  # 20.0774

        # Reference state
        self.T0 = 3000.0  # K
        self._V0 = 14.352  # Å³/atom

        # Initialize RTpress backend (atomic basis: V in Å³/atom, P in GPa)
        self._rtpress = _RTpress(N=self.n_atoms, mavg=self._mavg, basis='atomic')

        # Luo & Deng (2025) parameter set (Table 1)
        # Calibrated against PBEsol AIMD data for MgSiO₃ liquid,
        # 0-1200 GPa, 2200-14000 K
        PV = self._PV_UNIT
        param_values = np.array([
            14.352,             # V0 [Å³/atom]
            3000.0,             # T0 [K]
            0.0,                # S0 [GPa·Å³/(K·atom)]
            13.53,              # K0 [GPa]
            6.767,              # KP0 [dimensionless]
            -6.399 * PV,        # E0 [GPa·Å³/atom]
            0.158,              # gamma0 [dimensionless]
            -1.710,             # gammaP0 [dimensionless]
            0.6,                # m [dimensionless]
            1.763,              # b0 [GPa·Å³/atom]
            0.982,              # b1 [GPa·Å³/atom]
            2.11,               # b2 [GPa·Å³/atom]
            0.37,               # b3 [GPa·Å³/atom]
            1.9,                # b4 [GPa·Å³/atom]
        ])
        self._rtpress.set_params(param_values)

        # Reference-state values for energy/entropy offsets
        self.U0 = 4565746.      # J/mol
        self.S0 = 801.872       # J/(mol·K)

        # Unit conversion factor: GPa·Å³/atom → J/kg
        # (also applies to entropy and heat capacity per-K quantities)
        self._energy_to_SI = self._GPA_ANG3_TO_J / (self._mavg * self._AMU_TO_KG)

        # Unit conversion factor: amu/Å³ → kg/m³
        self._rho_to_SI = self._AMU_TO_KG / self._ANG3_TO_M3

    # =========================================================================
    # Volume solver
    # =========================================================================

    def _find_volume(self, P: float, T: float) -> float:
        """
        Find atomic volume for given pressure and temperature.

        Solves P_RTpress(V, T) = P_target using Brent's method.

        Parameters
        ----------
        P : float
            Target pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Atomic volume [Å³/atom]

        Raises
        ------
        RuntimeError
            If root finding fails to converge
        """
        P_GPa = P / self._GPA_TO_PA

        V_min = 0.70 * self._V0
        V_max = 1.04 * self._V0

        def residual(V):
            return self._rtpress.eval_press(V, T) - P_GPa

        try:
            V = brentq(residual, V_min, V_max, xtol=1e-14, rtol=1e-12)
            return V
        except ValueError:
            # Empirical integration brackets ensuring maximal convergence
            if T > 4000. and P_GPa < 1.:
                V_min = 1. * self._V0
                V_max = 2. * self._V0
            else:
                V_min = 0.60 * self._V0
                V_max = 1.15 * self._V0
            try:
                V = brentq(residual, V_min, V_max, xtol=1e-14, rtol=1e-12)
                return V
            except ValueError:
                try:
                    V_min = 0.01 * self._V0
                    V_max = 1.00 * self._V0
                    V = brentq(residual, V_min, V_max, xtol=1e-14, rtol=1e-12)
                    return V
                except ValueError:
                    raise RuntimeError(
                        f"Failed to find volume at P = {P/1e9:.2f} GPa, T = {T:.1f} K. "
                        f"Pressure may be outside valid range for Wolf18 liquid "
                        f"MgSiO₃ EoS."
            )

    # =========================================================================
    # Public methods: all take P [Pa] and T [K], return SI units
    # =========================================================================

    def density(self, P: float, T: float) -> float:
        """
        Calculate density.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Density [kg/m³]
        """
        V = self._find_volume(P, T)
        rho_atomic = self._rtpress.eval_rho(V)
        return rho_atomic * self._rho_to_SI

    def specific_internal_energy(self, P: float, T: float) -> float:
        """
        Calculate specific internal energy.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific internal energy [J/kg]
        """
        V = self._find_volume(P, T)
        E = self._rtpress.eval_energy(V, T)*self._energy_to_SI + self.U0/self.molar_mass
        return E

    def specific_entropy(self, P: float, T: float) -> float:
        """
        Calculate specific entropy.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific entropy [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        S = self._rtpress.eval_entropy(V, T)*self._energy_to_SI + self.S0/self.molar_mass
        return S

    def isobaric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isobaric heat capacity.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific isobaric heat capacity [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        Cp = self._rtpress.eval_heat_capacity(V, T, const='P')
        return Cp * self._energy_to_SI

    def isochoric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isochoric heat capacity.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific isochoric heat capacity [J/(kg·K)]
        """
        V = self._find_volume(P, T)
        Cv = self._rtpress.eval_heat_capacity(V, T, const='V')
        return Cv * self._energy_to_SI

    def thermal_expansion(self, P: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Thermal expansion coefficient [K⁻¹]
        """
        V = self._find_volume(P, T)
        return self._rtpress.eval_thermal_exp(V, T)

    def adiabatic_gradient(self, P: float, T: float) -> float:
        """
        Calculate dimensionless adiabatic temperature gradient.

            (∂ln T/∂ln P)_S = (P/T)(∂T/∂P)_S

        where (∂T/∂P)_S = γT/K_S is evaluated by RTpress.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Adiabatic gradient (dimensionless)
        """
        V = self._find_volume(P, T)
        P_GPa = P / self._GPA_TO_PA
        dTdP_S = self._rtpress.eval_thermal_gradient(V, T)  # K/GPa
        return P_GPa * dTdP_S / T


# =============================================================================
# Phase Diagram Functions
# =============================================================================
#
# Phase boundaries and determination functions for MgSiO₃.
#
# Solid-solid transitions:
#   LP-CEn ↔ HP-CEn: Sokolova et al. (2022), Phys. Chem. Minerals 49:37
#   LP-CEn ↔ OrthoEn: Sokolova et al. (2022), Phys. Chem. Minerals 49:37
#   OrthoEn ↔ HP-CEn: Sokolova et al. (2022), Phys. Chem. Minerals 49:37
#   HP-CEn ↔ Bridgmanite: Empirical upper limit (P = 12 GPa)
#   Bridgmanite ↔ Post-perovskite: Ono & Oganov (2005), Earth Planet. Sci. Lett. 236:914
# Melting curve: Belonoshko et al. (2005) for P < ~2.55 GPa,
#   Fei et al. (2021) lower bound solution for P ≥ ~2.55 GPa
#
# Phases:
#   lpcen: low-pressure clinoenstatite (P2₁/c), low T
#   en: orthoenstatite (Pbca), high T and/or low P
#   hpcen: high-pressure clinoenstatite (C2/c), moderate-to-high P
#   brg: bridgmanite (Pbnm), high P
#   ppv: post-perovskite (Cmcm), very high P
#   liquid: above the melting curve
# =============================================================================


# Pyroxene triple point (Sokolova et al. 2022)
_P_TRIPLE_PYR = 6.5e9    # Pa - En–LP-CEn–HP-CEn triple point
_T_TRIPLE_PYR = 1100.0   # K - En–LP-CEn–HP-CEn triple point

# HP-CEn → Bridgmanite transition pressure (empirical upper limit for
# Sokolova22 validity; roughly consistent with absorption of majorite
# and akimotoite into the bridgmanite field)
_P_HPCEN_BRG = 12.0e9    # Pa


def P_lpcen_hpcen(T: float) -> float:
    """
    LP-CEn ↔ HP-CEn phase transition pressure.
    
    From Sokolova et al. (2022):
    P(T) = 6.94 - 0.0004 T
    
    Emanates from the pyroxene triple point toward lower T. At fixed
    P above the triple point, this gives the low-T boundary between
    LP-CEn (below, i.e. lower T) and HP-CEn (above, i.e. higher T).
    
    Parameters
    ----------
    T : float
        Temperature [K]
        
    Returns
    -------
    float
        Transition pressure [Pa]
    """
    P_GPa = 6.94 - 0.0004 * T
    return P_GPa * 1e9


def P_lpcen_en(T: float) -> float:
    """
    LP-CEn ↔ OrthoEn phase transition pressure.
    
    From Sokolova et al. (2022):
    P(T) = 15.6 - 0.0478 T + 3.59×10⁻⁵ T²
    
    This is a parabola with a minimum at T ≈ 665 K (P ≈ −0.3 GPa).
    Only the T > ~750 K branch (positive dP/dT) is physical, sweeping
    from P ≈ 0 at T ~ 750 K up to the triple point at (6.5 GPa, 1100 K).
    OrthoEn is stable below this curve and LP-CEn above it.
    
    Parameters
    ----------
    T : float
        Temperature [K]
        
    Returns
    -------
    float
        Transition pressure [Pa]
    """
    P_GPa = 15.6 - 0.0478 * T + 3.59e-5 * T**2
    return P_GPa * 1e9


def P_en_hpcen(T: float) -> float:
    """
    OrthoEn ↔ HP-CEn phase transition pressure.
    
    From Sokolova et al. (2022):
    P(T) = 4.2 + 0.0021 T
    
    Emanates from the pyroxene triple point toward higher T. At fixed
    P above the triple point, this gives the high-T boundary between
    HP-CEn (below, i.e. lower T) and OrthoEn (above, i.e. higher T).
    
    Parameters
    ----------
    T : float
        Temperature [K]
        
    Returns
    -------
    float
        Transition pressure [Pa]
    """
    P_GPa = 4.2 + 0.0021 * T
    return P_GPa * 1e9


def P_brg_ppv(T: float) -> float:
    """
    Bridgmanite ↔ post-perovskite phase transition pressure.
    
    From Ono & Oganov (2005):
    P(T) = 130 + 0.0070 (T - 2500)
    
    Parameters
    ----------
    T : float
        Temperature [K]
        
    Returns
    -------
    float
        Transition pressure [Pa]
    """
    P_GPa = 130.0 + 0.0070 * (T - 2500.0)
    return P_GPa * 1e9


# Crossover pressure between Belonoshko et al. (2005) and Fei et al. (2021)
# melting curves, determined numerically to machine precision.
_P0_MELT_MGSIO3 = 2.551686137257537e9  # Pa


def T_melt_MgSiO3(P: float) -> float:
    """
    Melting temperature of MgSiO₃.
    
    Two-branch parameterisation:
    
    For P < P₀ ≈ 2.55 GPa (Belonoshko et al. 2005):
        T_m(P) = 1831 (1 + P/4.6 GPa)^0.33
    
    For P ≥ P₀ (Fei et al. 2021, lower bound):
        T_m(P) = 6000 (P/140 GPa)^0.26
    
    The crossover pressure P₀ is the intersection of the two curves,
    ensuring continuity of the melting temperature.
    
    Parameters
    ----------
    P : float
        Pressure [Pa]
        
    Returns
    -------
    float
        Melting temperature [K]
    """
    P_GPa = P / 1e9
    if P_GPa <= 0.0:
        return 1831.0
    if P < _P0_MELT_MGSIO3:
        return 1831.0 * (1.0 + P_GPa / 4.6)**0.33
    return 6000.0 * (P_GPa / 140.0)**0.26


def get_mgsio3_phase(P: float, T: float) -> str:
    """
    Determine the stable phase of pure MgSiO₃ at given P and T.
    
    This function implements the MgSiO₃ phase diagram using:
    - Pyroxene boundaries from Sokolova et al. (2022)
    - Bridgmanite ↔ post-perovskite from Ono & Oganov (2005)
    - Empirical HP-CEn ↔ bridgmanite transition at 12 GPa
    - Melting curve from Belonoshko et al. (2005) / Fei et al. (2021)
    
    Parameters
    ----------
    P : float
        Pressure [Pa]
    T : float
        Temperature [K]
        
    Returns
    -------
    str
        Phase identifier: 'solid-lpcen', 'solid-en', 'solid-hpcen',
        'solid-brg', 'solid-ppv', or 'liquid'
    
    Examples
    --------
    >>> get_mgsio3_phase(5e9, 300)       # Low T, moderate P
    'solid-lpcen'
    >>> get_mgsio3_phase(5e9, 1500)      # Moderate P, high T
    'solid-en'
    >>> get_mgsio3_phase(8e9, 800)       # Moderate P, low T
    'solid-hpcen'
    >>> get_mgsio3_phase(50e9, 2000)     # High P
    'solid-brg'
    >>> get_mgsio3_phase(140e9, 3000)    # Very high P
    'solid-ppv'
    >>> get_mgsio3_phase(50e9, 5000)     # Above melting
    'liquid'
    """
    # Check if liquid
    T_melt = T_melt_MgSiO3(P)
    if T >= T_melt:
        return 'liquid'
    
    # Check post-perovskite
    if P >= P_brg_ppv(T):
        return 'solid-ppv'
    
    # Check bridgmanite (above 12 GPa, below ppv boundary)
    if P >= _P_HPCEN_BRG:
        return 'solid-brg'
    
    # Pyroxene regime (P < 12 GPa)
    # The three boundaries emanate from the triple point (6.5 GPa, 1100 K).
    # We check which side of each relevant boundary the point falls on.
    
    if P >= P_en_hpcen(T):
        # Above the en-hpcen line: either hpcen or lpcen
        if P >= P_lpcen_hpcen(T):
            return 'solid-hpcen'
        else:
            return 'solid-lpcen'
    else:
        # Below the en-hpcen line: either en or lpcen
        # The parabola P_lpcen_en separates them (valid for T > ~750 K).
        # Below ~750 K the parabola is at P < 0, so the entire field
        # is lpcen (the low-T polymorph).
        if T > 750.0 and P < P_lpcen_en(T):
            return 'solid-en'
        else:
            return 'solid-lpcen'


def get_mgsio3_eos(phase: str):
    """
    Return the appropriate EoS instance for a given MgSiO₃ phase.
    
    This function returns an EoS instance configured for the specified
    MgSiO₃ phase:
    
    - En: Sokolova et al. (2022), phase='orthoen'
    - LP-CEn: Sokolova et al. (2022), phase='lp-cen'
    - HP-CEn: Sokolova et al. (2022), phase='hp-cen'
    - Bridgmanite: Wolf et al. (2015), x_Fe=0.0
    - Post-perovskite: Sakai et al. (2016)
    - Liquid: Wolf & Bower (2018)
    
    Parameters
    ----------
    phase : str
        Phase identifier: 'solid-lpcen', 'solid-en', 'solid-hpcen',
        'solid-brg', 'solid-ppv', or 'liquid'
        
    Returns
    -------
    object
        EoS instance with standard interface methods
        
    Raises
    ------
    ValueError
        If phase is not recognized
        
    Examples
    --------
    >>> eos = get_mgsio3_eos('solid-brg')
    >>> rho = eos.density(50e9, 2000)
    
    Notes
    -----
    For bridgmanite, returns pure MgSiO₃ (x_Fe = 0). For iron-bearing
    compositions, instantiate Wolf15 directly with the desired x_Fe.
    """
    phase_lower = phase.lower()
    
    if phase_lower == 'solid-en':
        return Sokolova22(phase='orthoen')
    elif phase_lower == 'solid-lpcen':
        return Sokolova22(phase='lp-cen')
    elif phase_lower == 'solid-hpcen':
        return Sokolova22(phase='hp-cen')
    elif phase_lower == 'solid-brg':
        return Wolf15(x_Fe=0.0)
    elif phase_lower == 'solid-ppv':
        return Sakai16()
    elif phase_lower == 'liquid':
        return Wolf18()
    else:
        raise ValueError(
            f"Unknown phase '{phase}'. "
            f"Valid options: 'solid-en', 'solid-lpcen', 'solid-hpcen', "
            f"'solid-brg', 'solid-ppv', 'liquid'"
        )


def get_mgsio3_eos_for_PT(P: float, T: float):
    """
    Return the appropriate EoS instance for given P-T conditions.
    
    Combines phase determination and EoS selection.
    
    Parameters
    ----------
    P : float
        Pressure [Pa]
    T : float
        Temperature [K]
        
    Returns
    -------
    tuple
        (eos_instance, phase_name)
        
    Examples
    --------
    >>> eos, phase = get_mgsio3_eos_for_PT(50e9, 2000)
    >>> rho = eos.density(50e9, 2000)
    
    >>> # Deep mantle conditions
    >>> eos, phase = get_mgsio3_eos_for_PT(120e9, 2500)
    >>> print(f"Using {type(eos).__name__} for {phase}")
    """
    phase = get_mgsio3_phase(P, T)
    eos = get_mgsio3_eos(phase)
    return eos, phase


# =============================================================================
# Wrapper Class
# =============================================================================


class MgSiO3EoS:
    """
    Wrapper equation of state for MgSiO3 with pre-instantiated phase classes.

    This class instantiates every individual MgSiO3 phase EoS class once at
    initialization and selects the appropriate one at each (P, T) query
    based on the MgSiO3 phase diagram. It avoids the overhead of repeated
    class construction (especially for Wolf18 which requires RTpress
    initialization with sympy compilation) when the EoS is queried many
    times.

    The seven standard PALEOS thermodynamic properties are exposed as
    public methods, together with a ``phase`` method that returns the
    stable phase label at the queried point.

    Attributes
    ----------
    _eos_lpcen : Sokolova22
        EoS instance for low-pressure clinoenstatite
    _eos_en : Sokolova22
        EoS instance for orthoenstatite
    _eos_hpcen : Sokolova22
        EoS instance for high-pressure clinoenstatite
    _eos_brg : Wolf15
        EoS instance for bridgmanite (pure MgSiO3)
    _eos_ppv : Sakai16
        EoS instance for post-perovskite
    _eos_liquid : Wolf18
        EoS instance for liquid MgSiO3

    Examples
    --------
    >>> eos = MgSiO3EoS()
    >>> rho = eos.density(50e9, 2000)
    >>> phase = eos.phase(50e9, 2000)
    >>> print(f"{phase}: rho = {rho:.1f} kg/m^3")
    """

    def __init__(self):
        """
        Initialize MgSiO3EoS by pre-instantiating all phase EoS classes.
        """
        self._eos_lpcen = Sokolova22(phase='lp-cen')
        self._eos_en = Sokolova22(phase='orthoen')
        self._eos_hpcen = Sokolova22(phase='hp-cen')
        self._eos_brg = Wolf15(x_Fe=0.0)
        self._eos_ppv = Sakai16()
        self._eos_liquid = Wolf18()

        self._phase_eos_map = {
            'solid-lpcen': self._eos_lpcen,
            'solid-en':    self._eos_en,
            'solid-hpcen': self._eos_hpcen,
            'solid-brg':   self._eos_brg,
            'solid-ppv':   self._eos_ppv,
            'liquid':      self._eos_liquid,
        }

    def _get_eos(self, P, T):
        """Return (eos_instance, phase_label) for given P-T conditions."""
        phase = get_mgsio3_phase(P, T)
        return self._phase_eos_map[phase], phase

    def phase(self, P, T):
        """
        Return the stable MgSiO3 phase at given P and T.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        str
            Phase identifier: 'solid-lpcen', 'solid-en', 'solid-hpcen',
            'solid-brg', 'solid-ppv', or 'liquid'
        """
        return get_mgsio3_phase(P, T)

    def density(self, P, T):
        """Calculate density [kg/m³]."""
        eos, _ = self._get_eos(P, T)
        return eos.density(P, T)

    def specific_internal_energy(self, P, T):
        """Calculate specific internal energy [J/kg]."""
        eos, _ = self._get_eos(P, T)
        return eos.specific_internal_energy(P, T)

    def specific_entropy(self, P, T):
        """Calculate specific entropy [J/(kg·K)]."""
        eos, _ = self._get_eos(P, T)
        return eos.specific_entropy(P, T)

    def isobaric_heat_capacity(self, P, T):
        """Calculate specific isobaric heat capacity [J/(kg·K)]."""
        eos, _ = self._get_eos(P, T)
        return eos.isobaric_heat_capacity(P, T)

    def isochoric_heat_capacity(self, P, T):
        """Calculate specific isochoric heat capacity [J/(kg·K)]."""
        eos, _ = self._get_eos(P, T)
        return eos.isochoric_heat_capacity(P, T)

    def thermal_expansion(self, P, T):
        """Calculate volumetric thermal expansion coefficient [K⁻¹]."""
        eos, _ = self._get_eos(P, T)
        return eos.thermal_expansion(P, T)

    def adiabatic_gradient(self, P, T):
        """Calculate dimensionless adiabatic temperature gradient."""
        eos, _ = self._get_eos(P, T)
        return eos.adiabatic_gradient(P, T)