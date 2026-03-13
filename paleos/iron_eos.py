"""
PALEOS Equations of State for Fe

This module contains implementations of various equations of state (EoS) for 
iron relevant to planetary interiors and high-pressure physics.

Each EoS is implemented as a separate class with consistent method signatures
to ensure interoperability across different implementations.

EoS Classes
-----------
- Dorogokupets17: bcc-Fe (α, δ) and fcc-Fe (γ) from Dorogokupets et al. (2017)
- Miozzi20: hcp-Fe (ε) from Miozzi et al. (2020)
- Hakim18: hcp-Fe (ε) at high pressure from Hakim et al. (2018)
- HcpIronEos: Composite hcp-Fe blending Miozzi20 and Hakim18 with smooth transition
- Luo24: liquid Fe from Luo et al. (2024)
- IronEoS: Wrapper class with pre-instantiated phases for efficient repeated
           evaluation with automatic phase selection

Phase Determination
-------------------
The module provides functions to determine the stable iron phase at given P-T
conditions, following the phase diagram in BICEPS (Haldemann et al. 2024):

- get_iron_phase(P, T): Returns the stable phase ('solid-alpha', 'solid-delta',
                        'solid-gamma', 'solid-epsilon', or 'liquid')
- get_iron_eos(phase): Returns EoS instance for a given phase name
- get_iron_eos_for_PT(P, T): Returns (EoS instance, phase) for given conditions

Phase boundary functions:
- T_gamma_epsilon(P): γ ↔ ε boundary (Dorogokupets et al. 2017)
- T_alpha_gamma(P): α ↔ γ boundary (Dorogokupets et al. 2017)
- T_delta_gamma(P): δ ↔ γ boundary (Dorogokupets et al. 2017)
- T_alpha_epsilon(P): α ↔ ε boundary (Dorogokupets et al. 2017)
- T_melt_Fe(P): Melting curve (Anzellini et al. 2013)

Author: Mara Attia
Date: November 2025
"""

import numpy as np
from scipy.optimize import brentq

# Physical constants
R_GAS = 8.314462618  # J/(mol·K) - Universal gas constant
ATOMIC_MASS_FE = 55.845e-3  # kg/mol - Atomic mass of iron
N_AVOGADRO = 6.02214076e23  # mol^-1 - Avogadro number


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


class Dorogokupets17:
    """
    Equation of state for iron phases from Dorogokupets et al. (2017).
    
    Reference:
    Dorogokupets, P.I., Dymshits, A.M., Litasov, K.D., Sokolova, T.S. (2017)
    "Thermodynamics and Equations of State of Iron to 350 GPa and 6000 K"
    Scientific Reports, 7:41863, DOI: 10.1038/srep41863
    
    This implementation covers:
    - bcc-Fe: body-centered cubic iron, stable at low P and T
    - fcc-Fe: face-centered cubic iron, stable at intermediate conditions
    
    The thermodynamic model is based on the Helmholtz free energy decomposition:
    F = U_0 + E_0(V) + F_th(V,T) + F_e(V,T) + F_mag(T)
    
    where:
    - E_0(V): cold compression energy (Vinet equation)
    - F_th(V,T): thermal contribution (Einstein model)
    - F_e(V,T): electronic contribution
    - F_mag(T): magnetic contribution (bcc-Fe only)
    
    All methods take pressure P [Pa] and temperature T [K] as inputs and
    return quantities in SI units unless otherwise specified.
    
    Parameters
    ----------
    phase : str
        Phase of iron: 'bcc' for α-phase/δ-phase or 'fcc' for γ-phase
    
    Attributes
    ----------
    phase : str
        Selected phase ('bcc' or 'fcc')
    T0 : float
        Reference temperature (298.15 K)
    params : dict
        Active parameters for the selected phase
    
    Examples
    --------
    >>> # Create instance for bcc-Fe
    >>> eos_bcc = Dorogokupets17(phase='bcc')
    >>> rho = eos_bcc.density(P=1e5, T=298.15)
    >>> 
    >>> # Create instance for fcc-Fe
    >>> eos_fcc = Dorogokupets17(phase='fcc')
    >>> rho = eos_fcc.density(P=1e9, T=1500)
    """
    
    def __init__(self, phase: str = 'bcc'):
        """
        Initialize the Dorogokupets17 EoS for a specific iron phase.
        
        Parameters
        ----------
        phase : str, optional
            Phase of iron: 'bcc' for α-phase/δ-phase or 'fcc' for γ-phase
            Default is 'bcc'
            
        Raises
        ------
        ValueError
            If phase is not 'bcc' or 'fcc'
        """
        
        if phase not in ['bcc', 'fcc']:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'bcc' or 'fcc'.")
        
        self.phase = phase
        
        # Reference conditions
        self.T0 = 298.15  # K
        
        # Parameters for bcc-Fe (α-phase/δ-phase) from Table 1
        self.bcc_params = {
            #'U0': 0.0,              # J/mol
            'U0': 116.636e3,        # J/mol
            'S0': -94.321,          # J/(mol·K)
            'V0': 7.092e-6,         # m³/mol
            'K0': 164.0e9,          # Pa
            'K0_prime': 5.50,       # dimensionless
            'Theta0': 303.0,        # K
            'gamma0': 1.736,        # dimensionless
            'beta': 1.125,          # dimensionless
            'gamma_inf': 0.0,       # dimensionless
            'e0': 198e-6,           # K⁻¹
            'g': 1.0,               # dimensionless
            'Tc': 1043.0,           # K
            'B0': 2.22,             # dimensionless
            'p': 0.4,               # dimensionless
            'n': 1.0                # dimensionless
        }
        
        # Parameters for fcc-Fe (γ-phase) from Table 1
        self.fcc_params = {
            #'U0': 4470.,            # J/mol
            'U0': 121.080e3,        # J/mol
            'S0': -94.091,          # J/(mol·K)
            'V0': 6.9285e-6,        # m³/mol
            'K0': 146.2e9,          # Pa
            'K0_prime': 4.67,       # dimensionless
            'Theta0': 222.5,        # K
            'gamma0': 2.203,        # dimensionless
            'beta': 0.01,           # dimensionless
            'gamma_inf': 0.0,       # dimensionless
            'e0': 198e-6,           # K⁻¹
            'g': 0.5,               # dimensionless
            'p': 0.28,              # dimensionless
            'n': 1.0                # dimensionless
        }
        
        # Set active parameters based on selected phase
        if phase == 'bcc':
            self.params = self.bcc_params
        else:  # phase == 'fcc'
            self.params = self.fcc_params
    
    # =============================================================================
    # Helper functions for thermodynamic components
    # =============================================================================
    
    def _gruneisen_parameter(self, V: float) -> float:
        """
        Calculate Grüneisen parameter as a function of volume.
        
        Equation (13): γ = γ_∞ + (γ_0 - γ_∞) x^β
        where x = V/V_0
        
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
        gamma_inf = self.params['gamma_inf']
        gamma0 = self.params['gamma0']
        beta = self.params['beta']
        
        return gamma_inf + (gamma0 - gamma_inf) * x**beta
    
    def _q_parameter(self, V: float) -> float:
        """
        Calculate q parameter (volume derivative of Grüneisen parameter).
        
        Equation (14): q = β x^β (γ_0 - γ_∞) / γ
        where x = V/V_0
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            q parameter (dimensionless)
        """
        x = V / self.params['V0']
        gamma = self._gruneisen_parameter(V)
        gamma_inf = self.params['gamma_inf']
        gamma0 = self.params['gamma0']
        beta = self.params['beta']
        
        if abs(gamma) < 1e-10:
            return 0.0
        
        return beta * x**beta * (gamma0 - gamma_inf) / gamma
    
    def _einstein_temperature(self, V: float) -> float:
        """
        Calculate Einstein temperature as a function of volume.
        
        Equation (15): Θ = Θ_0 x^(-γ_∞) exp[(γ_0 - γ_∞)/β (1 - x^β)]
        where x = V/V_0
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Einstein temperature [K]
        """
        x = V / self.params['V0']
        Theta0 = self.params['Theta0']
        gamma0 = self.params['gamma0']
        gamma_inf = self.params['gamma_inf']
        beta = self.params['beta']
        
        exponent = (gamma0 - gamma_inf) / beta * (1 - x**beta) if beta != 0 else 0
        return Theta0 * x**(-gamma_inf) * np.exp(exponent)
    
    def _cold_pressure(self, V: float) -> float:
        """
        Calculate cold compression pressure using Vinet-Rydberg equation.
        
        Equation (2): P_0(V) = 3 K_0 X^(-2) (1 - X) exp[η(1 - X)]
        where X = (V/V_0)^(1/3) and η = 3K_0'/2 - 3/2
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Cold compression pressure [Pa]
        """
        V0 = self.params['V0']
        K0 = self.params['K0']
        K0_prime = self.params['K0_prime']
        
        X = (V / V0)**(1/3)
        eta = 3 * K0_prime / 2 - 3/2
        
        return 3 * K0 * X**(-2) * (1 - X) * np.exp(eta * (1 - X))
    
    def _thermal_pressure(self, V: float, T: float) -> float:
        """
        Calculate thermal pressure contribution.
        
        Equation (10): P_th = (3nRγ/V) [Θ/(exp(Θ/T) - 1)]
        
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
        if T < 1e-6:
            return 0.0
        
        n = self.params['n']
        gamma = self._gruneisen_parameter(V)
        Theta = self._einstein_temperature(V)
        
        # Avoid overflow for large Theta/T
        if Theta / T > 100:
            return 0.0
        
        return (3 * n * R_GAS * gamma / V) * (Theta / (np.exp(Theta / T) - 1))
    
    def _electronic_pressure(self, V: float, T: float) -> float:
        """
        Calculate electronic pressure contribution.
        
        Equation (17): P_e = (3 n R e_0 g x^g T^2)/(2 V)
        where x = V/V_0, as e = e_0 x^g
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Electronic pressure [Pa]
        """
        x = V / self.params['V0']
        n = self.params['n']
        e0 = self.params['e0']
        g = self.params['g']
        
        return (3 * n * R_GAS * e0 * g * x**g * T**2) / (2 * V)
    
    def _magnetic_helmholtz(self, T: float) -> float:
        """
        Calculate magnetic contribution to Helmholtz free energy.
        
        Equations (19-21): F_mag(T) = RT ln(B_0 + 1) (f(τ) - 1)
        where τ = T/Tc and f(τ) is defined piecewise for τ ≤ 1 and τ > 1
        
        Only applicable to bcc-Fe with ferromagnetic properties.
        
        Parameters
        ----------
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Magnetic Helmholtz free energy [J/mol]
        """
        # Only bcc-Fe has magnetic contribution
        if 'Tc' not in self.params or 'B0' not in self.params:
            return 0.0
        
        Tc = self.params['Tc']
        B0 = self.params['B0']
        p = self.params['p']
        
        tau = T / Tc
        D = 518/1125 + 11692/15975 * (1/p - 1)
        
        if tau <= 1:
            # Equation (20)
            f_tau = 1 - (79*tau**(-1) / (140*p) + 
                         474/497 * (1/p - 1) * (tau**3/6 + tau**9/135 + tau**15/600)) / D
        else:
            # Equation (21)
            f_tau = -(tau**(-5)/10 + tau**(-15)/315 + tau**(-25)/1500) / D
        
        return R_GAS * T * np.log(B0 + 1) * (f_tau - 1)
    
    def _total_pressure(self, V: float, T: float) -> float:
        """
        Calculate total pressure at given volume and temperature.
        
        P(V,T) = P_0(V) + P_th(V,T) + P_e(V,T)
        
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
        P_cold = self._cold_pressure(V)
        P_thermal = self._thermal_pressure(V, T)
        P_electronic = self._electronic_pressure(V, T)
        
        return P_cold + P_thermal + P_electronic
    
    def _find_volume(self, P: float, T: float, V_guess: float = None) -> float:
        """
        Find molar volume for given pressure and temperature by root finding.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
        V_guess : float, optional
            Initial guess for volume [m³/mol]
            
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
        
        # Set reasonable bounds for volume search
        # Allow compression to 0.3*V0 and expansion to 1.5*V0
        V_min = 0.3 * V0
        V_max = 1.5 * V0
        
        # Define function whose root we seek
        def pressure_residual(V):
            return self._total_pressure(V, T) - P
        
        try:
            V = brentq(pressure_residual, V_min, V_max, xtol=1e-12)
            return V
        except ValueError:
            raise RuntimeError(
                f"Could not find volume for P = {P/1e9:.1f} GPa, T = {T:.1f} K. "
                f"Pressure may be outside valid range for this EoS."
            )
    
    # =============================================================================
    # Thermodynamic property calculations
    # =============================================================================
    
    def _isochoric_heat_capacity(self, V: float, T: float) -> float:
        """
        Calculate isochoric heat capacity.
        
        Equations (9, 17): C_V = 3nR (Θ/T)^2 exp(Θ/T)/[exp(Θ/T) - 1]^2 + 3nR e_0 x^g T
        where x = V/V_0, as e = e_0 x^g
        Magnetic contribution is derived analytically and added too
        
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
        
        x = V/self.params['V0']
        n = self.params['n']
        e0 = self.params['e0']
        g = self.params['g']
        Theta = self._einstein_temperature(V)
        
        # Thermal contribution
        if Theta / T > 100:
            Cv_th = 0.0
        else:
            exp_term = np.exp(Theta / T)
            Cv_th = 3 * n * R_GAS * (Theta / T)**2 * exp_term / (exp_term - 1)**2
        
        # Electronic contribution
        Cv_e = 3 * n * R_GAS * e0 * x**g * T

        # Magnetic contribution
        if 'Tc' not in self.params or 'B0' not in self.params:
            Cv_mag = 0.0

        else:
            Tc = self.params['Tc']
            B0 = self.params['B0']
            p = self.params['p']
            
            tau = T / Tc
            D = 518/1125 + 11692/15975 * (1/p - 1)
            if tau <= 1:
                # First derivative f'(tau) of f(tau), Equation (20)
                df_tau = (79*tau**(-2) / (140*p) - 
                          474/497 * (1/p - 1) * (tau**2/2 + tau**8/15 + tau**14/40)) / D
                
                # Second derivative f''(tau) of f(tau), Equation (20)
                d2f_tau = (-158*tau**(-3) / (140*p) - 
                           474/497 * (1/p - 1) * (tau + 8*tau**7/15 + 7*tau**13/20)) / D
            else:
                # First derivative f'(tau) of f(tau), Equation (21)
                df_tau = (tau**(-6)/2 + tau**(-16)/21 + tau**(-26)/60) / D
                
                # Second derivative f''(tau) of f(tau), Equation (21)
                d2f_tau = -(3*tau**(-7) + 16*tau**(-17)/21 + 13*tau**(-27)/30) / D
            
            Cv_mag = -R_GAS * np.log(B0 + 1) * (2*tau*df_tau + tau**2*d2f_tau)
        
        return Cv_th + Cv_e + Cv_mag
    
    def _thermal_expansion_coeff(self, V: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient.
        
        α = (1/V)(∂V/∂T)_P = (∂P/∂T)_V / K_T
        
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
        # Calculate (∂P/∂T)_V using equation (12)
        gamma = self._gruneisen_parameter(V)
        Cv = self._isochoric_heat_capacity(V, T)
        dP_dT_V = gamma * Cv / V
        
        # Calculate isothermal bulk modulus
        Kt = self._isothermal_bulk_modulus(V, T)
        
        return dP_dT_V / Kt
    
    def _isothermal_bulk_modulus(self, V: float, T: float) -> float:
        """
        Calculate isothermal bulk modulus.
        
        Equation (11): K_T = P_th(1 + γ - q) - γ^2 T C_V / V
        plus cold compression and electronic contributions
        
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
        # Cold compression bulk modulus (equation 3)
        V0 = self.params['V0']
        K0 = self.params['K0']
        K0_prime = self.params['K0_prime']
        
        X = (V / V0)**(1/3)
        eta = 3 * K0_prime / 2 - 3/2
        
        K_cold = K0 * X**(-2) * np.exp(eta*(1 - X)) * (1 + (1 - X)*(eta*X + 1))

        # Electronic contribution (equation 17)
        g = self.params['g']

        P_e = self._electronic_pressure(V, T)

        K_e = P_e * (1 - g)
        
        # Thermal contribution
        gamma = self._gruneisen_parameter(V)
        q = self._q_parameter(V)
        P_th = self._thermal_pressure(V, T)
        Cv = self._isochoric_heat_capacity(V, T)
        
        K_th = P_th * (1 + gamma - q) - gamma**2 * T * Cv / V
        
        return K_cold + K_e + K_th
    
    def _helmholtz_free_energy(self, V: float, T: float) -> float:
        """
        Calculate Helmholtz free energy.
        
        Equation (1): F = U_0 + E_0(V) + F_th(V,T) - F_th(V,T_0) + 
                          F_e(V,T) - F_e(V,T_0) + F_mag(T) - F_mag(T_0)
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Helmholtz free energy [J/mol]
        """
        # Reference internal energy
        U0 = self.params['U0']
        
        # Cold compression energy (equation 5)
        V0 = self.params['V0']
        K0 = self.params['K0']
        K0_prime = self.params['K0_prime']
        
        X = (V / V0)**(1/3)
        eta = 3 * K0_prime / 2 - 3/2
        
        E0 = 9 * K0 * V0 * (1 - (1 - eta*(1 - X))*np.exp(eta*(1 - X))) / eta**2
        
        # Thermal contribution (equation 6)
        n = self.params['n']
        Theta = self._einstein_temperature(V)
        
        F_th_T = 3 * n * R_GAS * T * np.log(1 - np.exp(-Theta / T))
        F_th_T0 = 3 * n * R_GAS * self.T0 * np.log(1 - np.exp(-Theta / self.T0))
        
        # Electronic contribution (equation 16)
        e0 = self.params['e0']
        g = self.params['g']
        x = V / V0
        
        F_e_T = -1.5 * n * R_GAS * e0 * x**g * T**2
        F_e_T0 = -1.5 * n * R_GAS * e0 * x**g * self.T0**2
        
        # Magnetic contribution (bcc-Fe only)
        F_mag_T = self._magnetic_helmholtz(T)
        F_mag_T0 = self._magnetic_helmholtz(self.T0)
        
        return (U0 + E0 + (F_th_T - F_th_T0) + (F_e_T - F_e_T0) + 
                (F_mag_T - F_mag_T0))
    
    def _entropy(self, V: float, T: float) -> float:
        """
        Calculate specific entropy.
        
        S = -(∂F/∂T)_V
        
        Equations (7, 17) provide the explicit form from thermal and electronic terms.
        Magnetic contribution is derived analytically.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Entropy [J/(kg·K)]
        """
        S0 = self.params['S0']

        if T < 1e-6:
            return 0.0
        
        n = self.params['n']
        Theta = self._einstein_temperature(V)
        
        # Thermal contribution (equation 7)
        if Theta / T > 100:
            S_th = 0.0
        else:
            exp_term = np.exp(Theta / T)
            S_th = 3 * n * R_GAS * (
                -np.log(1 - 1/exp_term) + 
                (Theta / T) / (exp_term - 1)
            )
        
        # Electronic contribution (equation 17)
        x = V/self.params['V0']
        e0 = self.params['e0']
        g = self.params['g']
        S_e = 3 * n * R_GAS * e0 * x**g * T

        # Magnetic contribution
        if 'Tc' not in self.params or 'B0' not in self.params:
            S_mag = 0.0

        else:
            Tc = self.params['Tc']
            B0 = self.params['B0']
            p = self.params['p']
            
            tau = T / Tc
            D = 518/1125 + 11692/15975 * (1/p - 1)
            if tau <= 1:
                # Equation (20) - f(tau)
                f_tau = 1 - (79*tau**(-1) / (140*p) + 
                             474/497 * (1/p - 1) * (tau**3/6 + tau**9/135 + tau**15/600)) / D
                
                # First derivative f'(tau)
                df_tau = (79*tau**(-2) / (140*p) - 
                          474/497 * (1/p - 1) * (tau**2/2 + tau**8/15 + tau**14/40)) / D
            else:
                # Equation (21) - f(tau)
                f_tau = -(tau**(-5)/10 + tau**(-15)/315 + tau**(-25)/1500) / D
                
                # First derivative f'(tau)
                df_tau = (tau**(-6)/2 + tau**(-16)/21 + tau**(-26)/60) / D
            
            S_mag = -R_GAS * np.log(B0 + 1) * (f_tau - 1 + tau*df_tau)
        
        # Convert from J/(mol·K) to J/(kg·K)
        S_molar = S_th + S_e + S_mag + S0
        return S_molar / ATOMIC_MASS_FE
    
    def _internal_energy(self, V: float, T: float) -> float:
        """
        Calculate specific internal energy.
        
        E = F + TS (equation 8)
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Internal energy [J/kg]
        """
        F = self._helmholtz_free_energy(V, T)
        S_molar = self._entropy(V, T) * ATOMIC_MASS_FE  # Convert to J/(mol·K)
        
        E_molar = F + T * S_molar
        return E_molar / ATOMIC_MASS_FE
    
    # =============================================================================
    # Public interface
    # =============================================================================
    
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
        return ATOMIC_MASS_FE / V
    
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
        return self._internal_energy(V, T)
    
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
        return self._entropy(V, T)
    
    def isobaric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isobaric heat capacity.
        
        C_P = C_V + α^2 T V K_T
        
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
        Kt = self._isothermal_bulk_modulus(V, T)
        
        Cp_molar = Cv + alpha**2 * T * V * Kt
        return Cp_molar / ATOMIC_MASS_FE
    
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
        return Cv_molar / ATOMIC_MASS_FE
    
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
        
        (∂ln T/∂ln P)_S = (α P) / (C_P ρ)
        
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


class Miozzi20:
    """
    Equation of state for hcp-Fe from Miozzi et al. (2020).
    
    Reference:
    Miozzi, F., Matas, J., Guignot, N., Badro, J., Siebert, J., Fiquet, G. (2020)
    "A New Reference for the Thermal Equation of State of Iron"
    Minerals, 10(2):100, DOI: 10.3390/min10020100
    
    This implementation uses:
    - 3rd order Birch-Murnaghan for cold compression
    - Mie-Grüneisen-Debye formalism for thermal pressure
    
    The equation of state is valid for hexagonal close-packed (hcp) ε-iron,
    which is the stable phase at high pressure (> 60 GPa) up to core conditions.
    
    Thermodynamic model:
    P(V,T) = P_cold(V) + P_th(V,T)
    
    where P_cold is from third-order Birch-Murnaghan and P_th from Debye thermal model.
    
    All methods take pressure P [Pa] and temperature T [K] as inputs and
    return quantities in SI units unless otherwise specified.
    
    Attributes
    ----------
    T0 : float
        Reference temperature (300 K)
    params : dict
        EoS parameters from the paper
    
    Examples
    --------
    >>> # Create instance for hcp-Fe
    >>> eos = Miozzi20()
    >>> rho = eos.density(P=1e11, T=3000)
    >>> s = eos.specific_entropy(P=1e11, T=3000)
    
    Notes
    -----
    The paper contains several typographical errors that have been corrected
    in this implementation:
    - Eq. (5): γ(V)/γ should be γ(V)/V
    - Eq. (7): (T/θ_D) should be (T/θ_D)³ and the integral up to θ_D/T
    - Sect. 3.3: they fit for θ_D,0, not θ_D 
    """
    
    def __init__(self):
        """
        Initialize the Miozzi20 EoS for hcp-Fe.
        
        Uses the preferred parameters from Table 1 and Section 3.3 of the paper.
        """
        
        # Reference conditions
        self.T0 = 300.0  # K (reference temperature used in the paper)
        
        # EoS parameters from the paper (preferred solution)
        # These are the parameters obtained with the full dataset including
        # low-pressure measurements
        self.params = {
            #'U0': 0.0,              # J/mol
            'U0': -41.649e3,        # J/mol
            'S0': -72.427,          # J/(mol·K)
            'V0': 6.87e-6,          # m³/mol
            'K0': 129.0e9,          # Pa
            'K0_prime': 6.24,       # dimensionless
            'theta_D0': 420.0,      # K
            'gamma0': 1.11,         # dimensionless
            'q': 0.3,               # dimensionless
            'n': 1.0                # dimensionless
        }
    
    # =============================================================================
    # Helper functions for EoS components
    # =============================================================================
    
    def _gruneisen_parameter(self, V: float) -> float:
        """
        Calculate Grüneisen parameter as a function of volume.
        
        Equation (6): γ(V) = γ_0 (V/V_0)^q
        
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
        return self.params['gamma0'] * x**self.params['q']
    
    def _debye_temperature(self, V: float) -> float:
        """
        Calculate Debye temperature as a function of volume.
        
        Equation (8): θ_D = θ_D,0 exp[(γ_0 - γ(V))/q]
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Debye temperature [K]
        """
        gamma_V = self._gruneisen_parameter(V)
        gamma0 = self.params['gamma0']
        q = self.params['q']
        theta_D0 = self.params['theta_D0']
        
        if abs(q) < 1e-10:
            return theta_D0
        
        exponent = (gamma0 - gamma_V) / q
        return theta_D0 * np.exp(exponent)
    
    def _debye_integral(self, x_max: float) -> float:
        """
        Calculate the Debye integral: ∫_0^x_max x^3/(exp(x) - 1) dx
        
        Uses numerical integration with adaptive handling for different regimes.
        
        Parameters
        ----------
        x_max : float
            Upper limit of integration (θ_D/T)
            
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
    
    def _thermal_energy(self, V: float, T: float) -> float:
        """
        Calculate thermal energy contribution using Debye model.
        
        Equation (7): E_th(V,T) = 9nR[θ_D/8 + T(T/θ_D)^3 ∫_0^(θ_D/T) x^3/(exp(x)-1) dx]
        
        Note: The paper has two typos, should be (T/θ_D)^3 not (T/θ_D), 
        and upper integration bound is (θ_D/T) not (θ_D,0/T)
        
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
        if T < 1e-6:
            # At T=0, only zero-point energy remains
            theta_D = self._debye_temperature(V)
            return 9 * R_GAS * theta_D / 8
        
        theta_D = self._debye_temperature(V)
        n = self.params['n']
        
        # Zero-point energy
        E_zero = theta_D / 8
        
        # Thermal contribution
        x_max = theta_D / T
        integral = self._debye_integral(x_max)
        E_thermal = T * (T / theta_D)**3 * integral
        
        return 9 * n * R_GAS * (E_zero + E_thermal)
    
    def _cold_pressure(self, V: float) -> float:
        """
        Calculate cold compression pressure using 3rd order Birch-Murnaghan.
        
        Equation (3): P = (3/2)K_0[x^(7/3) - x^(5/3)][1 + (3/4)(K'_0 - 4)[x^(2/3) - 1]]
        where x = V_0/V
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Cold compression pressure [Pa]
        """
        V0 = self.params['V0']
        K0 = self.params['K0']
        K0_prime = self.params['K0_prime']
        
        x = V0 / V
        f = x**(2/3) - 1
        
        # Eulerian strain
        P = (3/2) * K0 * (x**(7/3) - x**(5/3)) * (1 + (3/4) * (K0_prime - 4) * f)
        
        return P
    
    def _thermal_pressure(self, V: float, T: float) -> float:
        """
        Calculate thermal pressure contribution.
        
        Equation (5): ΔP_th = (γ(V)/V)[E_th(V,T) - E_th(V,T_0)]
        
        Note: The paper has a typo, should be γ(V)/V not γ(V)/γ
        
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
        
        gamma_V = self._gruneisen_parameter(V)
        E_T = self._thermal_energy(V, T)
        E_T0 = self._thermal_energy(V, self.T0)
        
        return (gamma_V / V) * (E_T - E_T0)
    
    def _total_pressure(self, V: float, T: float) -> float:
        """
        Calculate total pressure at given volume and temperature.
        
        Equation (4): P(V,T) = P(V,T_0) + ΔP_th(V,T)
        
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
        P_cold = self._cold_pressure(V)
        P_th = self._thermal_pressure(V, T)
        
        return P_cold + P_th
    
    def _find_volume(self, P: float, T: float) -> float:
        """
        Find molar volume for given pressure and temperature.
        
        Uses root finding to solve P(V,T) = P_target
        
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
        
        # Define function whose root we seek
        def pressure_diff(V):
            return self._total_pressure(V, T) - P
        
        # Set search bounds
        # Lower bound: compressed to 10% of V0
        V_min = 0.1 * V0
        # Upper bound: expanded to 200% of V0
        V_max = 2.0 * V0
        
        try:
            V_solution = brentq(pressure_diff, V_min, V_max, xtol=1e-12)
            return V_solution
        except ValueError:
            raise RuntimeError(
                f"Failed to find volume at P = {P/1e9:.2f} GPa, T = {T:.2f} K. "
                f"Pressure may be outside valid range."
            )
    
    def _entropy(self, V: float, T: float) -> float:
        """
        Calculate specific entropy using thermodynamic relations.
        
        S = -(∂F_th/∂T)_V = ∫(C_V/T)dT
        
        For Debye model (e.g., Gopal 1966, equation 2.16b):
        S = nR[4D_3(θ_D/T) - 3ln(1 - exp(-θ_D/T))]
        
        where D_3 is the third Debye function.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Entropy [J/(kg·K)]
        """
        S0 = self.params['S0']

        if T < 1e-6:
            return 0.0
        
        theta_D = self._debye_temperature(V)
        n = self.params['n']
        
        # Debye function integral
        x = theta_D / T
        
        if x > 100:
            # Low temperature limit
            S_molar = 0.0
        else:            
            # Debye entropy
            integral = self._debye_integral(x)
            D3 = (3 / x**3) * integral
            
            S_molar = n * R_GAS * (4 * D3 - 3 * np.log(1 - np.exp(-x)))
        
        # Convert from J/(mol·K) to J/(kg·K)
        S_molar += S0
        return S_molar / ATOMIC_MASS_FE
    
    def _internal_energy(self, V: float, T: float) -> float:
        """
        Calculate specific internal energy.
        
        E = E_cold + E_th
        
        where E_cold is reference (BM3) potential energy and E_th is thermal energy.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Internal energy [J/kg]
        """
        # Cold energy (BM3)
        U0 = self.params['U0']
        V0 = self.params['V0']
        K0 = self.params['K0']
        Kp = self.params['K0_prime']
        x = V0 / V
        E_cold = U0 + (9/16) * K0 * V0 * \
                 ((x**(2/3) - 1)**3 * Kp + (x**(2/3) - 1)**2 * (6 - 4*x**(2/3)))
        
        # Thermal energy (Debye)
        E_th = self._thermal_energy(V, T)
        E_th_ref = self._thermal_energy(V, self.T0)
        
        # Convert from J/mol to J/kg
        E_molar = E_cold + E_th - E_th_ref
        return E_molar / ATOMIC_MASS_FE
    
    def _isochoric_heat_capacity(self, V: float, T: float) -> float:
        """
        Calculate molar isochoric heat capacity.
        
        C_V = (∂E/∂T)_V
        
        For Debye model:
        C_V = 9nR(T/θ_D)^3 ∫_0^(θ_D/T) x^4 exp(x)/(exp(x) - 1)^2 dx
        
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
        
        theta_D = self._debye_temperature(V)
        n = self.params['n']
        
        x_max = theta_D / T
        
        # High temperature limit: C_V -> 3nR (Dulong-Petit)
        if x_max < 0.01:
            return 3 * n * R_GAS
        
        # Low temperature limit: C_V -> 0
        if x_max > 100:
            return 0.0
        
        # Numerical integration of the C_V integrand
        n_points = 1000
        x = np.linspace(1e-10, x_max, n_points)
        
        # Integrand: x^4 exp(x)/(exp(x) - 1)^2
        exp_x = np.exp(x)
        integrand = x**4 * exp_x / (exp_x - 1)**2
        
        integral = np.trapezoid(integrand, x)
        
        return 9 * n * R_GAS * (T / theta_D)**3 * integral
    
    def _isothermal_bulk_modulus(self, V: float, T: float) -> float:
        """
        Calculate isothermal bulk modulus using analytical derivatives.
        
        K_T = -V(∂P/∂V)_T
        
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
        V0 = self.params['V0']
        K0 = self.params['K0']
        K0_prime = self.params['K0_prime']
        q = self.params['q']
        
        x = V0 / V
        gamma = self._gruneisen_parameter(V)
        
        # Cold pressure derivative: ∂P_cold/∂V
        f1 = x**(7/3) - x**(5/3)
        f1_prime = (7/3) * x**(4/3) - (5/3) * x**(2/3)
        f2 = 1 + (3/4) * (K0_prime - 4) * (x**(2/3) - 1)
        f2_prime = (1/2) * (K0_prime - 4) * x**(-1/3)
        
        dP_cold_dV = -(3/2) * (x/V) * K0 * (f1_prime * f2 + f1 * f2_prime)
        
        # Thermal pressure derivative: ∂P_th/∂V
        P_th = self._thermal_pressure(V, T)
        Cv_T = self._isochoric_heat_capacity(V, T)
        Cv_T0 = self._isochoric_heat_capacity(V, self.T0)
        
        dP_th_dV = (q - 1 - gamma) * P_th/V + (gamma/V)**2 * (T*Cv_T - self.T0*Cv_T0)
        
        # Total: K_T = -V(∂P/∂V)_T
        dP_dV = dP_cold_dV + dP_th_dV
        
        return -V * dP_dV
    
    def _thermal_expansion_coeff(self, V: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient using analytical relations.
        
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
        if T < 1e-6:
            return 0.0
        
        gamma = self._gruneisen_parameter(V)
        Cv = self._isochoric_heat_capacity(V, T)
        K_T = self._isothermal_bulk_modulus(V, T)
        
        if abs(K_T) < 1e-6:
            return 0.0
        
        # (∂P/∂T)_V = (γ/V) C_V
        dP_dT = (gamma / V) * Cv
        
        return dP_dT / K_T
    
    # =============================================================================
    # Public interface
    # =============================================================================
    
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
        return ATOMIC_MASS_FE / V
    
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
        return self._internal_energy(V, T)
    
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
        return self._entropy(V, T)
    
    def isobaric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isobaric heat capacity.
        
        C_P = C_V + α^2 T V K_T
        
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
        Kt = self._isothermal_bulk_modulus(V, T)
        
        Cp_molar = Cv + alpha**2 * T * V * Kt
        return Cp_molar / ATOMIC_MASS_FE
    
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
        return Cv_molar / ATOMIC_MASS_FE
    
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
        
        (∂ln T/∂ln P)_S = (α P) / (C_P ρ)
        
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


class Hakim18:
    """
    Equation of state for hcp-Fe at super-Earth core conditions from Hakim et al. (2018).
    
    Reference:
    Hakim, K., Rivoldini, A., van Hoolst, T., Cottenier, S., Jaeken, J., Chust, T., Steinle-Neumann, G. (2018)
    "A new ab initio equation of state of hcp-Fe and its implication on the interior structure 
    and mass-radius relations of rocky super-Earths"
    Icarus, 313:61-78, DOI: 10.1016/j.icarus.2018.05.005
    
    This implementation uses:
    - Holzapfel equation for cold compression (valid for P > 234 GPa)
    - Einstein model for quasiharmonic thermal pressure (from Bouchet et al. 2013)
    - Anharmonic-electronic contribution for high-T thermal pressure
    
    The equation of state (SEOS) is based on DFT calculations valid up to 137 TPa
    (pressures relevant for super-Earth cores) and uses experimental data from 
    Fei et al. (2016) for P < 234 GPa.
    
    Thermodynamic model:
    P(V,T) = P_cold(V) + P_harm(V,T) + P_ae(V,T)
    
    where:
    - P_cold(V): Cold (0 K) pressure via Holzapfel equation
    - P_harm(V,T): Quasiharmonic thermal pressure (Einstein model)
    - P_ae(V,T): Anharmonic-electronic thermal pressure
    
    All methods take pressure P [Pa] and temperature T [K] as inputs and
    return quantities in SI units unless otherwise specified.

    Attributes
    ----------
    T0 : float
        Reference temperature (300 K)
    params : dict
        EoS parameters from the paper
    
    Examples
    --------
    >>> # Create instance for hcp-Fe at super-Earth conditions
    >>> eos = Hakim18()
    >>> rho = eos.density(P=1e12, T=5000)  # 1 TPa, 5000 K
    >>> print(f"Density: {rho:.1f} kg/m³")
    """
    
    def __init__(self):
        """
        Initialize the Hakim18 EoS for hcp-Fe.
        
        Parameters are from:
        - Cold EoS: Section 3.1, Eq. (5) and parameters below Eq. (5)
        - Thermal model: Section 3.2, Eqs. (6-8), Table F.7 (Bouchet et al. 2013)
        """
        
        # Reference temperature
        self.T0 = 300.0  # K (ambient temperature)
        
        # Cold compression parameters (Holzapfel equation, Section 3.1)
        # Thermal model parameters from Bouchet et al. (2013), Table F.7
        self.params = {
            #'U0': 0.0,              # J/mol
            'U0': 1127.986e3,       # J/mol
            'S0': -124.828,         # J/(mol·K)
            'V0': 4.28575e-6,       # m³/mol
            'P0': 234.4e9,          # Pa
            'KT0': 1145.7e9,        # Pa
            'c0': 3.19,             # dimensionless
            'c2': -2.40,            # dimensionless
            'Theta0': 44.574,       # K
            'gamma0': 1.408,        # dimensionless
            'gamma_inf': 0.827,     # dimensionless
            'b': 0.826,             # dimensionless
            'a0': 0.2121e-3,        # K^-1
            'm': 1.891,             # dimensionless
            'n': 1.0,               # dimensionless
        }
        
        # Volume adjustment factor (Section 3.2)
        # Bouchet et al. used ρ0 = 8878 kg/m³, giving V0_Bouchet = 6.29027e-6 m³/mol
        # We adjust x by multiplying by (V0_SEOS / V0_Bouchet) = 0.68133
        self.V0_Bouchet = 6.29027e-6  # m³/mol
        self.volume_adjustment = self.params['V0'] / self.V0_Bouchet  # 0.68133
        
    def _adjusted_x(self, V: float) -> float:
        """
        Calculate adjusted volume ratio for thermal pressure calculations.
        
        As noted in Section 3.2: "Since our reference volume V_0 is not the same
        as that of Bouchet et al. (2013), we replace x in the equations for the
        quasiharmonic and anharmonic-electronic thermal pressures by x(V_0/V_0,Bouchet)"
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Adjusted volume ratio (dimensionless)
        """
        x = V / self.params['V0']
        return x * self.volume_adjustment
    
    def _cold_pressure(self, V: float) -> float:
        """
        Calculate cold (0 K) pressure using Holzapfel equation.
        
        Equation (5): 
        P = P_0 + 3 K_T,0 x^{-5/3} (1 - x^{1/3}) (1 + c_2 x^{1/3} (1 - x^{1/3})) 
                          exp[c_0 (1 - x^{1/3})]
        
        where x = V/V_0
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Cold pressure [Pa]
        """
        x = V / self.params['V0']
        x_third = x**(1/3)
        
        # Holzapfel equation components
        term1 = 3 * self.params['KT0'] * x_third**(-5)
        term2 = (1 - x_third)
        term3 = (1 + self.params['c2'] * x_third * term2)
        term4 = np.exp(self.params['c0'] * term2)
        
        return self.params['P0'] + term1 * term2 * term3 * term4
    
    def _gruneisen_parameter(self, V: float) -> float:
        """
        Calculate Grüneisen parameter.
        
        Equation from Section 3.2:
        γ = γ_∞ + (γ_0 - γ_∞)x^b
        
        Uses adjusted volume ratio for consistency with Bouchet et al. (2013).
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Grüneisen parameter (dimensionless)
        """
        x = self._adjusted_x(V)
        gamma0 = self.params['gamma0']
        gamma_inf = self.params['gamma_inf']
        b = self.params['b']

        return gamma_inf + (gamma0 - gamma_inf) * x**b

    def _q_parameter(self, V: float) -> float:
        """
        Calculate q parameter (volume derivative of Grüneisen parameter).
        
        q = b x^b (γ_0 - γ_∞) / γ
        where x = V/V_0
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            q parameter (dimensionless)
        """
        x = V / self.params['V0']
        gamma = self._gruneisen_parameter(V)
        gamma_inf = self.params['gamma_inf']
        gamma0 = self.params['gamma0']
        b = self.params['b']
        
        if abs(gamma) < 1e-10:
            return 0.0
        
        return b * x**b * (gamma0 - gamma_inf) / gamma
    
    def _einstein_temperature(self, V: float) -> float:
        """
        Calculate Einstein temperature.
        
        Equation (7):
        Θ = Θ_0 x^{-γ_∞} exp[(γ_0 - γ_∞)/b (1 - x^b)]
        
        Uses adjusted volume ratio for consistency with Bouchet et al. (2013).
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Einstein temperature [K]
        """
        x = self._adjusted_x(V)
        Theta0 = self.params['Theta0']
        gamma0 = self.params['gamma0']
        gamma_inf = self.params['gamma_inf']
        b = self.params['b']
        
        exponent = (gamma0 - gamma_inf) / b * (1 - x**b)
        
        return Theta0 * x**(-gamma_inf) * np.exp(exponent)
    
    def _thermal_pressure_harmonic(self, V: float, T: float) -> float:
        """
        Calculate quasiharmonic thermal pressure.
        
        Equation (6):
        P_harm(V,T) = (3nRγ/V) [Θ/2 + Θ/(exp(Θ/T) - 1)]
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Harmonic thermal pressure [Pa]
        """
        n = self.params['n']
        
        gamma = self._gruneisen_parameter(V)
        Theta = self._einstein_temperature(V)
        
        # Avoid overflow for large Theta/T or zero-point energy
        if Theta / T > 100 or T < 1e-6:
            # In this limit, exp(Θ/T) >> 1, so the second term → 0
            return (3 * n * R_GAS * gamma / V) * (Theta / 2)
        
        # Full expression
        return (3 * n * R_GAS * gamma / V) * (Theta / 2 + Theta / (np.exp(Theta / T) - 1))
    
    def _thermal_pressure_anharmonic(self, V: float, T: float) -> float:
        """
        Calculate anharmonic-electronic thermal pressure.
        
        Equation (8):
        P_ae(V,T) = (3nR)/(2V) m a_0 x^m T^2
        
        Uses adjusted volume ratio for consistency with Bouchet et al. (2013).
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Anharmonic-electronic thermal pressure [Pa]
        """
        x = self._adjusted_x(V)
        a0 = self.params['a0']
        m = self.params['m']
        n = self.params['n']
        
        return (3 * n * R_GAS * m * a0 * x**m * T**2) / (2 * V)
    
    def _total_pressure(self, V: float, T: float) -> float:
        """
        Calculate total pressure at given volume and temperature.
        
        P(V,T) = P_cold(V) + P_harm(V,T) + P_ae(V,T)
        
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
        P_cold = self._cold_pressure(V)
        P_harm = self._thermal_pressure_harmonic(V, T)
        P_ae = self._thermal_pressure_anharmonic(V, T)
        
        return P_cold + P_harm + P_ae
    
    def _find_volume(self, P: float, T: float) -> float:
        """
        Find molar volume for given pressure and temperature by root finding.
        
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
            If root finding fails to converge
        """
        # Set reasonable bounds for volume search
        # Allow compression to 0.1*V0 (for extreme super-Earth pressures)
        # and expansion to 1.5*V0
        V_min = 0.1 * self.params['V0']
        V_max = 1.5 * self.params['V0']
        
        # Define function whose root we seek
        def pressure_residual(V):
            return self._total_pressure(V, T) - P
        
        try:
            V = brentq(pressure_residual, V_min, V_max, xtol=1e-12)
            return V
        except ValueError:
            raise RuntimeError(
                f"Could not find volume for P = {P/1e9:.1f} GPa, T = {T:.1f} K. "
                f"Pressure may be outside valid range for this EoS (P > 234 GPa recommended)."
            )
    
    def _isochoric_heat_capacity(self, V: float, T: float) -> float:
        """
        Calculate molar isochoric heat capacity.
        
        From thermodynamic relations and Equations (6, 8):
        C_V = 3nR (Θ/T)^2 exp(Θ/T)/[exp(Θ/T) - 1]^2 + 3nR a_0 x^m T
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Molar isochoric heat capacity [J/(mol·K)]
        """
        if T < 1e-6:
            return 0.0
        
        Theta = self._einstein_temperature(V)
        x = self._adjusted_x(V)
        a0 = self.params['a0']
        m = self.params['m']
        n = self.params['n']
        
        # Harmonic contribution (Einstein model)
        if Theta / T > 100:
            # At high Theta/T, the harmonic contribution vanishes
            Cv_harm = 0.0
        else:
            exp_term = np.exp(Theta / T)
            Cv_harm = 3 * n * R_GAS * (Theta / T)**2 * exp_term / (exp_term - 1)**2
        
        # Anharmonic-electronic contribution
        Cv_ae = 3 * n * R_GAS * a0 * x**m * T
        
        return Cv_harm + Cv_ae
    
    def _thermal_expansion_coeff(self, V: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient.
        
        α = (∂P/∂T)_V / K_T
        
        where (∂P/∂T)_V = γ C_V,harm/V + m C_V,ae/V
        
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
        
        n = self.params['n']
        m = self.params['m']
        a0 = self.params['a0']
        
        gamma = self._gruneisen_parameter(V)
        Theta = self._einstein_temperature(V)
        x = self._adjusted_x(V)
        
        # Harmonic heat capacity: C_V,harm = 3nR (Θ/T)^2 exp(Θ/T)/[exp(Θ/T) - 1]^2
        if Theta / T > 100:
            Cv_harm = 0.0
        else:
            exp_term = np.exp(Theta / T)
            Cv_harm = 3 * n * R_GAS * (Theta / T)**2 * exp_term / (exp_term - 1)**2
        
        # Anharmonic-electronic heat capacity: C_V,ae = 3nR a_0 x^m T
        Cv_ae = 3 * n * R_GAS * a0 * x**m * T
        
        # Temperature derivative of pressure at constant volume
        dP_dT_V = gamma * Cv_harm / V + m * Cv_ae / V
        
        # Isothermal bulk modulus
        KT = self._isothermal_bulk_modulus(V, T)
        
        return dP_dT_V / KT
    
    def _isothermal_bulk_modulus(self, V: float, T: float) -> float:
        """
        Calculate isothermal bulk modulus.
        
        K_T = -V (∂P/∂V)_T
        
        where (∂P/∂V)_T = ∂P_cold/∂V + ∂P_harm/∂V + ∂P_ae/∂V
        
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
        n = self.params['n']
        V0 = self.params['V0']
        P0 = self.params['P0']
        c0 = self.params['c0']
        c2 = self.params['c2']
        m = self.params['m']
        a0 = self.params['a0']
        
        x = V / V0
        y = x**(1/3)
        x_adj = self._adjusted_x(V)
        
        # Cold pressure derivative
        # ∂P_cold/∂V = (P_cold - P_0)/(3V) g(y)
        P_cold = self._cold_pressure(V)
        g_y = -5 - y/(1 - y) + c2*y*(1 - 2*y)/(1 + c2*y*(1 - y)) - c0*y
        dP_cold_dV = (P_cold - P0) / (3 * V) * g_y
        
        # Harmonic pressure and its components
        gamma = self._gruneisen_parameter(V)
        q = self._q_parameter(V)
        Theta = self._einstein_temperature(V)
        P_harm = self._thermal_pressure_harmonic(V, T)
        
        if T < 1e-6 or Theta / T > 100:
            Cv_harm = 0.0
        else:
            exp_term = np.exp(Theta / T)
            Cv_harm = 3 * n * R_GAS * (Theta / T)**2 * exp_term / (exp_term - 1)**2
        
        # ∂P_harm/∂V = (q - γ - 1) P_harm/V + (γ/V)² C_V,harm T
        dP_harm_dV = (q - gamma - 1) * P_harm / V + (gamma / V)**2 * Cv_harm * T
        
        # Anharmonic-electronic pressure derivative
        P_ae = self._thermal_pressure_anharmonic(V, T)
        Cv_ae = 3 * n * R_GAS * a0 * x_adj**m * T
        
        # ∂P_ae/∂V = -(1 + m) P_ae/V + (m/V)² C_V,ae T
        dP_ae_dV = -(1 + m) * P_ae / V + (m / V)**2 * Cv_ae * T
        
        # Total derivative
        dP_dV = dP_cold_dV + dP_harm_dV + dP_ae_dV
        
        return -V * dP_dV
    
    def _entropy(self, V: float, T: float) -> float:
        """
        Calculate specific entropy.
        
        S = -(∂F/∂T)_V
        
        For the Einstein + anharmonic-electronic model:
        S = 3nR[-ln(1 - exp(-Θ/T)) + (Θ/T)/(exp(Θ/T) - 1)] + 3nR a_0 x^m T
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific entropy [J/(kg·K)]
        """
        S0 = self.params['S0']

        if T < 1e-6:
            return 0.0
        
        Theta = self._einstein_temperature(V)
        x = self._adjusted_x(V)
        a0 = self.params['a0']
        m = self.params['m']
        n = self.params['n']
        
        # Harmonic contribution (Einstein model)
        if Theta / T > 100:
            S_harm = 0.0
        else:
            exp_term = np.exp(Theta / T)
            S_harm = 3 * n * R_GAS * (
                -np.log(1 - 1/exp_term) + 
                (Theta / T) / (exp_term - 1)
            )
        
        # Anharmonic-electronic contribution
        S_ae = 3 * n * R_GAS * a0 * x**m * T
        
        # Convert from J/(mol·K) to J/(kg·K)
        S_molar = S_harm + S_ae + S0
        return S_molar / ATOMIC_MASS_FE
    
    def _internal_energy(self, V: float, T: float) -> float:
        """
        Calculate specific internal energy.
        
        E = E_cold + E_harm + E_ae
        
        where:
        - E_cold = U_0 - P_0 V - 3 K_T,0 V_0 I(x)
        - E_harm = P_harm V / γ
        - E_ae = P_ae V / m
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific internal energy [J/kg]
        """
        from scipy.integrate import quad
        
        U0 = self.params['U0']
        V0 = self.params['V0']
        P0 = self.params['P0']
        KT0 = self.params['KT0']
        c0 = self.params['c0']
        c2 = self.params['c2']
        m = self.params['m']
        
        x = V / V0
        
        # Cold energy: E_cold = U_0 - P_0 V - 3 K_T,0 V_0 I(x)
        # where I(x) = ∫_1^x f(x') dx'
        # and f(x) = x^(-5/3) (1 - x^(1/3)) (1 - c_2 x^(1/3) (1 - x^(1/3))) 
        #            exp(c_0 (1 - x^(1/3)))
        
        def f_integrand(x_prime):
            x_third = x_prime**(1/3)
            return (x_third**(-5) * (1 - x_third) * (1 - c2 * x_third * (1 - x_third)) * 
                    np.exp(c0 * (1 - x_third)))
        
        if abs(x - 1.0) < 1e-10:
            I_x = 0.0
        else:
            I_x, _ = quad(f_integrand, 1.0, x, epsabs=1e-12, epsrel=1e-12)
        
        E_cold = U0 - P0 * V - 3 * KT0 * V0 * I_x
        
        # Harmonic energy: E_harm = P_harm V / γ
        gamma = self._gruneisen_parameter(V)
        P_harm = self._thermal_pressure_harmonic(V, T)
        E_harm = P_harm * V / gamma
        
        # Anharmonic-electronic energy: E_ae = P_ae V / m
        P_ae = self._thermal_pressure_anharmonic(V, T)
        E_ae = P_ae * V / m
        
        # Total molar internal energy
        E_molar = E_cold + E_harm + E_ae
        
        # Convert to specific (per kg)
        return E_molar / ATOMIC_MASS_FE
    
    # =============================================================================
    # Public interface
    # =============================================================================
    
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
        return ATOMIC_MASS_FE / V
    
    def specific_internal_energy(self, P: float, T: float) -> float:
        """
        Calculate specific internal energy at given pressure and temperature.
        
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
        return self._internal_energy(V, T)
    
    def specific_entropy(self, P: float, T: float) -> float:
        """
        Calculate specific entropy at given pressure and temperature.
        
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
        return self._entropy(V, T)
    
    def isobaric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isobaric heat capacity.
        
        C_P = C_V + α^2 T V K_T / m
        
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
        Cv_molar = self._isochoric_heat_capacity(V, T)
        alpha = self._thermal_expansion_coeff(V, T)
        KT = self._isothermal_bulk_modulus(V, T)
        
        Cp_molar = Cv_molar + alpha**2 * T * V * KT
        
        # Convert to specific (per kg)
        return Cp_molar / ATOMIC_MASS_FE
    
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
        return Cv_molar / ATOMIC_MASS_FE
    
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
        
        (∂ln T/∂ln P)_S = (α P) / (C_P ρ)
        
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


class HcpIronEos:
    """
    Composite equation of state for hcp-Fe (ε-phase) with smooth transition.
    
    This class blends Miozzi20 and Hakim18 EoS across a transition zone
    around 310 GPa to ensure smooth thermodynamic properties.
    
    The blending uses a smoothstep function:
        w(P) = 3t² - 2t³  where t = (P - P_low) / (P_high - P_low)
    
    For P < P_low:  pure Miozzi20
    For P > P_high: pure Hakim18
    In between:     smooth interpolation
    
    All seven thermodynamic quantities are blended directly.
    
    Parameters
    ----------
    P_center : float, optional
        Center of transition zone [Pa]. Default: 310 GPa
    delta_P : float, optional
        Half-width of transition zone [Pa]. Default: 30 GPa
    
    Attributes
    ----------
    eos_low : Miozzi20
        EoS for lower pressure range
    eos_high : Hakim18
        EoS for higher pressure range
    P_low : float
        Lower bound of transition zone [Pa]
    P_high : float
        Upper bound of transition zone [Pa]
    
    Examples
    --------
    >>> eos = HcpIronEos()
    >>> rho = eos.density(300e9, 4000)  # In transition zone
    >>> rho = eos.density(200e9, 3000)  # Pure Miozzi20
    >>> rho = eos.density(400e9, 5000)  # Pure Hakim18
    """
    
    def __init__(self, P_center: float = 310e9, delta_P: float = 100e9):
        """
        Initialize the composite hcp-Fe EoS.
        
        Parameters
        ----------
        P_center : float, optional
            Center of transition zone [Pa]. Default: 310 GPa
        delta_P : float, optional
            Half-width of transition zone [Pa]. Default: 100 GPa
        """
        self.eos_low = Miozzi20()
        self.eos_high = Hakim18()
        
        self.P_center = P_center
        self.delta_P = delta_P
        self.P_low = P_center - delta_P
        self.P_high = P_center + delta_P
    
    def _smoothstep(self, P: float) -> float:
        """
        Compute smoothstep weight function.
        
        Returns 0 for P <= P_low, 1 for P >= P_high,
        and smooth interpolation in between.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
            
        Returns
        -------
        float
            Weight in [0, 1]
        """
        if P <= self.P_low:
            return 0.0
        elif P >= self.P_high:
            return 1.0
        else:
            t = (P - self.P_low) / (self.P_high - self.P_low)
            return t * t * (3 - 2 * t)
    
    def _blend(self, P: float, val_low: float, val_high: float) -> float:
        """
        Blend two values using the smoothstep weight.
        
        Parameters
        ----------
        P : float
            Pressure [Pa]
        val_low : float
            Value from Miozzi20
        val_high : float
            Value from Hakim18
            
        Returns
        -------
        float
            Blended value
        """
        w = self._smoothstep(P)
        return (1 - w) * val_low + w * val_high
    
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
        if P <= self.P_low:
            return self.eos_low.density(P, T)
        elif P >= self.P_high:
            return self.eos_high.density(P, T)
        else:
            return self._blend(P, 
                               self.eos_low.density(P, T),
                               self.eos_high.density(P, T))
    
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
        if P <= self.P_low:
            return self.eos_low.specific_internal_energy(P, T)
        elif P >= self.P_high:
            return self.eos_high.specific_internal_energy(P, T)
        else:
            return self._blend(P,
                               self.eos_low.specific_internal_energy(P, T),
                               self.eos_high.specific_internal_energy(P, T))
    
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
        if P <= self.P_low:
            return self.eos_low.specific_entropy(P, T)
        elif P >= self.P_high:
            return self.eos_high.specific_entropy(P, T)
        else:
            return self._blend(P,
                               self.eos_low.specific_entropy(P, T),
                               self.eos_high.specific_entropy(P, T))
    
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
        if P <= self.P_low:
            return self.eos_low.isobaric_heat_capacity(P, T)
        elif P >= self.P_high:
            return self.eos_high.isobaric_heat_capacity(P, T)
        else:
            return self._blend(P,
                               self.eos_low.isobaric_heat_capacity(P, T),
                               self.eos_high.isobaric_heat_capacity(P, T))
    
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
        if P <= self.P_low:
            return self.eos_low.isochoric_heat_capacity(P, T)
        elif P >= self.P_high:
            return self.eos_high.isochoric_heat_capacity(P, T)
        else:
            return self._blend(P,
                               self.eos_low.isochoric_heat_capacity(P, T),
                               self.eos_high.isochoric_heat_capacity(P, T))
    
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
        if P <= self.P_low:
            return self.eos_low.thermal_expansion(P, T)
        elif P >= self.P_high:
            return self.eos_high.thermal_expansion(P, T)
        else:
            return self._blend(P,
                               self.eos_low.thermal_expansion(P, T),
                               self.eos_high.thermal_expansion(P, T))
    
    def adiabatic_gradient(self, P: float, T: float) -> float:
        """
        Calculate dimensionless adiabatic temperature gradient.
        
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
        if P <= self.P_low:
            return self.eos_low.adiabatic_gradient(P, T)
        elif P >= self.P_high:
            return self.eos_high.adiabatic_gradient(P, T)
        else:
            return self._blend(P,
                               self.eos_low.adiabatic_gradient(P, T),
                               self.eos_high.adiabatic_gradient(P, T))


class Luo24:
    """
    Equation of state for liquid iron from Luo et al. (2024).
    
    Reference:
    Luo, H., Dorn, C., Deng, J. (2024)
    "The interior as the dominant water reservoir in super-Earths and sub-Neptunes"
    Nature Astronomy 8:1399-1407, DOI: 10.1038/s41550-024-02347-z
    
    This implementation uses:
    - 3rd order Birch-Murnaghan (BM3) for cold compression
    - Linear thermal pressure with polynomial volume dependence
    
    The equation of state is calibrated from ab initio molecular dynamics 
    simulations at conditions relevant to super-Earth cores:
    ~8000-14000 K and ~50-1300 GPa.
    
    Thermodynamic model:
    P(V,T) = P_cold(V) + P_th(V,T)
    
    where:
    - P_cold(V): Birch-Murnaghan 3rd order cold pressure
    - P_th(V,T) = (T - T_0)/1000 × [a + b(V_0/V) + c(V_0/V)^2]
    
    All methods take pressure P [Pa] and temperature T [K] as inputs and
    return quantities in SI units.
    
    Attributes
    ----------
    T0 : float
        Reference temperature (8000 K)
    params : dict
        EoS parameters from the paper
    
    Examples
    --------
    >>> eos = Luo24()
    >>> rho = eos.density(P=500e9, T=9000)  # 500 GPa, 9000 K
    >>> print(f"Density: {rho:.1f} kg/m³")
    
    Notes
    -----
    This EoS is designed for liquid iron at extreme conditions (T > ~5000 K,
    P > ~50 GPa). The Dulong-Petit limit is assumed for heat capacity.
    """
    
    def __init__(self):
        """
        Initialize the Luo24 EoS for liquid iron.
        
        Parameters from Methods section, Equation (14) and text below:
        - V0 = 1043.912 Å³ (for 64 Fe atoms in supercell)
        - K_{T0} = 49.249 GPa
        - K'_{T0} = 4.976
        - T0 = 8000 K
        - a = -15.957 GPa
        - b = 20.946 GPa
        - c = -3.811 GPa
        """
        
        # Reference temperature
        self.T0 = 8000.0  # K
        
        # Convert V0 from 64-atom supercell to molar volume
        V0_cell = 1043.912e-30  # m³ (for 64 atoms)
        V0_molar = V0_cell * N_AVOGADRO / 64  # m³/mol
        
        # EoS parameters
        self.params = {
            #'U0': 0.0,              # J/mol
            'U0': 187.884e3,        # J/mol
            'S0': 44.828,           # J/(mol·K)
            'V0': V0_molar,         # m³/mol
            'K0': 49.249e9,         # Pa
            'K0_prime': 4.976,      # dimensionless
            'a': -15.957e9,         # Pa
            'b': 20.946e9,          # Pa
            'c': -3.811e9,          # Pa
            'n': 1.0,               # dimensionless
        }
        
        # Heat capacity in Dulong-Petit limit for liquid
        # C_V = 3nR per mole
        self.Cv_molar = 3 * self.params['n'] * R_GAS  # J/(mol·K)
    
    # =========================================================================
    # Helper functions for EoS components
    # =========================================================================
    
    def _cold_pressure(self, V: float) -> float:
        """
        Calculate cold compression pressure using 3rd order Birch-Murnaghan.
        
        P_cold = (3/2) K_0 [x^(7/3) - x^(5/3)] × {1 + (3/4)(K'_0 - 4)[x^(2/3) - 1]}
        
        where x = V_0/V
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Cold compression pressure [Pa]
        """
        V0 = self.params['V0']
        K0 = self.params['K0']
        K0_prime = self.params['K0_prime']
        
        x = V0 / V  # compression ratio
        f = x**(2/3) - 1  # Eulerian strain term
        
        P = (3/2) * K0 * (x**(7/3) - x**(5/3)) * (1 + (3/4) * (K0_prime - 4) * f)
        
        return P
    
    def _thermal_pressure(self, V: float, T: float) -> float:
        """
        Calculate thermal pressure contribution.
        
        P_th = (T - T_0)/1000 [a + b(V_0/V) + c(V_0/V)^2]
        
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
        V0 = self.params['V0']
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        
        x = V0 / V  # compression ratio
        
        # Polynomial in compression ratio
        poly = a + b * x + c * x**2
        
        return (T - self.T0) / 1000.0 * poly
    
    def _thermal_pressure_coeff(self, V: float) -> float:
        """
        Calculate coefficient of thermal pressure: (∂P_th/∂T)_V.
        
        (∂P_th/∂T)_V = [a + b(V_0/V) + c(V_0/V)^2] / 1000
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Thermal pressure coefficient [Pa/K]
        """
        V0 = self.params['V0']
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        
        x = V0 / V
        
        return (a + b * x + c * x**2) / 1000.0
    
    def _total_pressure(self, V: float, T: float) -> float:
        """
        Calculate total pressure at given volume and temperature.
        
        P(V,T) = P_cold(V) + P_th(V,T)
        
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
        
        Uses Brent's method to solve P(V,T) = P_target.
        
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
        
        # Volume search bounds
        # Compressed to 20% of V0 (extreme super-Earth pressures)
        # Expanded to 150% of V0 (lower pressures)
        V_min = 0.2 * V0
        V_max = 1.5 * V0
        
        def pressure_residual(V):
            return self._total_pressure(V, T) - P
        
        try:
            V = brentq(pressure_residual, V_min, V_max, xtol=1e-12)
            return V
        except ValueError:
            try:
                V_min *= 0.4
                V_max = V0
                V = brentq(pressure_residual, V_min, V_max, xtol=1e-12)
                return V 
            except ValueError:
                raise RuntimeError(
                    f"Failed to find volume at P = {P/1e9:.2f} GPa, T = {T:.1f} K. "
                    f"Pressure may be outside valid range for liquid iron EoS."
                )
    
    def _isothermal_bulk_modulus(self, V: float, T: float) -> float:
        """
        Calculate isothermal bulk modulus analytically.
        
        K_T = -V (∂P/∂V)_T = K_T,cold + K_T,th
        
        For BM3:
        ∂P_cold/∂V = -(x/V) (3/2) K_0 [f1' f2 + f1 f2']
        where:
        - f1 = x^(7/3) - x^(5/3)
        - f1' = (7/3)x^(4/3) - (5/3)x^(2/3)
        - f2 = 1 + (3/4)(K'_0 - 4)(x^(2/3) - 1)
        - f2' = (1/2)(K'_0 - 4)x^(-1/3)
        
        For thermal:
        ∂P_th/∂V = -(x/V) (T - T0)/1000 [b + 2 c x]
        
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
        V0 = self.params['V0']
        K0 = self.params['K0']
        K0_prime = self.params['K0_prime']
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        
        x = V0 / V
        
        # Cold pressure derivative (BM3)
        f1 = x**(7/3) - x**(5/3)
        f1_prime = (7/3) * x**(4/3) - (5/3) * x**(2/3)
        f2 = 1 + (3/4) * (K0_prime - 4) * (x**(2/3) - 1)
        f2_prime = (1/2) * (K0_prime - 4) * x**(-1/3)
        
        dP_cold_dV = -(x / V) * (3/2) * K0 * (f1_prime * f2 + f1 * f2_prime)
        
        # Thermal pressure derivative
        dP_th_dV = ((T - self.T0) / 1000.0) * (-x / V) * (b + 2 * c * x)
        
        # Total
        dP_dV = dP_cold_dV + dP_th_dV
        
        return -V * dP_dV
    
    def _thermal_expansion_coeff(self, V: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient.
        
        α = (∂P/∂T)_V / K_T
        
        A softplus floor is applied to ensure α ≥ α_min smoothly:
        α_smooth = α_min × (1 + softplus((α_raw - α_min) / α_min))
        
        This avoids discontinuities that would arise from a hard cutoff
        while ensuring thermal expansion remains positive.
        
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
        dP_dT = self._thermal_pressure_coeff(V)
        K_T = self._isothermal_bulk_modulus(V, T)
        
        if abs(K_T) < 1e-6:
            return 1e-7  # Return alpha_min
        
        alpha_raw = dP_dT / K_T
        
        # Softplus floor to ensure α ≥ α_min smoothly
        # Using scaled softplus: α = α_min × (1 + softplus((α_raw - α_min) / α_min))
        # This ensures: α → α_raw when α_raw >> α_min
        #               α → α_min when α_raw << α_min
        alpha_min = 1e-7
        
        x = (alpha_raw - alpha_min) / alpha_min
        
        # Numerical stability
        if x > 30:
            # softplus(x) ≈ x for large x
            return alpha_raw
        elif x < -30:
            # softplus(x) ≈ 0 for large negative x
            return alpha_min
        else:
            return alpha_min * (1 + np.log(1 + np.exp(x)))
    
    def _isochoric_heat_capacity(self, V: float, T: float) -> float:
        """
        Calculate molar isochoric heat capacity.
        
        For liquid iron at high temperatures (T >> θ_D), we use the 
        Dulong-Petit limit: C_V = 3nR.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Molar isochoric heat capacity [J/(mol·K)]
        """
        # Dulong-Petit limit for liquid at high T
        return self.Cv_molar
    
    def _entropy(self, V: float, T: float) -> float:
        """
        Calculate specific entropy.
        
        For constant C_V (Dulong-Petit limit) + thermal part:
        S = C_V × ln(T/T_0) + (a(V - V_0) + b V_0 ln(V/V_0) + c V_0 (1 - V_0/V))/1000
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific entropy [J/(kg·K)]
        """
        S0 = self.params['S0']
        V0 = self.params['V0']
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']

        if T < 1e-6:
            return 0.0
        
        # Dulong-Petit entropy (we force this term to have an explicit T dependence)
        # S_DP = ∫(C_V/T)dT from T0 to T = C_V × ln(T/T0)
        S_DP = self.Cv_molar * np.log(T / self.T0)

        # Thermal entropy (from Helmholtz free energy)
        # S_th = -∂F_th/∂T and F_th = -∫P_th dV
        S_th = (a*(V - V0) + b*V0*np.log(V/V0) + c*V0*(1 - V0/V))/1000

        # Total
        S_molar = S_th + S_DP + S0
        
        return S_molar / ATOMIC_MASS_FE
    
    def _internal_energy(self, V: float, T: float) -> float:
        """
        Calculate specific internal energy.
        
        E = E_cold + E_th + E_DP
        
        where:
        - E_cold: BM3 potential energy (integrated from V0 to V)
        - E_th = T_0/1000 (a(V - V_0) + b V_0 ln(V/V_0) + c V_0 (1 - V_0/V))
        - E_DP = C_V × (T - T_0) consistent with Dulong-Petit
        
        For BM3, the cold energy is:
        E_cold = (9/16) K_0 V_0 × [(x^(2/3) - 1)^3 K'_0 + (x^(2/3) - 1)^2 (6 - 4x^(2/3))]
        
        where x = V_0/V
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific internal energy [J/kg]
        """
        U0 = self.params['U0']
        V0 = self.params['V0']
        K0 = self.params['K0']
        K0_prime = self.params['K0_prime']
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        
        x = V0 / V
        f = x**(2/3) - 1  # Eulerian strain
        
        # BM3 cold energy
        E_cold = U0 + (9/16) * K0 * V0 * (f**3 * K0_prime + f**2 * (6 - 4 * x**(2/3)))

        # Thermal energy
        # E_th = F_th + T S_th 
        E_th = (a*(V - V0) + b*V0*np.log(V/V0) + c*V0*(1 - V0/V))*self.T0/1000
        
        # Dulong-Petit energy (we force this term to have an explicit T dependence)
        E_DP = self.Cv_molar * (T - self.T0)
        
        # Total molar energy
        E_molar = E_cold + E_DP + E_th
        
        return E_molar / ATOMIC_MASS_FE
    
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
        return ATOMIC_MASS_FE / V
    
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
        return self._internal_energy(V, T)
    
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
        return self._entropy(V, T)
    
    def isobaric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isobaric heat capacity.
        
        C_P = C_V + α^2 T V K_T
        
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
        K_T = self._isothermal_bulk_modulus(V, T)
        
        # Molar C_P
        Cp_molar = Cv + alpha**2 * T * V * K_T
        
        return Cp_molar / ATOMIC_MASS_FE
    
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
        return Cv_molar / ATOMIC_MASS_FE
    
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


class Ichikawa20:
    """
    Equation of state for liquid iron from Ichikawa & Tsuchiya (2020).
    
    Reference:
    Ichikawa, H. & Tsuchiya, T. (2020)
    "Ab Initio Thermoelasticity of Liquid Iron-Nickel-Light Element Alloys"
    Minerals 2020, 10, 59, DOI: 10.3390/min10010059
    
    This implementation uses:
    - Vinet (Morse-Rydberg) EoS for the reference isotherm at T0 = 8000 K
    - Constant Grüneisen parameter for thermal pressure
    - Combined phonon + electronic thermal contributions
    
    The EoS is calibrated from ab initio molecular dynamics simulations
    covering outer core P-T conditions: ~100-450 GPa and ~4000-8000 K.
    
    Thermodynamic model:
    P(V,T) = P_cold(V) + P_th(V,T)
    
    where:
    - P_cold(V): Vinet pressure at reference isotherm (T0 = 8000 K)
    - P_th(V,T) = (γ/V)[E_th(T) - E_th(T0)]
    - E_th(V,T) = 3nR[T + e0(V/V0)^g T^2]
    
    The first term in E_th is the phonon (atomic) contribution,
    the second is the electronic contribution.
    
    All methods take pressure P [Pa] and temperature T [K] as inputs and
    return quantities in SI units.
    
    Attributes
    ----------
    T0 : float
        Reference temperature (8000 K)
    params : dict
        EoS parameters from the paper (Table S1 for pure Fe)
    
    Examples
    --------
    >>> eos = Ichikawa20()
    >>> rho = eos.density(P=200e9, T=5000)  # 200 GPa, 5000 K
    >>> print(f"Density: {rho:.1f} kg/m³")
    
    Notes
    -----
    This EoS is designed for liquid iron at outer core conditions.
    Parameters are for pure Fe with TICB = 5000 K from Table S1.
    The electronic coefficient e0 is interpreted as having units of
    10^-4 K^-1 based on dimensional analysis and Ichikawa et al. (2014).
    """
    
    def __init__(self):
        """
        Initialize the Ichikawa20 EoS for liquid iron.
        
        Parameters from Table S1 (Supplementary Material) for pure Fe
        at TICB = 5000 K:
        - V0 = 1.20 × 10^-5 m³/mol
        - K_T0 = 15.68 GPa
        - K'_T0 = 6.86
        - γ = 1.5 (fixed constant)
        - e0 = 0.390 × 10^-6 K^-1 (electronic coefficient)
        - g = -0.070 (volume exponent for electronic term)
        """
        
        # Reference temperature (highest T used in calculations)
        self.T0 = 8000.0  # K
        
        # EoS parameters from Table S1 for pure Fe (TICB = 5000 K)
        self.params = {
            #'U0': 0.0,              # J/mol
            'U0': 138.81e3,         # J/mol
            'S0': 46.847,           # J/(mol·K)
            'V0': 1.20e-5,          # m³/mol
            'KT0': 15.68e9,         # Pa
            'KT0_prime': 6.86,      # dimensionless
            'gamma': 1.5,           # dimensionless
            'e0': 0.390e-4,         # K^-1
            'g': -0.070,            # dimensionless
            'n': 1.0,               # dimensionless
        }
    
    # =========================================================================
    # Helper functions for EoS components
    # =========================================================================
    
    def _cold_pressure(self, V: float) -> float:
        """
        Calculate cold compression pressure using Vinet (Morse-Rydberg) EoS.
        
        Equation (5) in the paper:
        P_T0(V) = 3 K_T0 y^(-2) (1 - y) exp[η(1 - y)]
        
        where y = (V/V0)^(1/3) and η = (3/2)(K'_T0 - 1)
        
        This gives the pressure along the reference isotherm at T0 = 8000 K.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
            
        Returns
        -------
        float
            Cold compression pressure [Pa]
        """
        V0 = self.params['V0']
        KT0 = self.params['KT0']
        KT0_prime = self.params['KT0_prime']
        
        y = (V / V0)**(1/3)  # y = x^(1/3) where x = V/V0
        eta = 1.5 * (KT0_prime - 1)
        
        return 3 * KT0 * y**(-2) * (1 - y) * np.exp(eta * (1 - y))
    
    def _thermal_energy(self, V: float, T: float) -> float:
        """
        Calculate thermal internal energy.
        
        Equation (6) in the paper:
        E_th(V, T) = 3nR [T + e0 (V/V0)^g T^2]
        
        First term: phonon (atomic) contribution
        Second term: electronic contribution with volume dependence
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Thermal internal energy [J/mol]
        """
        V0 = self.params['V0']
        e0 = self.params['e0']
        g = self.params['g']
        n = self.params['n']
        
        x = V / V0  # Volume ratio
        
        # Phonon term + electronic term
        return 3 * n * R_GAS * (T + e0 * x**g * T**2)
    
    def _thermal_pressure(self, V: float, T: float) -> float:
        """
        Calculate thermal pressure contribution.
        
        From Equation (7):
        P_th(V, T) = (γ/V) [E_th(V, T) - E_th(V, T0)]
        
        With constant Grüneisen parameter γ:
        P_th = (3nRγ/V) [(T - T0) + e0 x^g (T^2 - T0^2)]
        
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
        V0 = self.params['V0']
        gamma = self.params['gamma']
        e0 = self.params['e0']
        g = self.params['g']
        n = self.params['n']
        
        x = V / V0
        
        # Temperature differences from reference
        dT = T - self.T0
        dT2 = T**2 - self.T0**2
        
        return (3 * n * R_GAS * gamma / V) * (dT + e0 * x**g * dT2)
    
    def _total_pressure(self, V: float, T: float) -> float:
        """
        Calculate total pressure at given volume and temperature.
        
        P(V,T) = P_cold(V) + P_th(V,T)
        
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
        
        Uses Brent's method to solve P(V,T) = P_target.
        
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
        
        # Volume search bounds
        # Allow compression to 30% of V0 (extreme pressures)
        # and expansion to 200% of V0 (lower pressures / high T)
        V_min = 0.3 * V0
        V_max = 2.0 * V0
        
        def pressure_residual(V):
            return self._total_pressure(V, T) - P
        
        try:
            V = brentq(pressure_residual, V_min, V_max, xtol=1e-12)
            return V
        except ValueError:
            # Expand search range and retry
            try:
                V_min *= 0.1
                V_max *= 10.
                V = brentq(pressure_residual, V_min, V_max, xtol=1e-12)
                return V
            except ValueError:
                raise RuntimeError(
                    f"Failed to find volume at P = {P/1e9:.2f} GPa, T = {T:.1f} K. "
                    f"Pressure may be outside valid range for Ichikawa20 EoS "
                )
    
    def _isothermal_bulk_modulus(self, V: float, T: float) -> float:
        """
        Calculate isothermal bulk modulus analytically.
        
        K_T = -V (∂P/∂V)_T = K_T,cold + K_T,th
        
        For Vinet cold pressure:
        K_T,cold = K_T0 y^(-2) exp(η(1-y)) [2 - y + ηy(1-y)]
        
        where y = (V/V0)^(1/3), η = (3/2)(K'_T0 - 1)
        
        For thermal pressure:
        K_T,th = (3nRγ/V) [(T - T0) - e0 g x^g (T^2 - T0^2)]
        
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
        V0 = self.params['V0']
        KT0 = self.params['KT0']
        KT0_prime = self.params['KT0_prime']
        gamma = self.params['gamma']
        e0 = self.params['e0']
        g = self.params['g']
        n = self.params['n']
        
        y = (V / V0)**(1/3)
        x = V / V0
        eta = 1.5 * (KT0_prime - 1)
        
        # Cold bulk modulus from Vinet EoS
        # K_T,cold = K_T0 y^(-2) exp(η(1-y)) [2 - y + ηy(1-y)]
        KT_cold = KT0 * y**(-2) * np.exp(eta * (1 - y)) * (2 - y + eta * y * (1 - y))
        
        # Thermal contribution to bulk modulus
        # K_T,th = (3nRγ/V) [(T - T0) - e0 g x^g (T^2 - T0^2)]
        dT = T - self.T0
        dT2 = T**2 - self.T0**2
        KT_th = (3 * n * R_GAS * gamma / V) * (dT - e0 * g * x**g * dT2)
        
        return KT_cold + KT_th
    
    def _isochoric_heat_capacity(self, V: float, T: float) -> float:
        """
        Calculate molar isochoric heat capacity.
        
        C_V = ∂E_th/∂T = 3nR [1 + 2 e0 x^g T]
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Molar isochoric heat capacity [J/(mol·K)]
        """
        V0 = self.params['V0']
        e0 = self.params['e0']
        g = self.params['g']
        n = self.params['n']
        
        x = V / V0
        
        # Phonon contribution (3nR) + electronic contribution (6nR e0 x^g T)
        return 3 * n * R_GAS * (1 + 2 * e0 * x**g * T)
    
    def _thermal_expansion_coeff(self, V: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient.
        
        α = (∂P/∂T)_V / K_T
        
        where (∂P/∂T)_V = (3nRγ/V) [1 + 2 e0 x^g T]
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Thermal expansion coefficient [K^-1]
        """
        V0 = self.params['V0']
        gamma = self.params['gamma']
        e0 = self.params['e0']
        g = self.params['g']
        n = self.params['n']
        
        x = V / V0
        
        # Temperature derivative of pressure at constant volume
        # (∂P_th/∂T)_V = (3nRγ/V) [1 + 2 e0 x^g T]
        dP_dT_V = (3 * n * R_GAS * gamma / V) * (1 + 2 * e0 * x**g * T)
        
        # Isothermal bulk modulus
        KT = self._isothermal_bulk_modulus(V, T)
        
        if abs(KT) < 1e-6:
            return 0.0
        
        return dP_dT_V / KT
    
    def _entropy(self, V: float, T: float) -> float:
        """
        Calculate specific entropy.
        
        S = ∫(C_V/T)dT = 3nR [ln(T/T0) + 2 e0 x^g (T - T0)]
        
        The entropy is computed relative to the reference state at T0.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific entropy [J/(kg·K)]
        """
        S0 = self.params['S0']

        if T < 1e-6:
            return 0.0
        
        V0 = self.params['V0']
        e0 = self.params['e0']
        g = self.params['g']
        n = self.params['n']
        
        x = V / V0
        
        # Entropy change from reference state
        # S - S0 = 3nR [ln(T/T0) + 2 e0 x^g (T - T0)]
        S_molar = 3 * n * R_GAS * (np.log(T / self.T0) + 2 * e0 * x**g * (T - self.T0))
        S_molar += S0

        return S_molar / ATOMIC_MASS_FE
    
    def _internal_energy(self, V: float, T: float) -> float:
        """
        Calculate specific internal energy.
        
        E = E_cold(V) + E_th(V, T)
        
        For the Vinet EoS, the cold energy is obtained by integration:
        E_cold = ∫ P_cold dV from V0 to V
        
        The thermal energy is:
        E_th = 3nR [T + e0 x^g T^2]
        
        We compute the relative energy change from a reference state.
        
        Parameters
        ----------
        V : float
            Molar volume [m³/mol]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Specific internal energy [J/kg]
        """
        from scipy.integrate import quad
        
        U0 = self.params['U0']
        V0 = self.params['V0']
        KT0 = self.params['KT0']
        KT0_prime = self.params['KT0_prime']
        e0 = self.params['e0']
        g = self.params['g']
        n = self.params['n']
        
        # Cold energy from Vinet: E_cold = -∫ P_cold dV from V0 to V
        # The negative sign comes from dE = -P dV at constant T
        x = V / V0
        y = x**(1/3)
        eta = 1.5 * (KT0_prime - 1)
        
        if abs(V - V0) < 1e-15 * V0:
            E_cold = 0.0
        else:
            E_cold = 9 * KT0 * V0 * (1 - (1 - eta*(1 - y))*np.exp(eta*(1 - y))) / eta**2
        
        # Thermal energy at current state
        E_th = 3 * n * R_GAS * (T + e0 * x**g * T**2)
        
        # Thermal energy at reference state (V0, T0)
        E_th_ref = 3 * n * R_GAS * (self.T0 + e0 * self.T0**2)
        
        # Total molar energy relative to reference
        E_molar = U0 + E_cold + (E_th - E_th_ref)
        
        return E_molar / ATOMIC_MASS_FE
    
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
        return ATOMIC_MASS_FE / V
    
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
        return self._internal_energy(V, T)
    
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
        return self._entropy(V, T)
    
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
        Cv_molar = self._isochoric_heat_capacity(V, T)
        alpha = self._thermal_expansion_coeff(V, T)
        KT = self._isothermal_bulk_modulus(V, T)
        
        # Molar C_P
        Cp_molar = Cv_molar + alpha**2 * T * V * KT
        
        return Cp_molar / ATOMIC_MASS_FE
    
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
        return Cv_molar / ATOMIC_MASS_FE
    
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
            Thermal expansion coefficient [K^-1]
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


# =============================================================================
# Phase Diagram Functions
# =============================================================================
#
# Phase boundaries and determination functions following BICEPS model
# (Haldemann et al. 2024, A&A 681, A96), Section 2.3 Core layer model.
#
# Solid-solid transitions: Dorogokupets et al. (2017), Scientific Reports 7:41863
# Melting curve: Anzellini et al. (2013), Science 340:464
#
# Phases:
#   α-Fe (bcc): body-centered cubic, stable at low P and low/moderate T
#   δ-Fe (bcc): body-centered cubic, stable at low P and high T (near melting)
#   γ-Fe (fcc): face-centered cubic, stable at intermediate P and T
#   ε-Fe (hcp): hexagonal close-packed, stable at high P
#   liquid-Fe: above the melting curve
# =============================================================================


# Melting curve reference points (Anzellini et al. 2013)
_P0_MELT = 5.2e9      # Pa - Reference pressure
_T0_MELT = 1991.0     # K - Reference temperature
_PT_MELT = 98.5e9     # Pa - ε-γ-liquid triple point pressure
_TT_MELT = 3712.0     # K - ε-γ-liquid triple point temperature

# Critical pressure boundaries
_P_ALPHA_GAMMA_EPSILON = 7.3e9   # Pa - α-γ-ε triple point
_P_ALPHA_EPSILON_MAX = 15.8e9    # Pa - Upper limit of α-ε transition


def T_gamma_epsilon(P: float) -> float:
    """
    γ-ε (fcc-hcp) phase transition temperature.
    
    Equation (7) from BICEPS paper (Dorogokupets et al. 2017, Fig. 1):
    T_γε(P) = 575 + 18.7(P/GPa) + 0.213(P/GPa)^2 - 8.17×10^{-4}(P/GPa)^3
    
    Parameters
    ----------
    P : float
        Pressure [Pa]
        
    Returns
    -------
    float
        Transition temperature [K]
    """
    P_GPa = P / 1e9
    return 575.0 + 18.7 * P_GPa + 0.213 * P_GPa**2 - 8.17e-4 * P_GPa**3


def T_alpha_gamma(P: float) -> float:
    """
    α-γ (bcc-fcc) phase transition temperature.
    
    Equation (8) from BICEPS paper (Dorogokupets et al. 2017, Fig. 1):
    T_αγ(P) = 1120 + (820 - 1120)(P/7.3 GPa)
    
    Valid for P ≤ 7.3 GPa.
    
    Parameters
    ----------
    P : float
        Pressure [Pa]
        
    Returns
    -------
    float
        Transition temperature [K]
    """
    P_GPa = P / 1e9
    return 1120.0 + (820.0 - 1120.0) * (P_GPa / 7.3)


def T_delta_gamma(P: float) -> float:
    """
    δ-γ (bcc-fcc) phase transition temperature.
    
    Equation (9) from BICEPS paper (Dorogokupets et al. 2017, Fig. 1):
    T_δγ(P) = 1580 + (1998 - 1580)(P/5.2 GPa)
    
    This is the transition from δ-Fe (high-T bcc) to γ-Fe (fcc).
    Valid for P ≤ 5.2 GPa.
    
    Parameters
    ----------
    P : float
        Pressure [Pa]
        
    Returns
    -------
    float
        Transition temperature [K]
    """
    P_GPa = P / 1e9
    return 1580.0 + (1998.0 - 1580.0) * (P_GPa / 5.2)


def T_alpha_epsilon(P: float) -> float:
    """
    α-ε (bcc-hcp) phase transition temperature.
    
    Equation (10) from BICEPS paper (Dorogokupets et al. 2017, Fig. 1):
    T_αε(P) = 820 + (300 - 820)((P - 7.3 GPa)/(15.8 GPa - 7.3 GPa))
    
    Valid for 7.3 GPa ≤ P ≤ 15.8 GPa.
    
    Parameters
    ----------
    P : float
        Pressure [Pa]
        
    Returns
    -------
    float
        Transition temperature [K]
    """
    P_GPa = P / 1e9
    return 820.0 + (300.0 - 820.0) * ((P_GPa - 7.3) / (15.8 - 7.3))


def T_melt_Fe(P: float) -> float:
    """
    Melting temperature of pure iron.
    
    Equation (6) from BICEPS paper (Anzellini et al. 2013):
    
    For P < P_t:
        T_m(P) = T_0 × ((P - P_0)/(27.39 GPa) + 1)^(1/2.38)
        
    For P ≥ P_t:
        T_m(P) = T_t × ((P - P_t)/(161.2 GPa) + 1)^(1/1.72)
    
    where:
    - (P_0, T_0) = (5.2 GPa, 1991 K) is the reference point
    - (P_t, T_t) = (98.5 GPa, 3712 K) is the ε-γ-liquid triple point
    
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
    P0_GPa = _P0_MELT / 1e9
    Pt_GPa = _PT_MELT / 1e9
    
    if P < _PT_MELT:
        return _T0_MELT * ((P_GPa - P0_GPa) / 27.39 + 1)**(1/2.38)
    else:
        return _TT_MELT * ((P_GPa - Pt_GPa) / 161.2 + 1)**(1/1.72)


def get_iron_phase(P: float, T: float) -> str:
    """
    Determine the stable phase of pure iron at given P and T.
    
    This function implements the phase diagram described in BICEPS 
    (Haldemann et al. 2024), Section 2.3 Core layer model, using:
    - Solid-solid phase boundaries from Dorogokupets et al. (2017)
    - Melting curve from Anzellini et al. (2013)
    
    Parameters
    ----------
    P : float
        Pressure [Pa]
    T : float
        Temperature [K]
        
    Returns
    -------
    str
        Phase identifier: 'solid-alpha', 'solid-delta', 'solid-gamma',
        'solid-epsilon', or 'liquid'
    
    Examples
    --------
    >>> get_iron_phase(1e5, 300)      # Ambient conditions
    'solid-alpha'
    >>> get_iron_phase(1e5, 1700)     # High T, low P
    'solid-delta'
    >>> get_iron_phase(50e9, 2000)    # Moderate P and T
    'solid-gamma'
    >>> get_iron_phase(200e9, 4000)   # High P
    'solid-epsilon'
    >>> get_iron_phase(200e9, 6000)   # Above melting
    'liquid'
    """
    # Check if liquid
    T_melt = T_melt_Fe(P)
    if T >= T_melt:
        return 'liquid'
    
    P_GPa = P / 1e9
    
    # High pressure: only ε-Fe below melting (above γ-ε-liquid triple point)
    if P >= _PT_MELT:
        return 'solid-epsilon'
    
    # Calculate γ-ε boundary
    T_ge = T_gamma_epsilon(P)
    
    # Above α-ε transition range (P > 15.8 GPa)
    if P_GPa > 15.8:
        if T < T_ge:
            return 'solid-epsilon'
        else:
            return 'solid-gamma'
    
    # Intermediate pressure (7.3 GPa ≤ P ≤ 15.8 GPa)
    if P_GPa >= 7.3:
        T_ae = T_alpha_epsilon(P)
        if T < T_ae:
            return 'solid-alpha'
        elif T < T_ge:
            return 'solid-epsilon'
        else:
            return 'solid-gamma'
    
    # Low pressure (P < 7.3 GPa): α, γ, δ phases
    T_ag = T_alpha_gamma(P)
    
    if T < T_ag:
        return 'solid-alpha'
    
    # Check for δ phase (high T bcc, only at very low P)
    if P_GPa <= 5.2:
        T_dg = T_delta_gamma(P)
        if T >= T_dg:
            return 'solid-delta'
    
    # Otherwise γ phase
    return 'solid-gamma'


def get_iron_eos(phase: str):
    """
    Return the appropriate EoS instance for a given iron phase.
    
    This function returns an EoS instance configured for the specified
    iron phase:
    
    - α-Fe, δ-Fe (bcc): Dorogokupets et al. (2017), phase='bcc'
    - γ-Fe (fcc): Dorogokupets et al. (2017), phase='fcc'
    - ε-Fe (hcp): Miozzi et al. (2020)/Hakim et al. (2018)
    - liquid-Fe: Luo et al. (2024)
    
    Parameters
    ----------
    phase : str
        Phase identifier: 'solid-alpha', 'solid-delta', 'solid-gamma',
        'solid-epsilon', or 'liquid'
        
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
    >>> eos = get_iron_eos('solid-alpha')
    >>> rho = eos.density(1e9, 500)
    
    Notes
    -----
    For ε-Fe, returns HcpIronEos which smoothly blends Miozzi20 and
    Hakim18 across the transition zone around 310 GPa.
    """
    phase_lower = phase.lower()
    
    if phase_lower in ['solid-alpha', 'solid-delta']:
        return Dorogokupets17(phase='bcc')
    elif phase_lower == 'solid-gamma':
        return Dorogokupets17(phase='fcc')
    elif phase_lower == 'solid-epsilon':
        return HcpIronEos()
    elif phase_lower == 'liquid':
        return Luo24()
    else:
        raise ValueError(
            f"Unknown phase '{phase}'. "
            f"Valid options: 'solid-alpha', 'solid-delta', 'solid-gamma', "
            f"'solid-epsilon', 'liquid'"
        )


def get_iron_eos_for_PT(P: float, T: float):
    """
    Return the appropriate EoS instance for given P-T conditions.
    
    Combines phase determination and EoS selection. For ε-Fe (hcp), 
    uses HcpIronEos which smoothly blends Miozzi20 and Hakim18
    across the transition zone around 310 GPa.
    
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
    >>> eos, phase = get_iron_eos_for_PT(100e9, 3000)
    >>> rho = eos.density(100e9, 3000)
    
    >>> # Super-Earth core conditions
    >>> eos, phase = get_iron_eos_for_PT(500e9, 5000)
    >>> print(f"Using {type(eos).__name__} for {phase}")
    """
    phase = get_iron_phase(P, T)
    
    if phase in ['solid-alpha', 'solid-delta']:
        return Dorogokupets17(phase='bcc'), phase
    elif phase == 'solid-gamma':
        return Dorogokupets17(phase='fcc'), phase
    elif phase == 'solid-epsilon':
        return HcpIronEos(), phase
    elif phase == 'liquid':
        return Luo24(), phase
    else:
        raise RuntimeError(f"Unexpected phase: {phase}")


# =============================================================================
# Wrapper Class
# =============================================================================


class IronEoS:
    """
    Wrapper equation of state for iron with pre-instantiated phase classes.

    This class instantiates every individual iron phase EoS class once at
    initialization and selects the appropriate one at each (P, T) query
    based on the iron phase diagram. It avoids the overhead of repeated
    class construction that ``get_iron_eos_for_PT`` incurs, making it
    suitable for tight loops such as interior structure ODE integration.

    The seven standard PALEOS thermodynamic properties are exposed as
    public methods, together with a ``phase`` method that returns the
    stable phase label at the queried point.

    Attributes
    ----------
    _eos_bcc : Dorogokupets17
        EoS instance for alpha-Fe and delta-Fe (bcc)
    _eos_fcc : Dorogokupets17
        EoS instance for gamma-Fe (fcc)
    _eos_hcp : HcpIronEos
        Composite EoS instance for epsilon-Fe (hcp)
    _eos_liquid : Luo24
        EoS instance for liquid iron

    Examples
    --------
    >>> eos = IronEoS()
    >>> rho = eos.density(200e9, 4000)
    >>> phase = eos.phase(200e9, 4000)
    >>> print(f"{phase}: rho = {rho:.1f} kg/m^3")
    """

    def __init__(self):
        """
        Initialize IronEoS by pre-instantiating all phase EoS classes.
        """
        self._eos_bcc = Dorogokupets17(phase='bcc')
        self._eos_fcc = Dorogokupets17(phase='fcc')
        self._eos_hcp = HcpIronEos()
        self._eos_liquid = Luo24()

        self._phase_eos_map = {
            'solid-alpha':   self._eos_bcc,
            'solid-delta':   self._eos_bcc,
            'solid-gamma':   self._eos_fcc,
            'solid-epsilon': self._eos_hcp,
            'liquid':        self._eos_liquid,
        }

    def _get_eos(self, P, T):
        """Return the (eos_instance, phase_label) for given P-T conditions."""
        phase = get_iron_phase(P, T)
        return self._phase_eos_map[phase], phase

    def phase(self, P, T):
        """
        Return the stable iron phase at given P and T.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        str
            Phase identifier: 'solid-alpha', 'solid-delta', 'solid-gamma',
            'solid-epsilon', or 'liquid'
        """
        return get_iron_phase(P, T)

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
