"""
PALEOS Equations of State for H₂O

This module contains the implementation of the AQUA equation of state for
water (Haldemann et al. 2020) relevant to planetary interiors, covering the
full pressure-temperature range from 0.1 Pa to 400 TPa and 150 to 10⁵ K.

The AQUA EoS is a composite, tabulated description of H₂O assembled from
seven underlying equations of state covering all major thermodynamic phases:
ice-Ih, ice-II/III/V/VI, ice-VII/X, liquid, vapor, supercritical fluid,
and superionic water. The underlying EoS come from Feistel & Wagner (2006),
Journaux et al. (2020), French & Redmer (2015), Wagner & Pruß (2002),
Brown (2018), Gordon & McBride (1994), and Mazevet et al. (2019).

Entropy / energy correction
----------------------------
The Mazevet et al. (2019) Helmholtz free energy parametrization used in
AQUA Region 7 contains a sign error in the published Eq. (13).  The
corrected expression for F_T differs from the erroneous one by

    F_{T,shift} = 2 N_at [b₁ τ ln(1 + τ⁻²) + b₂ τ arctan τ]

where τ = T / T_crit (T_crit = 647 K), b₁ = 3 × 10^{-20} J, and
b₂ = 1.35 × 10^{-20} J. Since F_{T,shift} is independent of density, it
does not affect the pressure (and therefore density) but shifts the
specific entropy and specific internal energy via the standard
thermodynamic relations S = −∂F/∂T|_V and U = F + TS.

This correction is applied on top of the tabulated AQUA values so that
the returned entropy and internal energy are self-consistent with the
corrected Helmholtz free energy.

EoS Class
---------
- Haldemann20: Tabulated AQUA EoS for H₂O (Haldemann et al. 2020)
               with the Mazevet et al. (2019) entropy/energy correction

Phase Determination
-------------------
- get_water_phase(P, T, table_path): Returns the stable H₂O phase from
    the AQUA phase map
- get_water_eos(phase, table_path): Returns Haldemann20 instance
- get_water_eos_for_PT(P, T, table_path): Returns (EoS instance, phase)

AQUA phase identifiers (mapped to PALEOS string labels):
    -1  → 'solid-ice-Ih'
    -2  → 'solid-ice-II'
    -3  → 'solid-ice-III'
    -5  → 'solid-ice-V'
    -6  → 'solid-ice-VI'
    -7  → 'solid-ice-VII'
    -10 → 'solid-ice-X'
     3  → 'vapor'
     4  → 'liquid'
     5  → 'supercritical'

References
----------
Haldemann, J., Alibert, Y., Mordasini, C., Benz, W. (2020)
    "AQUA: a collection of H₂O equations of state for planetary models"
    A&A 643, A105, DOI: 10.1051/0004-6361/202038367

Mazevet, S., Licari, A., Chabrier, G., Potekhin, A. Y. (2019)
    "Ab initio based equation of state of dense water for planetary
    and exoplanetary modeling"
    A&A 621, A128, DOI: 10.1051/0004-6361/201833963

Author: Mara Attia
Date: March 2026
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Physical constants
R_GAS = 8.314462618  # J/(mol·K)
N_AVOGADRO = 6.02214076e23  # mol⁻¹
K_BOLTZMANN = 1.380649e-23  # J/K

# H₂O molecular properties
_M_H2O = 18.015e-3  # kg/mol — molar mass of water
_N_ATOMS_PER_MOL = 3  # atoms per molecule (2H + 1O)

# Mazevet et al. (2019) correction constants
_T_CRIT = 647.0  # K — critical temperature used in the parametrization
_b1 = 3.0e-20  # J — (3 × 10⁻¹³ erg converted to SI)
_b2 = 1.35e-20  # J — (1.35 × 10⁻¹³ erg converted to SI)

# Number of atoms per kg of H₂O
_N_AT_PER_KG = _N_ATOMS_PER_MOL * N_AVOGADRO / _M_H2O  # atoms/kg

# AQUA phase code → PALEOS label mapping
_PHASE_MAP = {
    -1:  'solid-ice-Ih',
    -2:  'solid-ice-II',
    -3:  'solid-ice-III',
    -5:  'solid-ice-V',
    -6:  'solid-ice-VI',
    -7:  'solid-ice-VII',
    -10: 'solid-ice-X',
    3:   'vapor',
    4:   'liquid',
    5:   'supercritical',
}

# Reverse map for convenience
_PHASE_MAP_INV = {v: k for k, v in _PHASE_MAP.items()}


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


class Haldemann20:
    """
    Tabulated equation of state for H₂O from Haldemann et al. (2020).

    Reference:
    Haldemann, J., Alibert, Y., Mordasini, C., Benz, W. (2020)
    "AQUA: a collection of H₂O equations of state for planetary models"
    A&A 643, A105, DOI: 10.1051/0004-6361/202038367

    This implementation loads the AQUA P–T table and provides the standard
    PALEOS thermodynamic interface via bilinear interpolation in
    log₁₀(P)–log₁₀(T) space.

    A correction to the Mazevet et al. (2019) free energy F_T is applied
    to the specific entropy and specific internal energy.  The correction
    arises from a sign error in the published Eq. (13) of Mazevet et al.
    (2019): the first two terms inside the brackets carry a minus sign in
    the paper but should be positive.  The difference,

        F_{T,shift}(T) = 2 N_at [b₁ τ ln(1+τ⁻²) + b₂ τ arctan(τ)]

    where N_at = 3 N_A / M_{H₂O} is the number of atoms per kg, τ = T/647 K,
    b₁ = 3 × 10⁻²⁰ J, b₂ = 1.35 × 10⁻²⁰ J, is propagated analytically
    to entropy (S_shift = −∂F_{T,shift}/∂T) and internal energy
    (U_shift = F_{T,shift} + T S_shift).

    Since F_{T,shift} is independent of density, the pressure and density
    are unaffected by this correction.

    Heat capacities and thermal expansion are derived from the tabulated
    quantities through standard thermodynamic relations:
        α  = −(1/ρ)(∂ρ/∂T)_P          (numerical derivative)
        C_P = α P / (ρ ∇_ad)          (from the adiabatic gradient)
        C_V = C_P² / (C_P + T α² w²)  (from the speed of sound)

    Parameters
    ----------
    table_path : str
        Path to the AQUA P–T table file (whitespace-delimited, with the
        standard CDS header).

    Attributes
    ----------
    log10_P_grid : ndarray
        Unique log₁₀(pressure/Pa) values on the grid
    log10_T_grid : ndarray
        Unique log₁₀(temperature/K) values on the grid
    n_P : int
        Number of pressure grid points
    n_T : int
        Number of temperature grid points

    Examples
    --------
    >>> eos = Haldemann20('path/to/AQUA_PT_table.dat')
    >>> rho = eos.density(50e9, 2000)
    >>> s = eos.specific_entropy(50e9, 2000)
    >>> cp = eos.isobaric_heat_capacity(50e9, 2000)

    Notes
    -----
    The table grid spans 0.1 Pa to 400 TPa in pressure (1093 points,
    70 per decade, log-spaced) and 100 K to 10⁵ K in temperature
    (301 points, 100 per decade, log-spaced).

    Extrapolation beyond the table boundaries is not permitted; the
    interpolators will raise ValueError for out-of-range queries.
    """

    def __init__(self, table_path: str):
        """
        Initialize the Haldemann20 EoS by loading the AQUA P–T table.

        Parameters
        ----------
        table_path : str
            Path to the AQUA P–T table file.

        Raises
        ------
        FileNotFoundError
            If the table file does not exist.
        ValueError
            If the table cannot be parsed or has unexpected shape.
        """
        self._load_table(table_path)
        self._build_interpolators()

    # =========================================================================
    # Table loading and interpolator construction
    # =========================================================================

    def _load_table(self, path: str):
        """
        Parse the AQUA P–T table into structured arrays.

        The file has a multi-line header (lines starting with non-numeric
        characters) followed by whitespace-delimited data rows with columns:
            P  T  ρ  ∇_ad  s  u  w  μ  x_ion  x_d  phase

        Parameters
        ----------
        path : str
            Path to the table file.
        """
        # Read all numeric rows
        data = np.loadtxt(path, skiprows=19, dtype=float)

        # If the generic comment approach fails, fall back to skipping
        # header lines manually.
        if data.ndim != 2 or data.shape[1] < 11:
            # Try reading by skipping non-numeric header lines
            rows = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # A data line starts with a digit or a sign
                    if line[0].isdigit() or line[0] in '+-':
                        try:
                            vals = [float(x) for x in line.split()]
                            if len(vals) >= 11:
                                rows.append(vals[:11])
                        except ValueError:
                            continue
            data = np.array(rows)

        if data.shape[1] < 11:
            raise ValueError(
                f"Expected at least 11 columns, got {data.shape[1]}"
            )

        # Extract columns
        P_all = data[:, 0]   # Pa
        T_all = data[:, 1]   # K
        rho_all = data[:, 2] # kg/m³
        nad_all = data[:, 3] # dimensionless — adiabatic gradient
        s_all = data[:, 4]   # J/(kg·K) — specific entropy
        u_all = data[:, 5]   # J/kg — specific internal energy
        w_all = data[:, 6]   # m/s — speed of sound
        # columns 7-9: μ, x_ion, x_d (not used in the public interface)
        phase_all = data[:, 10].astype(int)

        # Determine the unique grid axes
        P_unique = np.unique(P_all)
        T_unique = np.unique(T_all)

        self.n_P = len(P_unique)
        self.n_T = len(T_unique)
        self.log10_P_grid = np.log10(P_unique)
        self.log10_T_grid = np.log10(T_unique)

        expected_len = self.n_P * self.n_T
        if len(P_all) != expected_len:
            raise ValueError(
                f"Table has {len(P_all)} rows but grid is "
                f"{self.n_P} × {self.n_T} = {expected_len}"
            )

        # Reshape into 2-D arrays indexed as [i_P, i_T]
        # The table is ordered with P as the slow index and T as the fast index.
        self._rho_grid = rho_all.reshape(self.n_P, self.n_T)
        self._nad_grid = nad_all.reshape(self.n_P, self.n_T)
        self._s_grid = s_all.reshape(self.n_P, self.n_T)
        self._u_grid = u_all.reshape(self.n_P, self.n_T)
        self._w_grid = w_all.reshape(self.n_P, self.n_T)
        self._phase_grid = phase_all.reshape(self.n_P, self.n_T)

    def _build_interpolators(self):
        """
        Build RegularGridInterpolator objects on the log₁₀(P)–log₁₀(T) grid.

        Interpolation is bilinear in log-space, which is natural for the
        log-spaced AQUA grid and gives smooth behaviour across the many
        decades spanned by the table.
        """
        lP = self.log10_P_grid
        lT = self.log10_T_grid

        kw = dict(method='linear', bounds_error=True)

        # Primary table quantities
        self._interp_log_rho = RegularGridInterpolator(
            (lP, lT), np.log10(self._rho_grid), **kw
        )
        self._interp_nad = RegularGridInterpolator(
            (lP, lT), self._nad_grid, **kw
        )
        self._interp_s = RegularGridInterpolator(
            (lP, lT), self._s_grid, **kw
        )
        self._interp_u = RegularGridInterpolator(
            (lP, lT), self._u_grid, **kw
        )
        self._interp_log_w = RegularGridInterpolator(
            (lP, lT), np.log10(np.clip(self._w_grid, 1e-30, None)), **kw
        )

        # Phase grid (nearest-neighbour for integer-valued field)
        self._interp_phase = RegularGridInterpolator(
            (lP, lT), self._phase_grid.astype(float),
            method='nearest', bounds_error=True
        )

    # =========================================================================
    # Raw table interpolation (before corrections)
    # =========================================================================

    def _raw_density(self, P: float, T: float) -> float:
        """Interpolate density from the AQUA table [kg/m³]."""
        pt = np.array([[np.log10(P), np.log10(T)]])
        return 10.0**float(self._interp_log_rho(pt)[0])

    def _raw_adiabatic_gradient(self, P: float, T: float) -> float:
        """Interpolate adiabatic gradient from the AQUA table [dimensionless]."""
        pt = np.array([[np.log10(P), np.log10(T)]])
        return float(self._interp_nad(pt)[0])

    def _raw_entropy(self, P: float, T: float) -> float:
        """Interpolate specific entropy from the AQUA table [J/(kg·K)]."""
        pt = np.array([[np.log10(P), np.log10(T)]])
        return float(self._interp_s(pt)[0])

    def _raw_internal_energy(self, P: float, T: float) -> float:
        """Interpolate specific internal energy from the AQUA table [J/kg]."""
        pt = np.array([[np.log10(P), np.log10(T)]])
        return float(self._interp_u(pt)[0])

    def _raw_speed_of_sound(self, P: float, T: float) -> float:
        """Interpolate speed of sound from the AQUA table [m/s]."""
        pt = np.array([[np.log10(P), np.log10(T)]])
        return 10.0**float(self._interp_log_w(pt)[0])

    def _raw_phase_id(self, P: float, T: float) -> int:
        """Interpolate (nearest-neighbour) the AQUA phase ID."""
        pt = np.array([[np.log10(P), np.log10(T)]])
        return int(round(float(self._interp_phase(pt)[0])))

    # =========================================================================
    # Mazevet et al. (2019) F_T correction
    # =========================================================================

    @staticmethod
    def _f_shift(T: float) -> float:
        """
        Specific Helmholtz free energy correction F_{T,shift} / mass.

        F_{T,shift} = 2 N_at [b₁ τ ln(1 + τ⁻²) + b₂ τ arctan(τ)]

        where N_at = 3 N_A / M_{H₂O}, τ = T / 647 K.

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific free energy shift [J/kg]
        """
        tau = T / _T_CRIT
        term1 = _b1 * tau * np.log(1.0 + tau**(-2))
        term2 = _b2 * tau * np.arctan(tau)
        return 2.0 * _N_AT_PER_KG * (term1 + term2)

    @staticmethod
    def _df_shift_dT(T: float) -> float:
        """
        Temperature derivative of the specific free energy correction.

        ∂F_{T,shift}/∂T = 2 N_at [
            b₁ ( (1/T_c) ln(1 + T_c²/T²)  −  2 T_c / (T² + T_c²) )
          + b₂ ( (1/T_c) arctan(T/T_c)     +  T / (T² + T_c²) )
        ]

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            ∂F_{T,shift}/∂T [J/(kg·K)]
        """
        tau = T / _T_CRIT
        Tc = _T_CRIT
        denom = T**2 + Tc**2

        d_term1 = _b1 * ((1.0 / Tc) * np.log(1.0 + Tc**2 / T**2)
                         - 2.0 * Tc / denom)
        d_term2 = _b2 * ((1.0 / Tc) * np.arctan(T / Tc)
                         + T / denom)
        return 2.0 * _N_AT_PER_KG * (d_term1 + d_term2)

    @staticmethod
    def _entropy_shift(T: float) -> float:
        """
        Specific entropy correction.

        S_shift = −∂F_{T,shift}/∂T

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific entropy shift [J/(kg·K)]
        """
        return -Haldemann20._df_shift_dT(T)

    @staticmethod
    def _energy_shift(T: float) -> float:
        """
        Specific internal energy correction.

        U_shift = F_{T,shift} + T × S_shift
                = F_{T,shift} − T × ∂F_{T,shift}/∂T

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific internal energy shift [J/kg]
        """
        return (Haldemann20._f_shift(T)
                - T * Haldemann20._df_shift_dT(T))

    # =========================================================================
    # Derived quantities from table thermodynamic relations
    # =========================================================================

    def _thermal_expansion_numerical(self, P: float, T: float) -> float:
        """
        Volumetric thermal expansion coefficient via numerical differentiation.

        α = −(1/ρ)(∂ρ/∂T)_P

        computed as a centred finite difference in log₁₀(T) space,
        with one-sided stencils at the grid boundaries.

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
        # Step size: ~0.2% of T (small but well above interpolation noise)
        dT = T * 2e-3
        T_lo = T - dT
        T_hi = T + dT

        # Clamp to grid boundaries
        T_min = 10.0**self.log10_T_grid[0]
        T_max = 10.0**self.log10_T_grid[-1]

        if T_lo < T_min:
            # Forward difference
            rho0 = self._raw_density(P, T)
            rho1 = self._raw_density(P, T + dT)
            drho_dT = (rho1 - rho0) / dT
        elif T_hi > T_max:
            # Backward difference
            rho0 = self._raw_density(P, T - dT)
            rho1 = self._raw_density(P, T)
            drho_dT = (rho1 - rho0) / dT
        else:
            # Central difference
            rho_lo = self._raw_density(P, T_lo)
            rho_hi = self._raw_density(P, T_hi)
            drho_dT = (rho_hi - rho_lo) / (2.0 * dT)

        rho = self._raw_density(P, T)
        if abs(rho) < 1e-30:
            return 0.0
        return -drho_dT / rho

    # =========================================================================
    # Public interface — standard PALEOS thermodynamic properties
    # =========================================================================

    def density(self, P: float, T: float) -> float:
        """
        Calculate density.

        Interpolated directly from the AQUA table (unaffected by the
        Mazevet correction, which is volume-independent).

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
        return self._raw_density(P, T)

    def specific_internal_energy(self, P: float, T: float) -> float:
        """
        Calculate specific internal energy (corrected).

        U(P,T) = U_AQUA(P,T) + U_shift(T)

        The shift corrects for the sign error in Mazevet et al. (2019) Eq. 13.

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
        return self._raw_internal_energy(P, T) + self._energy_shift(T)

    def specific_entropy(self, P: float, T: float) -> float:
        """
        Calculate specific entropy (corrected).

        S(P,T) = S_AQUA(P,T) + S_shift(T)

        The shift corrects for the sign error in Mazevet et al. (2019) Eq. 13.

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
        return self._raw_entropy(P, T) + self._entropy_shift(T)

    def isobaric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isobaric heat capacity.

        Derived from the AQUA table via:
            C_P = α P / (ρ ∇_ad)

        where α is the thermal expansion coefficient obtained by numerical
        differentiation of the density, ρ is the table density, and ∇_ad
        is the table adiabatic gradient.

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
        alpha = self._thermal_expansion_numerical(P, T)
        rho = self._raw_density(P, T)
        nad = self._raw_adiabatic_gradient(P, T)

        if abs(nad) < 1e-30 or abs(rho) < 1e-30:
            return np.nan
        return alpha * P / (rho * nad)

    def isochoric_heat_capacity(self, P: float, T: float) -> float:
        """
        Calculate specific isochoric heat capacity.

        Derived from the thermodynamic identity:
            C_V = C_P² / (C_P + T α² w²)

        where w is the speed of sound from the AQUA table.

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
        cp = self.isobaric_heat_capacity(P, T)
        alpha = self._thermal_expansion_numerical(P, T)
        w = self._raw_speed_of_sound(P, T)

        denom = cp + T * alpha**2 * w**2
        if abs(denom) < 1e-30:
            return np.nan
        return cp**2 / denom

    def thermal_expansion(self, P: float, T: float) -> float:
        """
        Calculate volumetric thermal expansion coefficient.

        Computed as α = −(1/ρ)(∂ρ/∂T)_P using centred finite differences
        on the interpolated density field.

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
        return self._thermal_expansion_numerical(P, T)

    def adiabatic_gradient(self, P: float, T: float) -> float:
        """
        Calculate dimensionless adiabatic temperature gradient.

        ∇_ad = (∂ ln T / ∂ ln P)_S

        Interpolated directly from the AQUA table.

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
        return self._raw_adiabatic_gradient(P, T)

    # =========================================================================
    # Additional diagnostics
    # =========================================================================

    def speed_of_sound(self, P: float, T: float) -> float:
        """
        Bulk speed of sound from the AQUA table.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Speed of sound [m/s]
        """
        return self._raw_speed_of_sound(P, T)

    def phase(self, P: float, T: float) -> str:
        """
        Return the PALEOS phase label at given (P, T).

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        str
            Phase label (e.g. 'liquid', 'solid-ice-VII', 'supercritical', …)
        """
        pid = self._raw_phase_id(P, T)
        return _PHASE_MAP.get(pid, f'unknown({pid})')

    def adiabatic_gradient_from_corrected_entropy(
        self, P: float, T: float
    ) -> float:
        """
        Adiabatic gradient computed from the *corrected* specific entropy.

        ∇_ad = α P / (ρ C_P^corr)

        where C_P^corr = T (∂s_corr/∂T)_P is obtained by numerical
        differentiation of the corrected entropy.

        This function is intended as a sanity check: it should agree with
        the tabulated ∇_ad in regions where the Mazevet correction is
        negligible and deviate where the correction matters (Region 7 of
        AQUA).

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Adiabatic gradient from corrected entropy (dimensionless)
        """
        # Numerical derivative of corrected entropy w.r.t. T at constant P
        dT = T * 2e-3
        T_min = 10.0**self.log10_T_grid[0]
        T_max = 10.0**self.log10_T_grid[-1]

        T_lo = max(T - dT, T_min)
        T_hi = min(T + dT, T_max)
        actual_dT = T_hi - T_lo

        if actual_dT < 1e-10:
            return np.nan

        s_lo = self.specific_entropy(P, T_lo)
        s_hi = self.specific_entropy(P, T_hi)
        ds_dT = (s_hi - s_lo) / actual_dT

        cp_corr = T * ds_dT
        if abs(cp_corr) < 1e-30:
            return np.nan

        alpha = self._thermal_expansion_numerical(P, T)
        rho = self._raw_density(P, T)

        return alpha * P / (rho * cp_corr)


# =============================================================================
# Phase Determination Functions
# =============================================================================
#
# These functions provide a thin wrapper around the AQUA phase map stored
# in the tabulated EoS.  Unlike the iron and MgSiO₃ modules where phase
# boundaries are given as analytical functions, the H₂O phase diagram in
# AQUA is encoded implicitly in the per-grid-point phase identifier.
#
# For convenience and consistency with the PALEOS API, the functions
# accept a table_path (or a pre-loaded Haldemann20 instance) and query
# the nearest-neighbour phase from the grid.
# =============================================================================

# Module-level cache for the default table
_DEFAULT_EOS_CACHE = {}


def _get_cached_eos(table_path: str) -> Haldemann20:
    """Return a cached Haldemann20 instance for the given table path."""
    if table_path not in _DEFAULT_EOS_CACHE:
        _DEFAULT_EOS_CACHE[table_path] = Haldemann20(table_path)
    return _DEFAULT_EOS_CACHE[table_path]


def get_water_phase(P: float, T: float,
                    table_path: str = None,
                    eos: Haldemann20 = None) -> str:
    """
    Determine the stable H₂O phase at given P and T.

    The phase is looked up from the AQUA table's nearest-neighbour
    phase identifier grid.  The returned label follows the PALEOS
    convention.

    Parameters
    ----------
    P : float
        Pressure [Pa]
    T : float
        Temperature [K]
    table_path : str, optional
        Path to the AQUA P–T table.  Ignored if ``eos`` is given.
    eos : Haldemann20, optional
        Pre-loaded EoS instance.  Takes precedence over ``table_path``.

    Returns
    -------
    str
        Phase label: 'solid-ice-Ih', 'solid-ice-II', 'solid-ice-III', 
        'solid-ice-V', 'solid-ice-VI', 'solid-ice-VII', 'solid-ice-X', 
        'vapor', 'liquid', or 'supercritical'

    Raises
    ------
    ValueError
        If neither ``table_path`` nor ``eos`` is provided.

    Examples
    --------
    >>> phase = get_water_phase(1e5, 300, table_path='AQUA_PT.dat')
    >>> print(phase)
    'liquid'
    """
    if eos is None:
        if table_path is None:
            raise ValueError("Provide either table_path or eos")
        eos = _get_cached_eos(table_path)
    return eos.phase(P, T)


def get_water_eos(phase: str = None,
                  table_path: str = None,
                  eos: Haldemann20 = None) -> Haldemann20:
    """
    Return the H₂O EoS instance.

    Since all H₂O phases are covered by the single AQUA table, the
    ``phase`` argument is accepted for API consistency but does not
    alter the returned instance.

    Parameters
    ----------
    phase : str, optional
        Phase identifier (accepted but not used for selection).
    table_path : str, optional
        Path to the AQUA P–T table.  Ignored if ``eos`` is given.
    eos : Haldemann20, optional
        Pre-loaded EoS instance.

    Returns
    -------
    Haldemann20
        EoS instance covering all H₂O phases.
    """
    if eos is not None:
        return eos
    if table_path is None:
        raise ValueError("Provide either table_path or eos")
    return _get_cached_eos(table_path)


def get_water_eos_for_PT(P: float, T: float,
                         table_path: str = None,
                         eos: Haldemann20 = None):
    """
    Return the H₂O EoS instance and phase for given P–T conditions.

    Combines phase determination and EoS selection, consistent with the
    PALEOS API used by ``get_iron_eos_for_PT`` and
    ``get_mgsio3_eos_for_PT``.

    Parameters
    ----------
    P : float
        Pressure [Pa]
    T : float
        Temperature [K]
    table_path : str, optional
        Path to the AQUA P–T table.
    eos : Haldemann20, optional
        Pre-loaded EoS instance.

    Returns
    -------
    tuple
        (eos_instance, phase_name)

    Examples
    --------
    >>> eos, phase = get_water_eos_for_PT(50e9, 2000,
    ...                                    table_path='AQUA_PT.dat')
    >>> rho = eos.density(50e9, 2000)
    >>> print(f"{phase}: ρ = {rho:.1f} kg/m³")
    """
    if eos is None:
        if table_path is None:
            raise ValueError("Provide either table_path or eos")
        eos = _get_cached_eos(table_path)
    phase = eos.phase(P, T)
    return eos, phase
