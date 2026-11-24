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

Entropy correction
------------------
The Mazevet et al. (2019) free energy fit used in AQUA Region 7 was corrected after AQUA was built (Fortran implementation of the fit, version 18.08.21, http://www.ioffe.ru/astro/H2O/). That correction changes the free energy F of the fit and leaves its internal energy, pressure, and heat capacity unchanged. The AQUA density, adiabatic gradient, and internal energy therefore need no correction, and only the AQUA entropy, s = (u - f) / T, does. The free energy changes by F_shift = F_{sign} + F_{S₀}:

1. **Sign error in Eq. (13).**  The corrected expression for F_T differs from the published one by

       F_{sign}(T) = 2 N_at [b₁ τ ln(1 + τ⁻²) + b₂ τ arctan τ]

   where τ = T / T_crit, with T_crit = 0.00205 Ha (647.34 K), b₁ = 0.0069 Ha, b₂ = 0.0031 Ha, and N_at per kilogram at 18 g/mol, the constants of the code.

2. **Reference entropy.**  The corrected code raises the reference entropy from S₀,old = 4.9 kB n_at to S₀,new = 9.8 kB n_at. Since the free energy contains a −S₀ T term, the AQUA table (built with S₀,old) needs

       F_{S₀}(T) = (S₀,old − S₀,new) T = −4.9 kB n_at T

With u fixed, the entropy shifts by S_shift = −F_shift / T. Between 300 and 2000 K this shift rises by about 1.2 × 10⁴ J/(kg·K), which restores (∂s/∂T)_P = C_P / T > 0 in the cold, dense part of Region 7 that AQUA labels ice X. In pure Region 7 the corrected entropy equals that of the corrected code plus the −500 J/(kg·K) offset AQUA gives Region 7 (Haldemann et al. 2020, Table 3). The shift is gated by a Region-7 weight ``w_7(P, T)`` (see ``Haldemann20._phase_aware_weight``), so it applies only where AQUA uses the fit: fully inside Region 7 and with a ramp across the 3↔7, 5↔7, and 6↔7 transition bands of Haldemann et al. (2020).

Below about 1000 K in Region 7, the corrected entropy can be negative (about −3600 J/(kg·K) at 1 TPa and 300 K). The heat capacity of the fit there is about twice the Dulong-Petit value, so its absolute entropy breaks the third law, and only entropy differences within Region 7 are meaningful. Defects of AQUA itself remain: below 300 K above 316 GPa, AQUA holds the fit at its 300 K values, so s and ρ do not change with T there; across the blend bands, s does not follow the Maxwell relation (∂s/∂P)_T = −α/ρ; in the 5↔7 band, whose bounds move with T, C_P differs from T (∂s/∂T)_P by about 20%, and by up to a factor of 2.5 near 4500 K, where the bounds stop moving; and between 16 and 27 GPa near 850 K, s falls on heating by up to about 1600 J/(kg·K).

EoS Classes
-----------
- Haldemann20: Tabulated AQUA EoS for H₂O (Haldemann et al. 2020)
               with the Mazevet et al. (2019) entropy correction
- WaterEoS: Wrapper class with pre-loaded table for efficient repeated
            evaluation with automatic phase identification

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

Author: Mara Attia
Date: March 2026
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Physical constants
R_GAS = 8.314462618  # J/(mol·K)
N_AVOGADRO = 6.02214076e23  # mol⁻¹
K_BOLTZMANN = 1.380649e-23  # J/K

# Mazevet et al. (2019) correction constants, in the units of their code (version 18.08.21): PC1, PC2, and TCRIT in Hartree, 315774.6 K per Hartree, 18 g/mol per molecule
_HARTREE_IN_K = 315774.6
_T_CRIT = 0.00205 * _HARTREE_IN_K  # K, 647.34
_b1 = 0.0069 * _HARTREE_IN_K * K_BOLTZMANN  # J
_b2 = 0.0031 * _HARTREE_IN_K * K_BOLTZMANN  # J
_S0_OLD_PER_ATOM = 4.9 * K_BOLTZMANN  # J/(K atom), published fit
_S0_NEW_PER_ATOM = 9.8 * K_BOLTZMANN  # J/(K atom), corrected fit (SBASE)
_M_H2O = 18.0e-3  # kg/mol
_N_ATOMS_PER_MOL = 3
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

    The corrected Mazevet et al. (2019) free energy (version 18.08.21 of its Fortran implementation) differs from the one AQUA was built with by two density-independent terms. The correction changes F alone, so the AQUA density, adiabatic gradient, and internal energy stand, and the specific entropy shifts by S_shift = −F_shift / T, gated by a Region-7 weight ``w_7(P, T)`` that is non-zero only where AQUA uses the M19 EoS (see ``_phase_aware_weight``).

    1. **Sign error in Eq. (13)**: the first two terms inside the brackets
       carry a minus sign in the paper but should be positive, giving

           F_{sign}(T) = 2 N_at [b₁ τ ln(1+τ⁻²) + b₂ τ arctan(τ)]

       where N_at = 3 N_A / M_{H₂O}, τ = T/647 K, b₁ = 3 × 10⁻²⁰ J,
       b₂ = 1.35 × 10⁻²⁰ J.

    2. **Reference entropy revision**: S₀ was revised from 4.9 kB n_at to
       9.8 kB n_at, requiring an additive correction

           F_{S₀}(T) = (S₀,old − S₀,new) T = −4.9 kB N_at T

       This contributes a constant entropy shift of +4.9 kB N_at.

    With F_shift = F_{sign} + F_{S₀}, the entropy shift is S_shift = −F_shift / T and the internal energy needs no shift.

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
    interpolators will raise RuntimeError for out-of-range queries.
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
        P_all = data[:, 0]    # Pa
        T_all = data[:, 1]    # K
        rho_all = data[:, 2]  # kg/m³
        nad_all = data[:, 3]  # dimensionless — adiabatic gradient
        s_all = data[:, 4]    # J/(kg·K) — specific entropy
        u_all = data[:, 5]    # J/kg — specific internal energy
        w_all = data[:, 6]    # m/s — speed of sound
        mmw_all = data[:, 7]  # kg/mol — mean molecular weight
        xion_all = data[:, 8] # dimensionless — ionization fraction
        xd_all = data[:, 9]   # dimensionless — dissociation fraction
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
        self._mmw_grid = mmw_all.reshape(self.n_P, self.n_T)
        self._xion_grid = xion_all.reshape(self.n_P, self.n_T)
        self._xd_grid = xd_all.reshape(self.n_P, self.n_T)
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
        # Correct s on the AQUA nodes, where the phase and the M19 weight are those AQUA used, then interpolate
        P_node, T_node = np.meshgrid(10.0**lP, 10.0**lT, indexing='ij')
        w7 = self._phase_aware_weight(P_node, T_node, self._phase_grid)
        self._s_corrected_grid = self._s_grid + w7 * self._entropy_shift(np.maximum(T_node, 300.0))
        self._interp_s_corrected = RegularGridInterpolator(
            (lP, lT), self._s_corrected_grid, **kw
        )
        self._interp_u = RegularGridInterpolator(
            (lP, lT), self._u_grid, **kw
        )
        self._interp_log_w = RegularGridInterpolator(
            (lP, lT), np.log10(np.clip(self._w_grid, 1e-30, None)), **kw
        )

        # Composition / dissociation / ionization fields
        self._interp_mmw = RegularGridInterpolator(
            (lP, lT), self._mmw_grid, **kw
        )
        self._interp_xion = RegularGridInterpolator(
            (lP, lT), self._xion_grid, **kw
        )
        self._interp_xd = RegularGridInterpolator(
            (lP, lT), self._xd_grid, **kw
        )

        # Phase grid (nearest-neighbour for integer-valued field)
        self._interp_phase = RegularGridInterpolator(
            (lP, lT), self._phase_grid.astype(float),
            method='nearest', bounds_error=True
        )

    # =========================================================================
    # Interpolation helper with bounds-error conversion
    # =========================================================================

    def _eval_interp(self, interp, P: float, T: float) -> float:
        """
        Evaluate a RegularGridInterpolator, converting bounds errors.

        All AQUA interpolators use ``bounds_error=True``, which raises
        ``ValueError`` for out-of-range queries.  This helper catches
        that exception and re-raises it as ``RuntimeError``, consistent
        with the convention used by ``_find_volume`` in the iron and
        MgSiO₃ EoS modules.

        Parameters
        ----------
        interp : RegularGridInterpolator
            Interpolator to evaluate.
        P : float
            Pressure [Pa].
        T : float
            Temperature [K].

        Returns
        -------
        float
            Interpolated value.

        Raises
        ------
        RuntimeError
            If (P, T) falls outside the AQUA table bounds.
        """
        pt = np.array([[np.log10(P), np.log10(T)]])
        try:
            return float(interp(pt)[0])
        except ValueError as e:
            raise RuntimeError(
                f"H₂O EoS evaluation out of bounds at "
                f"P = {P:.3e} Pa, T = {T:.3e} K. "
                f"AQUA table covers "
                f"{10**self.log10_P_grid[0]:.1e}–"
                f"{10**self.log10_P_grid[-1]:.1e} Pa and "
                f"{10**self.log10_T_grid[0]:.1f}–"
                f"{10**self.log10_T_grid[-1]:.1f} K."
            ) from e

    # =========================================================================
    # Raw table interpolation (before corrections)
    # =========================================================================

    def _raw_density(self, P: float, T: float) -> float:
        """Interpolate density from the AQUA table [kg/m³]."""
        return 10.0**self._eval_interp(self._interp_log_rho, P, T)

    def _raw_adiabatic_gradient(self, P: float, T: float) -> float:
        """Interpolate adiabatic gradient from the AQUA table [dimensionless]."""
        return self._eval_interp(self._interp_nad, P, T)

    def _raw_entropy(self, P: float, T: float) -> float:
        """Interpolate specific entropy from the AQUA table [J/(kg·K)]."""
        return self._eval_interp(self._interp_s, P, T)

    def _raw_internal_energy(self, P: float, T: float) -> float:
        """Interpolate specific internal energy from the AQUA table [J/kg]."""
        return self._eval_interp(self._interp_u, P, T)

    def _raw_speed_of_sound(self, P: float, T: float) -> float:
        """Interpolate speed of sound from the AQUA table [m/s]."""
        return 10.0**self._eval_interp(self._interp_log_w, P, T)

    def _raw_phase_id(self, P: float, T: float) -> int:
        """Interpolate (nearest-neighbour) the AQUA phase ID."""
        return int(round(self._eval_interp(self._interp_phase, P, T)))

    def _raw_mmw(self, P: float, T: float) -> float:
        """Interpolate mean molecular weight from the AQUA table [kg/mol]."""
        return self._eval_interp(self._interp_mmw, P, T)

    def _raw_ionization_fraction(self, P: float, T: float) -> float:
        """Interpolate ionization fraction from the AQUA table [dimensionless]."""
        return self._eval_interp(self._interp_xion, P, T)

    def _raw_dissociation_fraction(self, P: float, T: float) -> float:
        """Interpolate dissociation fraction from the AQUA table [dimensionless]."""
        return self._eval_interp(self._interp_xd, P, T)

    # =========================================================================
    # Mazevet et al. (2019) F_T correction
    # =========================================================================

    @staticmethod
    def _region7_ramps(P, T):
        """
        Individual Region-7 transition ramps ``(w3, w5, w6)``.

        Each ramp is ``∈ [0, 1]`` and corresponds to one of the three
        transitions that AQUA uses to blend the Mazevet+19 EoS with its
        neighbours (Haldemann et al. 2020, Eqs. 20, 26, and 27):

        * ``w3``: 3 ↔ 7 (ice X / M19), T ≤ 2250 K, between 10^11.5 and 10^11.8 Pa (316 and 631 GPa). Haldemann et al. (2020) quote 300 to 700 GPa, but the blend in the AQUA table runs between these two pressures.
        * ``w5``: 5 ↔ 7 (Brown 2018 / M19), from P_5to7 of Eq. (26) to 1.5 P_5to7, T ∈ [1800, 5500] K, isothermal above 4500 K.
        * ``w6``: 6 ↔ 7 (CEA / M19), from P_6to7 to 3 P_6to7, T ∈ (5500, 1e5] K.

        Each ramp is linear in P across its band, as the AQUA blend factor of Eq. (20), and all three are zero above 1e5 K.

        The 3↔7 ramp stays active below 300 K, where AQUA holds M19 at its 300 K values (Haldemann+20 §2.3.3), so the shift that multiplies it is evaluated at ``max(T, 300 K)`` (see ``_build_interpolators``). The 5↔7 and 6↔7 ramps are zero below 1800 K.

        Splitting the ramps lets callers apply a phase-aware veto only
        where it is needed: the 5↔7 and 6↔7 ramps can be silenced on
        ice-VII/ice-X points that fall inside their overlap strip, while
        the 3↔7 ramp stays active there because ice-X is the *intended*
        low-P neighbour of that transition.

        Parameters
        ----------
        P : float or array-like
            Pressure [Pa]
        T : float or array-like
            Temperature [K]

        Returns
        -------
        tuple of float or ndarray
            ``(w3, w5, w6)``.  Scalars if both inputs are scalar.
        """
        P_arr = np.asarray(P, dtype=float)
        T_arr = np.asarray(T, dtype=float)

        # No 300 K floor: AQUA holds M19 at its 300 K values below 300 K, and the shift is evaluated at max(T, 300 K)
        in_domain = (T_arr <= 1.0e5) & (P_arr > 0.0)

        def ramp(lo, hi):
            return np.clip((P_arr - lo) / (hi - lo), 0.0, 1.0)

        # 3 <-> 7: constant P bounds at 10^11.5 and 10^11.8 Pa, for T <= 2250 K
        w3 = np.where((T_arr <= 2250.0) & in_domain, ramp(10.0**11.5, 10.0**11.8), 0.0)

        # 5 <-> 7: Eq. (26) for T in [1800, 5500] K, isothermal above 4500 K
        T_eff = np.minimum(T_arr, 4500.0)
        P57 = 10.0**(np.log10(42e9) - np.log10(6.0) * (T_eff / 1000.0 - 2.0) / 18.0)
        w5 = np.where((T_arr >= 1800.0) & (T_arr <= 5500.0) & in_domain, ramp(P57, 1.5 * P57), 0.0)

        # 6 <-> 7: for T > 5500 K, where Region 6 (CEA) is the low-P neighbor
        P67 = (0.05 + (3.0 - 0.05) * (T_arr / 1000.0 - 1.0) / 39.0) * 1e9
        w6 = np.where((T_arr > 5500.0) & in_domain, ramp(P67, 3.0 * P67), 0.0)

        if w3.ndim == 0:
            return float(w3), float(w5), float(w6)
        return w3, w5, w6

    @staticmethod
    def _region7_weight(P, T):
        """
        Analytic Region-7 weight ``w ∈ [0, 1]`` (max of the three
        transition ramps, ignoring phase).

        This is the pure analytic form of the weight, useful for visualization on a (P, T) grid. The corrected entropy does not use it: it takes ``_phase_aware_weight`` on every AQUA node, which silences the 5↔7 and 6↔7 ramps on ice nodes inside their overlap strip with the 3↔7 transition.

        See ``_region7_ramps`` for the per-ramp definitions and for a
        description of the transition boundaries.

        Parameters
        ----------
        P : float or array-like
            Pressure [Pa]
        T : float or array-like
            Temperature [K]

        Returns
        -------
        float or ndarray
            ``max(w3, w5, w6)``.  Scalar if both inputs are scalar.
        """
        w3, w5, w6 = Haldemann20._region7_ramps(P, T)
        w = np.maximum(np.maximum(w3, w5), w6)
        return w if np.ndim(w) > 0 else float(w)

    @staticmethod
    def _f_shift(T: float) -> float:
        """
        Specific Helmholtz free energy correction F_shift / mass.

        F_shift = 2 N_at [b₁ τ ln(1 + τ⁻²) + b₂ τ arctan(τ)]
                  + (S₀,old − S₀,new) N_at T

        where N_at = 3 N_A / (18 g/mol), τ = T / 647.34 K, and
        S₀,old = 4.9 kB, S₀,new = 9.8 kB (per atom).

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
        f_sign = 2.0 * _N_AT_PER_KG * (term1 + term2)
        f_s0 = (_S0_OLD_PER_ATOM - _S0_NEW_PER_ATOM) * _N_AT_PER_KG * T
        return f_sign + f_s0

    @staticmethod
    def _df_shift_dT(T: float) -> float:
        """
        Temperature derivative of the specific free energy correction. The entropy shift does not use it (see ``_entropy_shift``), and it is kept for the benchmark notebook.

        ∂F_shift/∂T = 2 N_at [
            b₁ ( (1/T_c) ln(1 + T_c²/T²)  −  2 T_c / (T² + T_c²) )
          + b₂ ( (1/T_c) arctan(T/T_c)     +  T / (T² + T_c²) )
        ] + (S₀,old − S₀,new) N_at

        where S₀,old = 4.9 kB, S₀,new = 9.8 kB (per atom).

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            ∂F_shift/∂T [J/(kg·K)]
        """
        tau = T / _T_CRIT
        Tc = _T_CRIT
        denom = T**2 + Tc**2

        d_term1 = _b1 * ((1.0 / Tc) * np.log(1.0 + Tc**2 / T**2)
                         - 2.0 * Tc / denom)
        d_term2 = _b2 * ((1.0 / Tc) * np.arctan(T / Tc)
                         + T / denom)
        df_sign = 2.0 * _N_AT_PER_KG * (d_term1 + d_term2)
        df_s0 = (_S0_OLD_PER_ATOM - _S0_NEW_PER_ATOM) * _N_AT_PER_KG
        return df_sign + df_s0

    @staticmethod
    def _entropy_shift(T: float) -> float:
        """
        Specific entropy correction.

        S_shift = −F_shift / T

        The corrected fit keeps the internal energy of the one AQUA was built with and changes only F, so s = (u − f) / T moves by −F_shift / T. This is not −∂F_shift/∂T, because F_shift is not paired with a matching change of u.

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific entropy shift [J/(kg·K)]
        """
        return -Haldemann20._f_shift(T) / T

    @staticmethod
    def _energy_shift(T: float) -> float:
        """
        Specific internal energy correction, zero, kept for the benchmark notebook.

        The corrected fit has the internal energy of the one AQUA was built with, so U_shift = F_shift + T S_shift = 0.

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            Specific internal energy shift [J/kg]
        """
        return 0.0 * T

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

    def _mazevet_correction_weight(self, P: float, T: float) -> float:
        """
        Effective Mazevet-correction weight at (P, T), combining the
        three analytic Region-7 ramps with a phase-aware veto on the
        5↔7 and 6↔7 ramps.

        The 3↔7 ramp is applied unconditionally within its T ≤ 2250 K
        strip because ice-VII/ice-X is the intended low-P neighbour of
        that transition in AQUA: points labelled ``solid-ice-X`` sitting
        between 300 and 700 GPa are precisely the ones for which AQUA
        blends M19 with Region 3 and so must still carry the M19 shift.

        The 5↔7 and 6↔7 ramps, on the other hand, connect M19 to
        liquid/supercritical neighbours (Brown 2018, CEA); they are
        silenced whenever the AQUA phase code is not 5 (supercritical +
        superionic).  This kills the residual over-correction produced
        by the ``max``-combination rule on ice-bearing points inside the
        T ∈ [1800, 2250] K overlap strip between the 5↔7 and 3↔7
        transitions, without interfering with the 3↔7 ramp itself.

        The entropy correction applies the same rule on the AQUA nodes, with the phase code of each node (see ``_build_interpolators``). At an off-node query this function takes the nearest-node phase, so it can return 0 between an ice X node and a supercritical node (just above 2250 K) although both carry the shift.
        """
        return float(self._phase_aware_weight(P, T, self._raw_phase_id(P, T)))

    @staticmethod
    def _phase_aware_weight(P, T, phase_code):
        """Region-7 weight max(w3, w5, w6), with w5 and w6 kept only where the AQUA phase code is 5 (supercritical and superionic)."""
        w3, w5, w6 = Haldemann20._region7_ramps(P, T)
        return np.maximum(w3, np.where(np.asarray(phase_code) == 5, np.maximum(w5, w6), 0.0))

    def specific_internal_energy(self, P: float, T: float) -> float:
        """
        Calculate specific internal energy.

        U(P,T) = U_AQUA(P,T)

        The corrected Mazevet+19 fit has the internal energy of the version AQUA was built with, so Region 7 needs no shift here (see ``_energy_shift``).

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
        return self._raw_internal_energy(P, T)

    def specific_entropy(self, P: float, T: float) -> float:
        """
        Calculate specific entropy (corrected).

        S = S_AQUA + w_7 · S_shift(max(T, 300 K)) on every AQUA node, S_shift = −F_shift / T, interpolated like the other AQUA columns

        The Mazevet+19 shift is gated by the Region-7 weight ``w_7``, so it only affects nodes where AQUA uses the M19 EoS, fully or through a transition blend. Each node takes the weight of its own AQUA phase code (see ``_phase_aware_weight``), so a query between an ice X node and a supercritical node, both from M19, gets the shift of both. It is evaluated at ``max(T, 300 K)`` because AQUA extends M19 below 300 K by holding it at its 300 K values (Haldemann+20 §2.3.3), so the low-temperature nodes carry the 300 K error.

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
        return self._eval_interp(self._interp_s_corrected, P, T)

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
            return cp
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

    # Smallest change of log10(rho) across the difference stencil that the 9-digit AQUA density resolves to a few percent
    _LOG_RHO_RESOLUTION = 1e-7

    def isothermal_bulk_modulus(self, P: float, T: float) -> float:
        """
        Isothermal bulk modulus K_T = (∂P/∂ ln ρ)_T of the tabulated density [Pa].

        Centered difference one AQUA grid step on either side in log P, one-sided where a side leaves the table or lies in another phase. NaN where the change of log10 ρ across the stencil is below 1e-7, the precision of the AQUA density, as for ice below about 0.35 bar and liquid water below about 0.04 bar.
        """
        lp = np.log10(P)
        step = self.log10_P_grid[1] - self.log10_P_grid[0]
        phase = self._raw_phase_id(P, T)
        sides = [x for x in (lp - step, lp + step)
                 if self.log10_P_grid[0] <= x <= self.log10_P_grid[-1] and self._raw_phase_id(10.0**x, T) == phase]
        if not sides:
            return np.nan
        lo, hi = (sides[0], sides[-1]) if len(sides) == 2 else sorted((sides[0], lp))
        dlogrho = self._eval_interp(self._interp_log_rho, 10.0**hi, T) - self._eval_interp(self._interp_log_rho, 10.0**lo, T)
        if not dlogrho > self._LOG_RHO_RESOLUTION:
            return np.nan
        return P * (hi - lo) / dlogrho

    def adiabatic_bulk_modulus(self, P: float, T: float) -> float:
        """Adiabatic bulk modulus K_S = K_T C_P / C_V = K_T (1 + T α² w² / C_P) [Pa], with the C_V identity above. NaN where C_P <= 0."""
        K_T = self.isothermal_bulk_modulus(P, T)
        alpha = self._thermal_expansion_numerical(P, T)
        if alpha == 0:
            return K_T
        cp = self.isobaric_heat_capacity(P, T)
        if not cp > 0:
            return np.nan
        return K_T * (1 + T * alpha**2 * self._raw_speed_of_sound(P, T)**2 / cp)

    def shear_modulus(self, P: float, T: float) -> float:
        """Unrelaxed shear modulus [Pa]. PALEOS has no shear model for water or ice yet, so this is NaN."""
        return np.nan

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

    def mmw(self, P: float, T: float) -> float:
        """
        Mean molecular weight from the AQUA table.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Mean molecular weight [kg/mol]
        """
        return self._raw_mmw(P, T)

    def ionization_fraction(self, P: float, T: float) -> float:
        """
        Ionization fraction from the AQUA table, ``x_ion = N_e / N_tot``.

        The bilinearly interpolated value is clipped to ``[0, 1]`` as a
        defensive measure: the tabulated AQUA values are already in this
        range, so linear interpolation cannot leave it, but the clip
        guards against future numerical rounding.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Ionization fraction (dimensionless, ∈ [0, 1])
        """
        return float(np.clip(self._raw_ionization_fraction(P, T), 0.0, 1.0))

    def dissociation_fraction(self, P: float, T: float) -> float:
        """
        Dissociation fraction from the AQUA table, ``x_d = 1 - N_H2O / N_tot``.

        The bilinearly interpolated value is clipped to ``[0, 1]`` as a
        defensive measure (see :meth:`ionization_fraction`).

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        float
            Dissociation fraction (dimensionless, ∈ [0, 1])
        """
        return float(np.clip(self._raw_dissociation_fraction(P, T), 0.0, 1.0))

    def adiabatic_gradient_from_corrected_entropy(
        self, P: float, T: float
    ) -> float:
        """
        Adiabatic gradient computed from the *corrected* specific entropy.

        ∇_ad = α P / (ρ C_P^corr)

        where C_P^corr = T (∂s_corr/∂T)_P is obtained by numerical
        differentiation of the corrected entropy.

        This is a consistency check: it should agree with the tabulated ∇_ad wherever the corrected entropy is consistent with the AQUA density and adiabatic gradient, Region 7 and its 3↔7 band included. It does not agree in the 5↔7 band, whose bounds move with T, and below 300 K above 316 GPa, where AQUA holds M19 at its 300 K values and C_P^corr vanishes.

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


# =============================================================================
# Wrapper Class
# =============================================================================


class WaterEoS:
    """
    Wrapper equation of state for H₂O with pre-loaded AQUA table.

    This class loads the Haldemann20 tabulated EoS once at initialization
    and exposes the standard PALEOS public interface plus a ``phase``
    method. It avoids repeated table loading and interpolator construction
    when the EoS is queried many times (e.g. during interior structure
    integration).

    Since all H₂O phases are described by a single tabulated EoS (AQUA),
    the wrapper delegates every call directly to the underlying
    Haldemann20 instance.  The ``phase`` method returns the AQUA phase
    label at the queried (P, T) point.

    Parameters
    ----------
    table_path : str
        Path to the AQUA P-T table file.

    Attributes
    ----------
    _eos : Haldemann20
        Underlying tabulated EoS instance.

    Examples
    --------
    >>> eos = WaterEoS('path/to/AQUA_PT_table.dat')
    >>> rho = eos.density(50e9, 2000)
    >>> phase = eos.phase(50e9, 2000)
    >>> print(f"{phase}: rho = {rho:.1f} kg/m³")
    """

    def __init__(self, table_path):
        """
        Initialize WaterEoS by loading the AQUA P-T table.

        Parameters
        ----------
        table_path : str
            Path to the AQUA P-T table file.
        """
        self._eos = Haldemann20(table_path)

    def phase(self, P, T):
        """
        Return the stable H₂O phase at given P and T.

        Parameters
        ----------
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]

        Returns
        -------
        str
            Phase label (e.g. 'liquid', 'solid-ice-VII', 'supercritical')
        """
        return self._eos.phase(P, T)

    def density(self, P, T):
        """Calculate density [kg/m³]."""
        return self._eos.density(P, T)

    def specific_internal_energy(self, P, T):
        """Calculate specific internal energy [J/kg]."""
        return self._eos.specific_internal_energy(P, T)

    def specific_entropy(self, P, T):
        """Calculate specific entropy [J/(kg·K)]."""
        return self._eos.specific_entropy(P, T)

    def isobaric_heat_capacity(self, P, T):
        """Calculate specific isobaric heat capacity [J/(kg·K)]."""
        return self._eos.isobaric_heat_capacity(P, T)

    def isochoric_heat_capacity(self, P, T):
        """Calculate specific isochoric heat capacity [J/(kg·K)]."""
        return self._eos.isochoric_heat_capacity(P, T)

    def thermal_expansion(self, P, T):
        """Calculate volumetric thermal expansion coefficient [K⁻¹]."""
        return self._eos.thermal_expansion(P, T)

    def adiabatic_gradient(self, P, T):
        """Calculate dimensionless adiabatic temperature gradient."""
        return self._eos.adiabatic_gradient(P, T)

    def isothermal_bulk_modulus(self, P, T):
        """Calculate isothermal bulk modulus K_T [Pa]."""
        return self._eos.isothermal_bulk_modulus(P, T)

    def adiabatic_bulk_modulus(self, P, T):
        """Calculate adiabatic bulk modulus K_S [Pa]."""
        return self._eos.adiabatic_bulk_modulus(P, T)

    def shear_modulus(self, P, T):
        """Unrelaxed shear modulus [Pa], NaN for every water phase."""
        return self._eos.shear_modulus(P, T)

    def mmw(self, P, T):
        """Mean molecular weight [kg/mol]."""
        return self._eos.mmw(P, T)

    def ionization_fraction(self, P, T):
        """Ionization fraction x_ion = N_e/N_tot (dimensionless, ∈ [0, 1])."""
        return self._eos.ionization_fraction(P, T)

    def dissociation_fraction(self, P, T):
        """Dissociation fraction x_d = 1 - N_H2O/N_tot (dimensionless, ∈ [0, 1])."""
        return self._eos.dissociation_fraction(P, T)
