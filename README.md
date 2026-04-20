# PALEOS

**Planetary Assemblage Layers: Equations Of State**

An open-source Python package for computing thermodynamic properties of planetary materials under extreme pressure–temperature conditions relevant to planetary interiors.

> **Just want the data?** Precomputed tables are available on Zenodo—no installation required:
>
> | **EoS lookup tables** (Fe, MgSiO₃, H₂O) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19000316.svg)](https://doi.org/10.5281/zenodo.19000316) |
> |---|---|
> | **Mass–radius tables** (rocky & water-rich) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19221215.svg)](https://doi.org/10.5281/zenodo.19221215) |

## Overview

Interior modeling of rocky and water-rich exoplanets requires thermodynamic properties of core, mantle, and volatile-layer materials over many orders of magnitude in pressure and temperature. PALEOS consolidates published equations of state for iron, MgSiO₃, and H₂O into a single validated toolkit with automatic phase selection, consistent reference states, and a uniform interface across all materials and phases.

PALEOS can be used in two ways:

1. **Lookup tables** — precomputed grids of thermodynamic quantities, ready to load and interpolate. Best for speed-critical applications such as interior structure integration or MCMC retrievals.
2. **Python API** — on-the-fly EoS evaluation with automatic phase determination. Best for exploratory work, plotting, or when you need quantities at arbitrary (P, T) points without preinterpolation.

## Materials and phases

### Iron (Fe)

Five phases covering the complete planetary core phase diagram: α-bcc, δ-bcc, γ-fcc (Dorogokupets et al. 2017), ε-hcp (Miozzi et al. 2020 blended with Hakim et al. 2018), and liquid (Luo et al. 2024). Solid phase boundaries from Dorogokupets et al. (2017); melting curve from Anzellini et al. (2013). Reference state at the fcc–hcp–liquid triple point (98.5 GPa, 3712 K), pinning the anchor on the three phases that dominate planetary cores. Per-phase reference constants $(U_0, S_0)$ are set by decoupling: $S_0$ is fixed on thermodynamic grounds—ensuring $\Delta S > 0$ across every heating-accessible solid–solid boundary and $\Delta S_\mathrm{fusion} > 0$ on every branch of the Anzellini et al. (2013) melting curve—and $U_0$ is then fixed per phase by enforcing Gibbs-energy continuity at the triple point.

### Magnesium silicate (MgSiO₃)

Six phases covering the planetary mantle: three pyroxene polymorphs—LP-clinoenstatite, orthoenstatite, HP-clinoenstatite (Sokolova et al. 2022)—bridgmanite (Wolf et al. 2015), postperovskite (Sakai et al. 2016), and liquid (Wolf & Bower 2018 with parameters from Luo & Deng 2025). Solid phase boundaries from Sokolova et al. (2022), Ono & Oganov (2005); melting curve from Belonoshko et al. (2005) and Fei et al. (2021). Reference state at the brg–ppv–liquid triple point (155.7 GPa, 6168 K), computed numerically as the intersection of the Ono & Oganov brg↔ppv boundary with the Fei et al. (2021) melting branch. Per-phase reference constants $(U_0, S_0)$ are set by decoupling: $S_0$ is fixed on thermodynamic grounds—ensuring $\Delta S > 0$ across every heating-accessible solid–solid boundary and matching the Stebbins et al. (1984) calorimetric anchor $\Delta S_\mathrm{fusion}(1\,\mathrm{bar}, 1831\,\mathrm{K}) = 41.99\,\mathrm{J/(mol\cdot K)}$ exactly—and $U_0$ is then fixed per phase by enforcing Gibbs-energy continuity at the triple point.

### Water (H₂O)

Based on the AQUA equation of state (Haldemann et al. 2020), covering ice polymorphs (Ih through X), liquid, vapor, and supercritical/superionic water. Includes two corrections to the entropy and internal energy for bugs in the Mazevet et al. (2019) Helmholtz free-energy parametrization used in the supercritical/superionic regime—a sign error on the first two terms of $F_T$, and the revision of the reference entropy from $S_0 = 4.9\,k_B\,n_\mathrm{at}$ to $9.8\,k_B\,n_\mathrm{at}$—gated analytically to the Region 7 blend bands AQUA uses to stitch Mazevet et al. (2019) with its neighbors.

## Thermodynamic quantities

All materials provide the same seven quantities as functions of pressure (Pa) and temperature (K):

| Quantity                      | Symbol               | Method                           | Units         |
|-------------------------------|----------------------|----------------------------------|---------------|
| Density                       | $\rho$               | `density(P, T)`                  | kg/m³         |
| Specific internal energy      | $u$                  | `specific_internal_energy(P, T)` | J/kg          |
| Specific entropy              | $s$                  | `specific_entropy(P, T)`         | J/(kg·K)      |
| Isobaric heat capacity        | $C_P$                | `isobaric_heat_capacity(P, T)`   | J/(kg·K)      |
| Isochoric heat capacity       | $C_V$                | `isochoric_heat_capacity(P, T)`  | J/(kg·K)      |
| Thermal expansion coefficient | $\alpha$             | `thermal_expansion(P, T)`        | K⁻¹           |
| Adiabatic gradient            | $\nabla_\mathrm{ad}$ | `adiabatic_gradient(P, T)`       | dimensionless |

Additionally, `phase(P, T)` returns the stable phase label as a string.

## Installation

```bash
git clone https://github.com/maraattia/PALEOS.git
cd PALEOS
pip install -e .
```

## Usage

### Option 1: Lookup tables

Precomputed tables are hosted on [Zenodo](https://doi.org/10.5281/zenodo.19000316) as plain-text, whitespace-delimited files on log-uniform $(P, T)$ grids. The standard 150 points-per-decade (ppd) variants are the everyday workhorses: bilinear interpolation in $(\log_{10} P, \log_{10} T)$ keeps relative density errors below $10^{-4}$ at the 99th percentile. High-resolution 600 ppd variants are also available for Fe and MgSiO₃, for workflows where thermodynamic derivatives ($\alpha$, $C_P$, $\nabla_\mathrm{ad}$) need tighter accuracy than bilinear interpolation of the 150 ppd grid can deliver.

| Table                                 | Material | P range          | T range     | Resolution |
|---------------------------------------|----------|------------------|-------------|------------|
| `paleos_iron_tables_pt.dat`           | Fe       | 1 bar – 100 TPa  | 300 – 10⁵ K | 150 ppd    |
| `paleos_iron_tables_pt_highres.dat`   | Fe       | 1 bar – 100 TPa  | 300 – 10⁵ K | 600 ppd    |
| `paleos_mgsio3_tables_pt.dat`         | MgSiO₃   | 1 bar – 100 TPa  | 300 – 10⁵ K | 150 ppd    |
| `paleos_mgsio3_tables_pt_highres.dat` | MgSiO₃   | 1 bar – 100 TPa  | 300 – 10⁵ K | 600 ppd    |
| `paleos_water_tables_pt.dat`          | H₂O      | 1 μbar – 100 TPa | 100 – 10⁵ K | 150 ppd    |

Each file contains ten columns: `P`, `T`, `rho`, `u`, `s`, `cp`, `cv`, `alpha`, `nabla_ad`, `phase`.

**Loading and interpolating a table:**

```python
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Grid parameters (see file headers for exact values)
n_P, n_T = 1351, 380
P_min, P_max = 1e5, 1e14       # Pa
T_min, T_max = 300.0, 1e5      # K

# Load
cols = ['P','T','rho','u','s','cp','cv','alpha','nabla_ad','phase']
data = np.genfromtxt('paleos_iron_tables_pt.dat',
                     names=cols, dtype=None, encoding='utf-8')

# Reconstruct full regular grid
log_P = np.linspace(np.log10(P_min), np.log10(P_max), n_P)
log_T = np.linspace(np.log10(T_min), np.log10(T_max), n_T)

rho = np.full((n_P, n_T), np.nan)
for row in data:
    i = np.searchsorted(log_P, np.log10(row['P']))
    j = np.searchsorted(log_T, np.log10(row['T']))
    if i < n_P and j < n_T:
        rho[i, j] = row['rho']

# Interpolate
interp = RegularGridInterpolator((log_P, log_T), rho)
rho_query = interp([[np.log10(100e9), np.log10(4000)]])[0]
```

> **Note:** Grid points where the EoS failed to converge are omitted from the MgSiO₃ table (primarily in the low-P, high-T liquid regime, reflecting the vapor/supercritical regime not captured by PALEOS). Reconstruct the full rectangular grid with NaN fill before building an interpolator, as shown above.

### Option 2: Python API

PALEOS provides a high-level interface for each material that handles phase selection automatically. You instantiate one object per material; all subsequent queries resolve the stable phase internally.

**Fe:**

```python
from paleos.iron_eos import IronEoS

fe = IronEoS()

rho   = fe.density(200e9, 4000)
phase = fe.phase(200e9, 4000)                    # 'solid-epsilon'
cp    = fe.isobaric_heat_capacity(200e9, 4000)
s     = fe.specific_entropy(200e9, 4000)
```

**MgSiO₃:**

```python
from paleos.mgsio3_eos import MgSiO3EoS

mg = MgSiO3EoS()

rho   = mg.density(50e9, 2000)
phase = mg.phase(50e9, 2000)                      # 'solid-brg'
s     = mg.specific_entropy(50e9, 2000)
u     = mg.specific_internal_energy(50e9, 2000)
```

**H₂O:**

```python
from paleos.water_eos import WaterEoS

h2o = WaterEoS('/path/to/AQUA_PT_table.dat')

rho   = h2o.density(50e9, 2000)
phase = h2o.phase(50e9, 2000)                     # 'solid-ice-VII'
alpha = h2o.thermal_expansion(50e9, 2000)
```

### Mass–radius tables

Precomputed mass–radius relations for rocky and water-rich planets are hosted on [Zenodo](https://doi.org/10.5281/zenodo.19221215). The archive contains one `.dat` file per surface temperature, organized in `rocky/` and `water/` subdirectories.

| Family | Composition | CMF/WMF values | T_surf range | Files |
|--------|-------------|----------------|--------------|-------|
| Rocky | Fe core + MgSiO₃ mantle | 22 CMF (0.00–1.00, incl. Earth-like 0.325) | 300–4000 K (9 values) | 9 |
| Water-rich | Earth-like core + H₂O envelope | 20 WMF (0.05–1.00) | 300–1000 K (8 values) | 8 |

All models use 1 bar surface pressure, adiabatic temperature profiles, and 50 log-uniform masses from 0.1 to 100 M⊕. Each file contains three columns: `CMF` (or `WMF`), `M_earth`, `R_earth`.

**Loading a table:**

```python
import pandas as pd

df = pd.read_csv('paleos_mr_rocky_T300K.dat', comment='#', skip_blank_lines=True)
# columns: CMF, M_earth, R_earth
```

### Phase labels

All modules use a consistent `solid-*` / `liquid` naming convention:

- **Fe:** `solid-alpha`, `solid-delta`, `solid-gamma`, `solid-epsilon`, `liquid`
- **MgSiO₃:** `solid-lpcen`, `solid-en`, `solid-hpcen`, `solid-brg`, `solid-ppv`, `liquid`
- **H₂O:** `solid-ice-Ih`, `solid-ice-II`, `solid-ice-III`, `solid-ice-V`, `solid-ice-VI`, `solid-ice-VII`, `solid-ice-X`, `vapor`, `liquid`, `supercritical`

## License

BSD 3-Clause License—see [LICENSE](LICENSE).

## Acknowledgements

Developed by Mara Attia. We acknowledge the use of the Claude AI assistant (Anthropic) for documentation and code optimization.

## Contact

Mara Attia — maraaattia@gmail.com