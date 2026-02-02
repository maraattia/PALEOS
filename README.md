# PALEOS

**Planetary Assemblage Layers: Equations Of State**

Python package for calculating thermodynamic properties of planetary materials under extreme pressure—temperature conditions relevant to planetary interiors.

## Current Implementation

### Iron Equations of State

Complete coverage of the iron phase diagram for planetary core modeling:

| Class | Phase | Structure | Reference |
|-------|-------|-----------|-----------|
| `Dorogokupets17` | α-Fe, δ-Fe | bcc | Dorogokupets et al. (2017) |
| `Dorogokupets17` | γ-Fe | fcc | Dorogokupets et al. (2017) |
| `Miozzi20` | ε-Fe | hcp | Miozzi et al. (2020) |
| `Hakim18` | ε-Fe | hcp | Hakim et al. (2018) |
| `HcpIronEos` | ε-Fe | hcp | Blended `Miozzi20` + `Hakim18` |
| `Luo24` | liquid | — | Luo et al. (2024) |

All EoS classes share a consistent interface, returning seven thermodynamic quantities as functions of pressure (Pa) and temperature (K):

| Method | Quantity | Units |
|--------|----------|-------|
| `density(P, T)` | Density | kg/m³ |
| `specific_internal_energy(P, T)` | Specific internal energy | J/kg |
| `specific_entropy(P, T)` | Specific entropy | J/(kg·K) |
| `isobaric_heat_capacity(P, T)` | Isobaric heat capacity | J/(kg·K) |
| `isochoric_heat_capacity(P, T)` | Isochoric heat capacity | J/(kg·K) |
| `thermal_expansion(P, T)` | Thermal expansion coefficient | K⁻¹ |
| `adiabatic_gradient(P, T)` | Adiabatic gradient $(\partial \ln T/\partial \ln P)_S$ | dimensionless |

### Python API

Automatic phase selection following the BICEPS model (Haldemann et al. 2024):

```python
from paleos import iron_eos as fe

# Get stable phase at given P-T
phase = fe.get_iron_phase(P=100e9, T=3000)

# Get EoS instance for a specific phase
eos = fe.get_iron_eos('epsilon')

# Get EoS and phase for given P-T (recommended)
eos, phase = fe.get_iron_eos_for_PT(P=200e9, T=4000)
rho = eos.density(200e9, 4000)
```

Phase boundaries from Dorogokupets et al. (2017); melting curve from Anzellini et al. (2013).

### Lookup Tables

Precomputed tables for fast interpolation, generated at 150 points per decade on a log-uniform grid:

| Table | P range | T range | Grid size | File size |
|-------|---------|---------|-----------|-----------|
| `iron_eos_table.dat` | 1 bar – 10 TPa | 300 K – 100,000 K | 1201 × 380 | ~60 MB |

Tables include all seven thermodynamic quantities plus phase identification. Interpolation in (log P, log T) space yields relative errors below 10⁻⁴ for density at the 99th percentile.

## Installation

```bash
git clone https://github.com/maraattia/PALEOS.git
cd PALEOS
pip install -e .
```

## Usage

```python
from paleos import iron_eos as fe

# Direct EoS evaluation
eos, phase = fe.get_iron_eos_for_PT(P=300e9, T=5000)
print(f"Phase: {phase}")
print(f"Density: {eos.density(300e9, 5000):.1f} kg/m³")
print(f"Entropy: {eos.specific_entropy(300e9, 5000):.1f} J/(kg·K)")

# Using lookup tables
import numpy as np
from scipy.interpolate import RegularGridInterpolator

cols = ['P', 'T', 'rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad', 'phase']
data = np.genfromtxt('iron_eos_table.dat', comments='#',
                     names=cols, dtype=None, encoding='utf-8')

n_P, n_T = 1201, 380
log_P = np.log10(data['P'].reshape(n_P, n_T)[:, 0])
log_T = np.log10(data['T'].reshape(n_P, n_T)[0, :])
rho = data['rho'].reshape(n_P, n_T)

interp_rho = RegularGridInterpolator((log_P, log_T), rho)
rho_interp = interp_rho([[np.log10(300e9), np.log10(5000)]])[0]
```

## Future Development

- MgSiO3 equations of state for silicate mantles
- Updated AQUA H₂O tables with corrected entropy
- Combined iron-silicate-water planetary models

## License

This code is licensed under the BSD 3-Clause License—see the [LICENSE](LICENSE) file for details.

## Acknowledgements

The developer of this software is Mara Attia. We acknowledge the use of the Claude AI assistant (Anthropic, 2024) for code optimization. The following publications have been employed:

- Anzellini et al. (2013) *Science* 340:464–466
- Dorogokupets et al. (2017) *Sci. Rep.* 7:41863
- Hakim et al. (2018) *Icarus* 313:61–78
- Haldemann et al. (2024) *A&A* 681:A96
- Luo et al. (2024) *Nature Commun.* 15:1417
- Miozzi et al. (2020) *JGR Planets* 125:e2019JE006294

## Contact

Mara Attia — maraaattia@gmail.com
