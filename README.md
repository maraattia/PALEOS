# PALEOS

**Planetary Assemblage Layers: Equations Of State**

Python package for calculating thermodynamic properties of planetary materials.

## Current Implementation

### Iron EoS
- **Dorogokupets17**: bcc-Fe and fcc-Fe (Dorogokupets et al., 2017)
- **Miozzi20**: hcp-Fe for high-pressure conditions (Miozzi et al., 2020)
- **Hakim18**: hcp-Fe for super-Earth conditions (Hakim et al., 2018)

All classes return 7 thermodynamic quantities from pressure and temperature:
density, specific internal energy, specific entropy, isobaric heat capacity, 
isochoric heat capacity, thermal expansion coefficient, adiabatic gradient.

## Installation
```bash
echo 'PLACEHOLDER'
```

## Usage
```python
print('PLACEHOLDER')
```

## Future Development

- MgSiO3 equations of state
- H2O equations of state
- Lookup tables for fast interpolation

## References

- Dorogokupets et al. (2017) *Sci. Rep.* 7:41863
- Miozzi et al. (2020) *JGR Planets* 125:e2019JE006294
- Hakim et al. (2018) *Icarus* 313:61-78

## Contact

Mara Attia