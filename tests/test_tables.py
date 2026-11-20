"""
The table rules of paleos/checks.py, applied to coarse grids built the way the generators build the published tables.

Every generator runs check_table_consistency before writing. These tests run the same check on coarse grids over the iron and MgSiO3 table boxes (8 and 6 points per decade, to keep the suite fast). The sign, stability, and identity rules bite at any resolution, but the difference rules (alpha, cp, first_law) widen their tolerance with the grid step, so here they only catch gross errors. The full-resolution check at release is stricter.
"""

import numpy as np
import pytest

from paleos import iron_eos as fe
from paleos import mgsio3_eos as mg
from paleos.checks import check_table_consistency

QUANTITIES = (('rho', 'density'), ('u', 'specific_internal_energy'), ('s', 'specific_entropy'),
              ('cp', 'isobaric_heat_capacity'), ('cv', 'isochoric_heat_capacity'),
              ('alpha', 'thermal_expansion'), ('nabla_ad', 'adiabatic_gradient'))


def log_grid(lo, hi, ppd):
    return 10 ** np.linspace(np.log10(lo), np.log10(hi), int(np.ceil(np.log10(hi / lo) * ppd)) + 1)


def row(eos, phase, P, T):
    return {'P': P, 'T': T, 'phase': phase, **{k: getattr(eos, m)(P, T) for k, m in QUANTITIES}}


@pytest.fixture(scope="module")
def iron_rows():
    wrapper = fe.IronEoS()
    rows = []
    for P in log_grid(1e5, 1e14, 8):
        for T in log_grid(300.0, 1e5, 8):
            phase = wrapper.phase(P, T)
            eos = wrapper._phase_eos_map[phase]
            if phase == 'liquid' and not eos.in_scope(P, T):
                continue
            rows.append(row(eos, phase, P, T))
    return rows


def test_iron_table_rules(iron_rows):
    check_table_consistency(iron_rows, "iron, 8 points per decade")


def test_mgsio3_table_rules(mgsio3_phases):
    rows = []
    for P in log_grid(1e5, 1e14, 6):
        for T in log_grid(300.0, 1e5, 6):
            phase = mg.get_mgsio3_phase(P, T)
            try:
                rows.append(row(mgsio3_phases[phase], phase, P, T))
            except (RuntimeError, ValueError):
                continue  # the generator omits points where the liquid EoS has no solution
    check_table_consistency(rows, "MgSiO3, 6 points per decade")


def test_the_rules_catch_an_alpha_off_its_density(iron_rows):
    """The check must not be vacuous: alpha scaled by 1.3 in the liquid, as the old floor did in part of it, has to fail."""
    bad = [dict(r, alpha=r['alpha'] * 1.3, nabla_ad=r['nabla_ad'] * 1.3) if r['phase'] == 'liquid' else r for r in iron_rows]
    with pytest.raises(ValueError, match=r"\[alpha\]"):
        check_table_consistency(bad, "corrupted iron")
