"""
Thermodynamic consistency checks for tabulated EoS rows, shared by every PALEOS table generator.

Each row is a dict with keys P [Pa], T [K], rho, u, s, cp, cv, alpha, nabla_ad (SI, per kg), and phase. The rules hold for any material and any table:

- finite: every quantity is finite.
- heat_capacities: C_V > 0 and C_P >= C_V (equivalent to K_T > 0).
- nabla_ad: nabla_ad = P alpha / (rho C_P), to 1e-6.
- entropy_isobar: s rises with T along every isobar, phase boundaries included.
- density_isotherm: rho rises with P along every isotherm within one phase.
- entropy_isotherm: s falls with P along every isotherm within one phase where alpha > 0, as the Maxwell relation (ds/dP)_T = -alpha/rho requires.
- alpha, cp, first_law: within one phase, by centered differences along each isobar, alpha matches -d ln(rho)/dT, C_P matches T ds/dT, and du matches T ds - P dv, each to rel_tol plus half the change of the tabulated quantity across the three-point stencil, which bounds the truncation error of the difference where the quantity varies fast (near a spinodal, for instance).

Rows that carry the elastic moduli K_T, K_S, and mu (Pa) also face these rules:

- finite: K_T and K_S are finite. mu may be NaN, where no shear model applies or the model is out of range.
- bulk_moduli: K_T > 0 and K_S / K_T = C_P / C_V, to 1e-6.
- shear_modulus: mu > 0 wherever it is not NaN.
- K_T: within one phase, by centered differences along each isotherm, K_T matches d P / d ln(rho), to the same tolerance as alpha.
"""

import numpy as np

QUANTITIES = ('rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad')
MODULI = ('K_T', 'K_S', 'mu')


def table_consistency_problems(rows, rel_tol=0.1):
    """Return the list of failure messages, one per broken rule, empty if the rows pass."""
    P = np.array([r['P'] for r in rows], dtype=float)
    T = np.array([r['T'] for r in rows], dtype=float)
    moduli = bool(rows) and 'K_T' in rows[0]
    q = {k: np.array([r[k] for r in rows], dtype=float) for k in QUANTITIES + (MODULI if moduli else ())}
    phase = np.array([r['phase'] for r in rows])
    problems = []

    def report(rule, bad, what, idx):
        if bad.any():
            k = idx[np.flatnonzero(bad)[0]]
            problems.append(f"[{rule}] {bad.sum()} rows with {what} (first at P = {P[k]:.6e} Pa, T = {T[k]:.2f} K, {phase[k]})")

    all_rows = np.arange(len(rows))
    finite = np.isfinite(np.column_stack([q[k] for k in QUANTITIES + (('K_T', 'K_S') if moduli else ())])).all(axis=1)
    report('finite', ~finite, "NaN or infinite values", all_rows)

    with np.errstate(invalid='ignore', divide='ignore'):
        bad = finite & (~(q['cv'] > 0) | ~(q['cp'] >= q['cv']))
        report('heat_capacities', bad, "C_V <= 0 or C_P < C_V", all_rows)

        lhs, rhs = q['nabla_ad'] * q['rho'] * q['cp'], P * q['alpha']
        bad = finite & ~(np.abs(lhs - rhs) <= 1e-6 * np.abs(rhs))
        report('nabla_ad', bad, "nabla_ad != P alpha / (rho C_P)", all_rows)

        if moduli:
            bad = finite & (~(q['K_T'] > 0) | ~(np.abs(q['K_S'] * q['cv'] - q['K_T'] * q['cp']) <= 1e-6 * np.abs(q['K_T'] * q['cp'])))
            report('bulk_moduli', bad, "K_T <= 0 or K_S / K_T != C_P / C_V", all_rows)
            bad = ~np.isnan(q['mu']) & ~(q['mu'] > 0)
            report('shear_modulus', bad, "mu <= 0", all_rows)

        # Along isobars, in order of rising T
        o = np.lexsort((T, P))
        Po, To, ph, fo = P[o], T[o], phase[o], finite[o]
        rho, s, u, cp, alpha = (q[k][o] for k in ('rho', 's', 'u', 'cp', 'alpha'))
        step = (Po[1:] == Po[:-1]) & fo[1:] & fo[:-1]
        bad = step & ~(np.diff(s) > 0)
        report('entropy_isobar', bad, "entropy not rising with T along the isobar", o)

        # Centered differences on interior rows whose two neighbors share the isobar and the phase
        inner = (Po[2:] == Po[:-2]) & (ph[2:] == ph[1:-1]) & (ph[:-2] == ph[1:-1]) & fo[2:] & fo[1:-1] & fo[:-2]
        Tc, Pc = To[1:-1], Po[1:-1]
        dlnT = np.log(To[2:] / To[:-2])
        ds, du, dv = s[2:] - s[:-2], u[2:] - u[:-2], 1 / rho[2:] - 1 / rho[:-2]
        pad = lambda m: np.concatenate([[False], m, [False]])
        alpha_fd = -np.log(rho[2:] / rho[:-2]) / (Tc * dlnT)
        bad = inner & ~(np.abs(alpha[1:-1] - alpha_fd) <= rel_tol * np.abs(alpha_fd) + 0.5 * np.abs(alpha[2:] - alpha[:-2]))
        report('alpha', pad(bad), f"alpha off -d ln(rho)/dT by more than {rel_tol:.0%}", o)
        cp_fd = ds / dlnT
        bad = inner & ~(np.abs(cp[1:-1] - cp_fd) <= rel_tol * np.abs(cp_fd) + 0.5 * np.abs(cp[2:] - cp[:-2]))
        report('cp', pad(bad), f"C_P off T ds/dT by more than {rel_tol:.0%}", o)
        work = Tc * ds - Pc * dv
        slack = 0.5 * np.abs((To[2:] - To[:-2]) * ds)
        bad = inner & ~(np.abs(du - work) <= rel_tol * (np.abs(Tc * ds) + np.abs(Pc * dv)) + slack)
        report('first_law', pad(bad), f"du off T ds - P dv by more than {rel_tol:.0%}", o)

        # Along isotherms, in order of rising P, within one phase
        o = np.lexsort((P, T))
        step = (T[o][1:] == T[o][:-1]) & (phase[o][1:] == phase[o][:-1]) & finite[o][1:] & finite[o][:-1]
        bad = step & ~(np.diff(q['rho'][o]) > 0)
        report('density_isotherm', bad, "density not rising with P along the isotherm", o)
        expands = (q['alpha'][o][1:] > 0) & (q['alpha'][o][:-1] > 0)
        bad = step & expands & ~(np.diff(q['s'][o]) < 0)
        report('entropy_isotherm', bad, "entropy not falling with P along the isotherm where alpha > 0", o)

        if moduli:
            To, ph, fo, K, rho = T[o], phase[o], finite[o], q['K_T'][o], q['rho'][o]
            inner = (To[2:] == To[:-2]) & (ph[2:] == ph[1:-1]) & (ph[:-2] == ph[1:-1]) & fo[2:] & fo[1:-1] & fo[:-2]
            K_fd = (P[o][2:] - P[o][:-2]) / np.log(rho[2:] / rho[:-2])
            bad = inner & ~(np.abs(K[1:-1] - K_fd) <= rel_tol * np.abs(K_fd) + 0.5 * np.abs(K[2:] - K[:-2]))
            report('K_T', pad(bad), f"K_T off dP/d ln(rho) by more than {rel_tol:.0%}", o)

    return problems


def check_table_consistency(rows, label, rel_tol=0.1, strict=True):
    """Raise ValueError if the rows of a table break any rule, otherwise print a one-line pass report. With strict=False the failures are printed and the table can still be written, for a table with known open defects."""
    problems = table_consistency_problems(rows, rel_tol=rel_tol)
    if problems:
        message = f"{label} fails the thermodynamic consistency check:\n  " + "\n  ".join(problems)
        if strict:
            raise ValueError(message)
        print("WARNING, written anyway (strict=False): " + message)
        return
    print(f"{label}: all {len(rows):,} rows pass the consistency check.")
