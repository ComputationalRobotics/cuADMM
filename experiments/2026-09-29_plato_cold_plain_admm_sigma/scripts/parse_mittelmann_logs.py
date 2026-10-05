#!/usr/bin/env python3
"""Historical accuracy on the 8 Kocvara problems benchmarked on the PLATO sparse-SDP page: the six DIMACS errors that
each solver reports at the end of its log (plato.asu.edu/ftp/sparse_logs/<solver>/<problem>.<ext>, downloaded to
~/cuadmm-data/plato_kocvara/docs/sparse_logs/). Writes data/historical_dimacs.csv."""
import csv, glob, os, re, sys
LOGS = os.path.expanduser('~/cuadmm-data/plato_kocvara/docs/sparse_logs')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'historical_dimacs.csv')
num = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
rows = []
for path in sorted(glob.glob(os.path.join(LOGS, '*', '*.*'))):
    if path.endswith('.html'):
        continue
    solver, fn = os.path.basename(os.path.dirname(path)), os.path.basename(path)
    problem = fn.rsplit('.', 1)[0]
    txt = open(path, errors='replace').read()
    errs, fmt = {}, ''
    # MOSEK / SDPA: 'Error1: v' ... 'Error6: v'
    for k in range(1, 7):
        m = re.findall(r'(?i)err(?:or)?\s*' + str(k) + r'\s*[:=]\s*' + num, txt)
        if m:
            errs[k], fmt = float(m[-1]), 'ErrorK: lines'
    if not errs:
        # COPT 'DIMACS errors: 6 values', CSDP 'DIMACS error measures: 6 values', SDPT3 'DIMACS errors: 6 values',
        # SeDuMi 'DIMACS error measures' + header line + 6 values, HDSDP 'DIMACS error metric:' + 6 values
        m = re.findall(r'(?i)DIMACS error(?:s| measures| metric)\s*:?\s*(?:\n\s*PInf[^\n]*)?\s*' + r'\s+'.join([num] * 6), txt)
        if m:
            errs, fmt = {k + 1: float(v) for k, v in enumerate(m[-1])}, 'six-value line'
    if not errs:
        # cuLoRADS: three DIMACS quantities in its final table (primal infeasibility, dual infeasibility, gap)
        vals = {name: re.findall(r'(?i)' + name + r' \(DIMACS\)\s*\S\s*' + num, txt) for name in
                ('primal infeasibility', 'dual infeasibility', 'primal dual gap')}
        if all(vals.values()):
            errs = {1: float(vals['primal infeasibility'][-1]), 3: float(vals['dual infeasibility'][-1]), 5: float(vals['primal dual gap'][-1])}
            fmt = 'cuLoRADS table (err1, err3, err5 only)'
    rows.append(dict(solver=solver, problem=problem, **{f'err{k}': errs.get(k) for k in range(1, 7)},
                     max_abs=max((abs(v) for v in errs.values()), default=None), n_found=len(errs), format=fmt, log=os.path.relpath(path, LOGS)))
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['solver']:9s} {r['problem']:8s} found {r['n_found']}  max|err| {r['max_abs'] if r['max_abs'] is not None else 'n/a':>12}  "
          + ' '.join(f"{r[f'err{k}']:.1e}" if r[f'err{k}'] is not None else '   n/a  ' for k in range(1, 7)))
