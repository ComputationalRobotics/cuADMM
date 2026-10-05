#!/usr/bin/env python3
"""Reference objectives for the relative objective error (SDPA convention, the value of the .dat-s file).

The published PENSDP values (data/published_objectives.json) are compared with independent solutions of the SAME
files: Clarabel on the converted data (data/clarabel_reference.csv, small instances) and MOSEK's primal objective in
Mittelmann's PLATO logs (the 8 benchmarked instances; MOSEK was not run here). Where the ratio is a power of ten
(within 1%), the published value is in other units than the file and is multiplied by that power; a family whose other
members all need the same factor passes it to a member without an independent solution ("inferred"). The reference
uncertainty is the resolution of the published digits (0 for "exact value"), raised to the relative difference to
MOSEK only where MOSEK's own max |DIMACS error| in that log is <= 1e-6 (a reliable solution); the Kocvara page warns
that the large problems' values may differ between codes in the 5th-6th digit.
Writes data/reference_objectives.csv and data/reference_objectives.json."""
import csv, json, math, os, re

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pub = json.load(open(os.path.join(EXP, 'data', 'published_objectives.json')))
cla = {r['problem']: r for r in csv.DictReader(open(os.path.join(EXP, 'data', 'clarabel_reference.csv')))}
LOGS = os.path.expanduser('~/cuadmm-data/plato_kocvara/docs/sparse_logs/MOSEK')
mosek = {}
for f in os.listdir(LOGS):
    m = re.findall(r'Primal\.\s*obj:\s*([-+]?\d[\d.]*e[-+]?\d+)', open(os.path.join(LOGS, f), errors='replace').read())
    if m:
        mosek[f.rsplit('.', 1)[0]] = float(m[-1])
hist = {r['problem']: float(r['max_abs']) for r in csv.DictReader(open(os.path.join(EXP, 'data', 'historical_dimacs.csv')))
        if r['solver'] == 'MOSEK' and r['max_abs']}


def power10(x, ref):
    if not x or not ref:
        return None
    r = abs(x / ref)
    k = round(math.log10(r))
    return k if abs(r / 10 ** k - 1) < 0.01 else None


rows = {}
for name, p in pub.items():
    c = float(cla[name]['sdpa_convention']) if name in cla and cla[name]['sdpa_convention'] else None
    mo = mosek.get(name)
    ks = {k for k in (power10(c, p['objective']), power10(mo, p['objective'])) if k is not None}
    rows[name] = dict(problem=name, published=p['objective'], published_text=p['text'], exact=p['exact'], digits=p['digits'],
                      clarabel=c, mosek_plato_log=mo, factor_exponent=ks.pop() if len(ks) == 1 else None,
                      factor_source='independent solution' if (c or mo) else None)
# family inference for members without an independent solution
for name, r in rows.items():
    if r['factor_exponent'] is None:
        fam = re.sub(r'-?\d+$', '', name)
        others = {o['factor_exponent'] for n, o in rows.items() if re.sub(r'-?\d+$', '', n) == fam and o['factor_exponent'] is not None and n != name}
        big = [o['factor_exponent'] for n, o in rows.items() if re.sub(r'-?\d+$', '', n) == fam and n != name and int(re.findall(r'\d+$', n)[0]) > 1
               and o['factor_exponent'] is not None]
        if len(set(big)) == 1:
            r['factor_exponent'], r['factor_source'] = big[0], 'inferred from the family (' + fam + ' 2-5)'
        elif len(others) == 1:
            r['factor_exponent'], r['factor_source'] = others.pop(), 'inferred from the family'
for r in rows.values():
    k = r['factor_exponent'] or 0
    ref = r['published'] * 10 ** k
    res = 0.0 if r['exact'] else 0.5 * 10 ** (math.floor(math.log10(abs(r['published']))) - (r['digits'] - 1)) * 10 ** k
    dm = abs(ref - r['mosek_plato_log']) if r['mosek_plato_log'] else 0.0
    dc = abs(ref - r['clarabel']) if r['clarabel'] else None
    reliable = r['mosek_plato_log'] is not None and hist.get(r['problem'], 1.0) <= 1e-6 and not r['exact']
    unc = max(res, dm if reliable else 0.0) / (1 + abs(ref))
    r['mosek_max_abs_dimacs'] = hist.get(r['problem'])
    r['mosek_used_for_uncertainty'] = reliable
    r.update(reference=ref, reference_resolution_rel=res / (1 + abs(ref)), diff_to_mosek_rel=dm / (1 + abs(ref)) if r['mosek_plato_log'] else None,
             diff_to_clarabel_rel=dc / (1 + abs(ref)) if dc is not None else None, reference_uncertainty_rel=unc,
             objective_check_decidable_at_1e6=unc <= 1e-6)
out = list(rows.values())
with open(os.path.join(EXP, 'data', 'reference_objectives.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
json.dump(rows, open(os.path.join(EXP, 'data', 'reference_objectives.json'), 'w'), indent=1)
for r in sorted(out, key=lambda r: r['problem']):
    print(f"{r['problem']:8s} published {r['published_text']:>14s}  x10^{r['factor_exponent'] or 0} ({r['factor_source'] or 'none needed'})  ref {r['reference']:.9g}  "
          f"clarabel {r['clarabel']}  mosek {r['mosek_plato_log']}  uncertainty {r['reference_uncertainty_rel']:.1e}  decidable@1e-6 {r['objective_check_decidable_at_1e6']}")
