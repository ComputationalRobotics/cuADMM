#!/usr/bin/env python3
"""Report of the campaign: results/official_table.md (the requested columns), results/stage1_table.md,
results/stage2_table.md, and the workbook plato_cold_plain_admm_sigma_results.xlsx (every table, every run, the sigma
updates of the adaptive runs, the threshold times, the accuracy references, the inventory, the checks, the budget).
Uses scripts/analyze.py for the per-run metrics.  usage: make_report.py"""
import csv, glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

EXP = A.EXP
RES = A.RES
WB = os.path.join(EXP, 'plato_cold_plain_admm_sigma_results.xlsx')


def fmt_t(x, limit=None, reached=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return f'> {limit:g} s' if limit else 'n/a'
    return f'{x:.1f} s' if x >= 1 else f'{x:.3f} s'


def fe(x, p=1):
    if x is None or x == '':
        return 'n/a'
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return str(v)
    return f'{v:.{p}e}'


def cfg_label(c, sigma):
    return {'adaptive': 'adaptive (legacy)', 'fixed': 'fixed', 'tuned': 'fixed, tuned'}.get(c, c) + (f' σ={sigma:g}' if c == 'tuned' else '')


def official_md(rows):
    cols = ['Dataset', 'Sigma policy', 'Initial sigma', 'Final sigma', 'Sigma changes', 'Iterations', 'Time to eta 1e-4', 'Time to DIMACS 1e-6',
            'End-to-end time', 'eta', 'Max DIMACS', 'Objective error', 'X cone violation', 'S cone violation', 'Status']
    L = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for e in rows:
        lim = e['time_limit_s']
        reps = f" (median of {e['reps']})" if e['reps'] > 1 else ''
        L.append('| ' + ' | '.join([
            e['dataset'], cfg_label(e['config'], e['sigma_initial']), f"{e['sigma_initial']:g}",
            f"{e['sigma_final']:g}" if e.get('sigma_final') is not None else 'n/a', str(e.get('sigma_changes', 'n/a')), str(e.get('iterations', 'n/a')),
            fmt_t(e.get('eta_0.0001_time_s'), lim), fmt_t(e.get('dimacs_1e-06_time_s'), lim),
            (fmt_t(e.get('end_to_end_s')) + reps) if e.get('end_to_end_s') is not None else 'n/a',
            fe(e.get('eta')), fe(e.get('max_abs_dimacs')), fe(e.get('objective_error')) + (' (ref. uncertain)' if str(e.get('objective_check', '')).startswith('undecidable') else ''),
            fe(e.get('X_cone')), fe(e.get('S_cone')), e['status']]) + ' |')
    return '\n'.join(L)


def comparison_md(comp, summ, cap):
    t = lambda x: fmt_t(x, cap)
    sp = lambda x: f'{x:.2f}x' if x is not None else ''
    T = A.SNAP_TIMES
    L = ['| Dataset | Status, adaptive / fixed | Time to practical target, adaptive / fixed | Speedup (fixed / adaptive) | Time to strict target, adaptive / fixed | '
         'Iterations, adaptive / fixed | ms per iteration, adaptive / fixed | Max DIMACS at ' + ' / '.join(f'{x} s' for x in T) + ', adaptive; fixed | '
         'Max DIMACS of the returned iterate, adaptive / fixed | Adaptive final sigma (changes) | Winner |', '|' + '---|' * 11]
    for r in comp:
        snap = lambda w: ' / '.join(fe(r[f'{w}_dimacs_at_{x}s']) for x in T)
        L.append(f"| {r['dataset']} | {r['adaptive_status']} / {r['fixed_status']} | {t(r['adaptive_practical_s'])} / {t(r['fixed_practical_s'])} | "
                 f"{sp(r.get('practical_speedup_adaptive'))} | {t(r['adaptive_strict_s'])} / {t(r['fixed_strict_s'])} | {r['adaptive_iterations']} / {r['fixed_iterations']} | "
                 f"{r['adaptive_ms_per_iteration']:.2f} / {r['fixed_ms_per_iteration']:.2f} | {snap('adaptive')}; {snap('fixed')} | "
                 f"{fe(r['adaptive_max_dimacs'])} / {fe(r['fixed_max_dimacs'])} | {r['adaptive_sigma_final']:g} ({r['adaptive_sigma_changes']}) | {r['practical_winner']} |")
    g = lambda x: f'{x:.2f}x' if x is not None else 'n/a'
    L += ['', f"- Datasets: {summ['datasets']}. Practical target reached (validator-confirmed): adaptive {summ['practical_reached_adaptive']}, fixed "
          f"{summ['practical_reached_fixed']}; both {summ['practical_both']}, only adaptive {summ['practical_only_adaptive']}, only fixed "
          f"{summ['practical_only_fixed']}, neither {summ['practical_neither']}.",
          f"- Strict target (validator-confirmed max_abs_DIMACS <= 1e-6 and objective): adaptive {summ['strict_reached_adaptive']}, fixed {summ['strict_reached_fixed']}, both {summ['strict_both']}.",
          f"- Geometric-mean speedup of adaptive over fixed on the mutually validated instances: practical target {g(summ['gmean_speedup_practical_adaptive_over_fixed'])} "
          f"({summ['practical_both']} instances; adaptive faster on {summ['practical_both_adaptive_faster']}, fixed on {summ['practical_both_fixed_faster']}); "
          f"strict target {g(summ['gmean_speedup_strict_adaptive_over_fixed'])} ({summ['strict_both']} instances).",
          f"- Neither reached the practical target on {summ['practical_neither']} datasets: after the same wall time the returned iterate has the lower external max DIMACS error "
          f"with adaptive on {summ['neither_lower_dimacs_adaptive']} and with fixed on {summ['neither_lower_dimacs_fixed']} (geometric mean of fixed / adaptive: "
          f"{g(summ['gmean_dimacs_ratio_fixed_over_adaptive_neither'])}).",
          '- Convergence over equal wall time (lower external max DIMACS error, adaptive vs fixed): ' +
          '; '.join(f"at {x} s {summ[f'lower_dimacs_at_{x}s_adaptive']} vs {summ[f'lower_dimacs_at_{x}s_fixed']}" for x in A.SNAP_TIMES) + '.',
          f"- Cost per iteration: geometric mean of adaptive / fixed {g(summ['gmean_iteration_time_ratio_adaptive_over_fixed'])}.",
          f"- Timeouts (no practical target within the cap): adaptive {summ['timeouts_adaptive']}, fixed {summ['timeouts_fixed']}.",
          '- Status semantics: the practical target is eta <= 1e-4 with every normalized cone violation <= 1e-4; the strict target adds max_abs_DIMACS '
          '<= 1e-6 and a relative objective error <= 1e-6 (both validator-confirmed). A run whose solver stopped at max_abs_DIMACS <= 1e-6 but whose '
          'objective error exceeds 1e-6 is PRACTICAL_VALIDATED. Times are solve-clock seconds at validator confirmation, medians over repetitions.']
    return '\n'.join(L)


def write_workbook(path, sheets):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    for name, blocks in sheets:
        ws = wb.create_sheet(name[:31])
        row = 1
        for title, rows, cols in blocks:
            ws.cell(row=row, column=1, value=title).font = Font(bold=True, size=12)
            row += 1
            if not rows:
                ws.cell(row=row, column=1, value='(none)')
                row += 2
                continue
            cols = cols or list(dict.fromkeys(k for r in rows for k in r))
            for j, h in enumerate(cols):
                c = ws.cell(row=row, column=j + 1, value=h)
                c.font = Font(bold=True)
                c.fill = PatternFill('solid', fgColor='DDEBF7')
                c.alignment = Alignment(wrap_text=True, vertical='top')
            row += 1
            for d in rows:
                for j, h in enumerate(cols):
                    v = d.get(h)
                    if isinstance(v, (list, dict, tuple)):
                        v = json.dumps(v)
                    if isinstance(v, float) and not math.isfinite(v):
                        v = str(v)
                    c = ws.cell(row=row, column=j + 1, value=v)
                    if isinstance(v, float):
                        c.number_format = '0.000E+00' if (v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e6)) else '0.000'
                    if isinstance(v, str) and len(v) > 60:
                        c.alignment = Alignment(wrap_text=True, vertical='top')
                row += 1
            row += 1
        for j in range(1, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(j)].width = 40 if j == 1 else 16
        ws.freeze_panes = 'B3'
    wb.save(path)


def read_csv(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def sigma_updates(stage):
    out = []
    for d in sorted(glob.glob(os.path.join(EXP, 'runs', stage, '*'))):
        for u in A.read_csv(os.path.join(d, 'sigma_log.csv')):
            out.append(dict(run_id=os.path.basename(d), **u))
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    sheets = []
    readme = [dict(item=k, value=v) for k, v in [
        ('experiment', 'PLATO / Kocvara sparse SDPs, cold-start plain ADMM, adaptive vs fixed sigma (2026-09-29)'),
        ('protocol', 'PROTOCOL.md (predeclared) and accuracy_protocol.md (criteria)'),
        ('MOSEK', 'not run, no MOSEK solution used (logs/job_inspection_and_mosek_cleanup.md)'),
        ('status source', 'the independent revalidation of the returned iterate (scripts/dimacs_validate.py)'),
        ('times', 'solve clock at validator confirmation; end-to-end = load + CUDA init + solver init + solve; medians over repetitions'),
    ]]
    s3 = s2 = s1 = None
    if glob.glob(os.path.join(EXP, 'runs', 'stage3', '*', 'run.json')):
        runs3, s3 = A.stage3()
        open(os.path.join(RES, 'official_table.md'), 'w').write(official_md(s3) + '\n')
        comp, summ = A.compare(s3)
        open(os.path.join(RES, 'comparison.md'), 'w').write(comparison_md(comp, summ, s3[0]['time_limit_s']) + '\n')
        sheets.append(('Official table', [('Stage 3: one row per dataset and configuration (medians over repetitions)', s3, None)]))
        sheets.append(('Adaptive vs fixed', [('per dataset', comp, None), ('aggregate', [dict(measure=k, value=v) for k, v in summ.items()], None)]))
        sheets.append(('Official runs', [('Stage 3: every run', runs3, None)]))
        sheets.append(('Official sigma updates', [('every sigma change of the adaptive runs', sigma_updates('stage3'), None)]))
    if glob.glob(os.path.join(EXP, 'runs', 'stage2', '*', 'run.json')):
        s2, sel = A.stage2()
        sheets.append(('Pilot selection', [('Stage 2 selection (frozen rule)', s2, None),
                                           ('selected', [dict(tuned_sigma=sel['tuned_sigma'], tied=sel['tied'], rule=sel['rule'])] if sel else [], None)]))
        sheets.append(('Pilot runs', [('Stage 2: every run', A.load_stage('stage2'), None)]))
    if glob.glob(os.path.join(EXP, 'runs', 'stage1', '*', 'run.json')):
        s1, iv = A.stage1()
        sheets.append(('Smoke runs', [('Stage 1: every run (120 s)', s1, None)]))
        c1, sm1 = A.compare(A.aggregate('stage1')[1])
        sheets.append(('Smoke adaptive vs fixed', [('Stage 1 (120 s; not the official comparison)', c1, None),
                                                   ('aggregate', [dict(measure=k, value=v) for k, v in sm1.items()], None)]))
        if iv:
            sheets.append(('Validation intervals', [('per dataset (rule of PROTOCOL.md)', [dict(dataset=k, interval=v) for k, v in iv.items()], None)]))
    sheets.append(('References', [('reference objectives (accuracy_protocol.md section 4)', read_csv(os.path.join(EXP, 'data', 'reference_objectives.csv')), None),
                                  ('historical DIMACS errors (PLATO logs)', read_csv(os.path.join(EXP, 'data', 'historical_dimacs.csv')), None)]))
    sheets.append(('Datasets', [('inventory', read_csv(os.path.join(EXP, 'dataset_inventory.csv')), None),
                                ('conversion checks', read_csv(os.path.join(EXP, 'data', 'conversion_checks.csv')), None)]))
    budget = [dict(line=l.rstrip()) for l in open(os.path.join(EXP, 'logs', 'budget.log'))] if os.path.exists(os.path.join(EXP, 'logs', 'budget.log')) else []
    sheets.append(('Budget', [('budget ledger (logs/budget.log)', budget, None)]))
    sheets.insert(0, ('README', [('about this workbook', readme, None)]))
    write_workbook(WB, sheets)
    print(f'wrote {WB}')


if __name__ == '__main__':
    main()
