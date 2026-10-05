#!/usr/bin/env python3
"""Per-run metrics of the campaign (all stages) and the stage-specific outputs.

  analyze.py stage1   results/stage1_runs.csv, plans/validation_intervals.json (the interval rule of PROTOCOL.md),
                      results/budget_projection.md
  analyze.py stage2   results/stage2_pilot.csv, plans/tuned_sigma.json (the frozen selection rule)
  analyze.py stage3   results/stage3_runs.csv, results/stage3_table.csv/.md (medians over repetitions)

Per run: the outcome of the driver, the solver summary, the validation history (threshold times), the sigma log
(checks: fixed sigma never changes; the adaptive log is a consistent chain from sigma0 to final_sig), and the
independent revalidation of the returned and final iterates (postval_*.json; the status comes from it)."""
import csv, glob, gzip, json, math, os, statistics, sys

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(EXP, 'results')
TOLS = (1e-3, 1e-4, 1e-5, 1e-6)
INTERVALS = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
INV = {r['dataset']: r for r in csv.DictReader(open(os.path.join(EXP, 'dataset_inventory.csv')))}


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float('nan')


def read_csv(path):
    """rows of a CSV file, or of its gzipped copy (path + '.gz'); [] if neither exists"""
    if os.path.exists(path):
        return list(csv.DictReader(open(path)))
    if os.path.exists(path + '.gz'):
        with gzip.open(path + '.gz', 'rt') as f:
            return list(csv.DictReader(f))
    return []


def load_run(d):
    r = json.load(open(os.path.join(d, 'run.json')))
    s = None
    if os.path.exists(os.path.join(d, 'summary.json')):
        lines = [l for l in open(os.path.join(d, 'summary.json')) if l.strip()]
        s = json.loads(lines[-1]) if lines else None
    vh = read_csv(os.path.join(d, 'validation_history.csv'))
    sl = read_csv(os.path.join(d, 'sigma_log.csv'))
    pv = {w: json.load(open(os.path.join(d, f'postval_{w}.json'))) for w in ('returned', 'final') if os.path.exists(os.path.join(d, f'postval_{w}.json'))}
    return r, s, vh, sl, pv


def thresholds(vh):
    """first validation snapshot at or below each tolerance, for eta and max_abs_DIMACS: (iteration, solve-clock time
    when the validator confirmed it = snapshot time + its validation time)"""
    out = {}
    for key in ('eta', 'dimacs'):
        for tol in TOLS:
            hit = None
            for v in vh:
                val = max(fnum(v['primal_res']), fnum(v['dual_res']), fnum(v['relgap'])) if key == 'eta' else fnum(v['max_abs_dimacs'])
                if math.isfinite(val) and val <= tol:
                    hit = v
                    break
            out[f'{key}_{tol:g}_iter'] = int(hit['iter']) if hit else None
            out[f'{key}_{tol:g}_time_s'] = fnum(hit['time_s']) + fnum(hit['validation_s']) + fnum(hit.get('callback_s', 0) or 0) if hit else None
    return out


SNAP_TIMES = (60, 300, 1800)


def at_times(vh):
    """external eta and max_abs_DIMACS of the last validation confirmed at or before each snapshot time (a run that stopped
    earlier keeps its last value: its returned iterate does not change afterwards)"""
    out = {}
    for T in SNAP_TIMES:
        last = None
        for v in vh:
            if fnum(v['time_s']) + fnum(v['validation_s']) <= T:
                last = v
        out[f'eta_at_{T}s'] = max(fnum(last['primal_res']), fnum(last['dual_res']), fnum(last['relgap'])) if last else None
        out[f'dimacs_at_{T}s'] = fnum(last['max_abs_dimacs']) if last else None
    return out


def sigma_checks(r, s, sl):
    notes = []
    if s is None:
        return False, ['no summary']
    if r['policy'] == 'fixed':
        ok = s['sigma_changes'] == 0 and s['sigma_log_entries'] == 0 and len(sl) == 0 and s['final_sig'] == s['sigma0'] == r['sigma']
        if not ok:
            notes.append(f"FIXED SIGMA VIOLATED: changes {s['sigma_changes']}, log {s['sigma_log_entries']}, final {s['final_sig']}, sigma0 {s['sigma0']}")
    else:
        ok = len(sl) == s['sigma_log_entries']
        prev = s['sigma0']
        for u in sl:
            if abs(fnum(u['old_sigma']) - prev) > 1e-12 * max(1.0, abs(prev)):
                ok = False
                notes.append(f"sigma log chain broken at iteration {u['iter']}")
                break
            prev = fnum(u['new_sigma'])
        if abs(prev - s['final_sig']) > 1e-9 * max(1.0, abs(prev)):
            ok = False
            notes.append(f'sigma log ends at {prev}, final_sig {s["final_sig"]}')
        if len({int(u['iter']) for u in sl}) != s['sigma_changes']:
            ok = False
            notes.append(f"sigma log has {len({int(u['iter']) for u in sl})} distinct iterations, sigma_changes {s['sigma_changes']}")
    return ok, notes


def status_of(r, s, pv, practical_in_solver):
    if r['outcome'] == 'out_of_memory':
        return 'OUT_OF_MEMORY'
    if r['outcome'] in ('crashed', 'hung') or s is None:
        return 'NOT_VALIDATED'
    ret = pv.get('returned')
    if ret is None:
        return 'NOT_VALIDATED'
    st = ret['status']
    if st == 'NOT_VALIDATED' and s['stop_reason'] in ('time_limit', 'max_iter'):
        return 'TIMEOUT'
    if st != 'NOT_VALIDATED' and not practical_in_solver:
        return st  # the independent validator passes an iterate the solver did not validate: reported as such
    return st


def metrics(d):
    r, s, vh, sl, pv = load_run(d)
    m = dict(run_id=r['run_id'], stage=r['stage'], dataset=r['dataset'], config=r['config'], policy=r['policy'], sigma_initial=r['sigma'], rep=r['rep'],
             time_limit_s=r['time_limit_s'], validate_interval=r['validate_interval'], outcome=r['outcome'], exit_code=r['exit_code'], wall_s=r['wall_s'],
             host=r['host'], gpu=(r.get('gpu_before') or '').split(',')[1].strip() if r.get('gpu_before', '').count(',') > 2 else r.get('gpu_before'))
    if s:
        e2e = s['t_load_s'] + s['t_cuda_init_s'] + s['time_s']
        m.update(sigma_final=s['final_sig'], sigma_changes=s['sigma_changes'], sigma_log_entries=s['sigma_log_entries'], iterations=s['iterations'],
                 stop_reason=s['stop_reason'], solve_time_s=s['solve_time_s'], init_time_s=s['init_time_s'], load_s=s['t_load_s'], cuda_init_s=s['t_cuda_init_s'],
                 end_to_end_s=e2e, validation_time_s=s['validation_time_s'], validation_count=s['validation_count'],
                 validation_s_per_snapshot=s['validation_time_s'] / max(1, s['validation_count']),
                 iter_time_s=(s['solve_time_s'] - s['validation_time_s'] - s['callback_time_s']) / max(1, s['iterations']),
                 y_solves_per_iteration=s['y_solves_per_iteration'], plain_admm_only_build=s['plain_admm_only_build'],
                 practical_validated=s['practical_validated'], first_practical_iteration=s['first_practical_iteration'],
                 first_practical_time_s=s['first_practical_time_s'], strict_validated=s['strict_validated'],
                 first_strict_iteration=s['first_strict_iteration'], first_strict_time_s=s['first_strict_time_s'],
                 returned_best_external=s['returned_best_external'], peak_gpu_mem_mib=s.get('peak_gpu_mem_mib'),
                 lobpcg_calls=s['lobpcg_calls'], lobpcg_fallbacks=s['lobpcg_fallbacks'], aat_tiny_pivots=s['aat_tiny_pivots'],
                 aat_nonpositive_pivots=s['aat_nonpositive_pivots'], aat_nnz_L=s['aat_nnz_L'])
    m.update(thresholds(vh))
    m.update(at_times(vh))
    ok, notes = sigma_checks(r, s, sl)
    m['sigma_check_ok'] = ok
    if s and s['y_solves_per_iteration'] != 1.0 and s['iterations'] > 0:
        notes.append(f"y_solves_per_iteration = {s['y_solves_per_iteration']}")
    for w in ('returned', 'final'):
        p = pv.get(w)
        if not p:
            continue
        m.update({f'{w}_{k}': p[k] for k in ('status', 'eta', 'eta_p', 'eta_d', 'eta_g', 'max_abs_dimacs', 'X_cone', 'S_cone', 'Z_cone', 'objective_error',
                                                'objective_check', 'sdpa_objective', 'plato_class', 'min_eig_X', 'min_eig_S')})
        for i in range(6):
            m[f'{w}_dimacs{i + 1}'] = p['dimacs'][i]
    # agreement of the independent validator with the solver's own record of the returned iterate
    if s and pv.get('returned'):
        rec = 'best' if s['returned_best_external'] else 'final'
        c_md, p_md = s.get(f'{rec}_max_abs_dimacs'), pv['returned']['max_abs_dimacs']
        if c_md is not None and p_md is not None and math.isfinite(c_md) and math.isfinite(p_md):
            m['validator_diff_max_abs_dimacs'] = abs(c_md - p_md)
            if abs(c_md - p_md) > 1e-8 * max(abs(p_md), 1e-6):
                notes.append(f'validators disagree: solver {c_md:.3e}, independent {p_md:.3e}')
    m['status'] = status_of(r, s, pv, bool(s and s['practical_validated']))
    m['notes'] = '; '.join(notes)
    return m


def load_stage(stage):
    return [metrics(d) for d in sorted(glob.glob(os.path.join(EXP, 'runs', stage, '*'))) if os.path.exists(os.path.join(d, 'run.json'))]


def write_csv(path, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def stage1():
    rows = load_stage('stage1')
    plan = json.load(open(os.path.join(EXP, 'plans', 'stage1.json')))
    if len(rows) != len(plan['runs']):
        # the intervals come from the complete smoke stage only (PROTOCOL.md): nothing is written from a partial one
        print(f"stage 1 incomplete: {len(rows)} of {len(plan['runs'])} runs; no validation intervals written")
        return rows, None
    write_csv(os.path.join(RES, 'stage1_runs.csv'), rows)
    iv, lines = {}, []
    for d in sorted({r['dataset'] for r in rows}, key=lambda x: int(INV[x]['cuadmm_vec_len'])):
        rs = [r for r in rows if r['dataset'] == d and r.get('iterations')]
        if not rs:
            continue
        v = max(r['validation_s_per_snapshot'] for r in rs)
        t = min(r['iter_time_s'] for r in rs)
        iv[d] = next((I for I in INTERVALS if v / (I * t) <= 0.10), INTERVALS[-1])
        lines.append(f"| {d} | {t * 1e3:.3f} | {v:.3f} | {iv[d]} | {v / (iv[d] * t):.1%} |")
    ivp = os.path.join(EXP, 'plans', 'validation_intervals.json')
    if os.path.exists(ivp):
        # frozen once written (the official plans were built from it): only check that the rule reproduces it
        old_iv = json.load(open(ivp))
        if old_iv != iv:
            raise SystemExit(f'validation intervals differ from the frozen {ivp}: {old_iv} vs {iv}')
    else:
        json.dump(iv, open(ivp, 'w'), indent=1)
    print('\n'.join(['| dataset | ms per iteration | s per validation | interval | overhead |', '|---|---|---|---|---|'] + lines))
    return rows, iv


def practical_time(r):
    """solve-clock time to the practical target, confirmed by both validators (None if not reached)"""
    if r.get('practical_validated') and r.get('returned_status') in ('PRACTICAL_VALIDATED', 'STRICT_VALIDATED'):
        return r['first_practical_time_s']
    return None


def stage2():
    rows = load_stage('stage2')
    write_csv(os.path.join(RES, 'stage2_pilot.csv'), rows)
    plan = json.load(open(os.path.join(EXP, 'plans', 'stage2.json')))
    limit, pilot = plan['limit_s'], plan['pilot']
    table = []
    for cfgname in ['adaptive'] + [f'fixed_s{g:g}' for g in plan['grid']]:
        rs = {r['dataset']: r for r in rows if r['config'] == cfgname}
        times = {d: practical_time(rs[d]) if d in rs else None for d in pilot}
        par2 = [t if t is not None else 2 * limit for t in times.values()]
        sigma = rs[pilot[0]]['sigma_initial'] if pilot[0] in rs else None
        table.append(dict(config=cfgname, sigma=sigma, reached=sum(t is not None for t in times.values()), complete=len(rs) == len(pilot),
                          geo_mean_par2_s=math.exp(sum(math.log(max(t, 1e-3)) for t in par2) / len(par2)), **{f'{d}_s': times[d] for d in pilot}))
    write_csv(os.path.join(RES, 'stage2_selection.csv'), table)
    fixed = [t for t in table if t['config'] != 'adaptive']
    if not all(t['complete'] for t in fixed):
        print('stage 2 incomplete: no selection')
        return table, None
    # frozen rule: most datasets reached, then the smallest geometric mean (PAR2), ties within 5 % -> sigma closest to 1
    best_n = max(t['reached'] for t in fixed)
    cand = [t for t in fixed if t['reached'] == best_n]
    g0 = min(t['geo_mean_par2_s'] for t in cand)
    tied = [t for t in cand if t['geo_mean_par2_s'] <= 1.05 * g0]
    pick = min(tied, key=lambda t: (abs(math.log10(t['sigma'])), t['geo_mean_par2_s']))
    sel = dict(tuned_sigma=pick['sigma'], rule='max datasets reached; min geometric mean of the time to the practical target (timeout = 2 x limit); '
               'ties within 5 % -> sigma closest to 1 (log scale)', reached=best_n, geo_mean_par2_s=pick['geo_mean_par2_s'],
               tied=[t['sigma'] for t in tied], table=table)
    json.dump(sel, open(os.path.join(EXP, 'plans', 'tuned_sigma.json'), 'w'), indent=1)
    for t in table:
        print(f"{t['config']:12s} reached {t['reached']}/{len(pilot)} geo-mean(PAR2) {t['geo_mean_par2_s']:8.1f} s  " +
              ' '.join(f"{d}={t[d + '_s']:.1f}" if t[d + '_s'] is not None else f'{d}=TIMEOUT' for d in pilot))
    print(f"selected tuned sigma = {pick['sigma']:g} (tied: {sel['tied']})")
    return table, sel


def med(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.median(xs) if xs else None


def stage3():
    rows, out = aggregate('stage3')
    write_csv(os.path.join(RES, 'stage3_runs.csv'), rows)
    write_csv(os.path.join(RES, 'stage3_table.csv'), out)
    comp, summ = compare(out)
    write_csv(os.path.join(RES, 'stage3_comparison.csv'), comp)
    json.dump(summ, open(os.path.join(RES, 'stage3_comparison_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))
    return rows, out


def aggregate(stage):
    rows = load_stage(stage)
    groups = {}
    for r in rows:
        groups.setdefault((r['dataset'], r['config']), []).append(r)
    out = []
    for (d, c), rs in sorted(groups.items(), key=lambda kv: (int(INV[kv[0][0]]['cuadmm_vec_len']), kv[0][0], kv[0][1])):
        rs.sort(key=lambda r: r['rep'])
        r1 = rs[0]
        its = {r.get('iterations') for r in rs if r['outcome'] == 'completed' and r.get('stop_reason') not in ('time_limit',)}
        e = dict(dataset=d, config=c, policy=r1['policy'], sigma_initial=r1['sigma_initial'], sigma_final=r1.get('sigma_final'),
                 sigma_changes=r1.get('sigma_changes'), iterations=r1.get('iterations'), reps=len(rs), iterations_equal_across_reps=len(its) <= 1,
                 time_limit_s=r1['time_limit_s'], stop_reason=r1.get('stop_reason'),
                 in_sample=d in json.load(open(os.path.join(EXP, 'plans', 'stage2.json')))['pilot'] if c == 'tuned' else None)
        for tol in TOLS:
            for key in ('eta', 'dimacs'):
                vals = [r.get(f'{key}_{tol:g}_time_s') for r in rs]
                e[f'{key}_{tol:g}_time_s'] = med(vals) if all(v is not None for v in vals) else None
                e[f'{key}_{tol:g}_iter'] = r1.get(f'{key}_{tol:g}_iter')
        e['practical_time_s'] = med([practical_time(r) for r in rs]) if all(practical_time(r) is not None for r in rs) else None
        e['end_to_end_s'] = med([r.get('end_to_end_s') for r in rs])
        e['solve_time_s'] = med([r.get('solve_time_s') for r in rs])
        e['setup_s'] = med([(r.get('load_s') or 0) + (r.get('cuda_init_s') or 0) + (r.get('init_time_s') or 0) for r in rs if r.get('iterations') is not None])
        e['validation_time_s'] = med([r.get('validation_time_s') for r in rs])
        e['first_practical_iteration'] = r1.get('first_practical_iteration')
        e['first_strict_iteration'] = r1.get('first_strict_iteration')
        e['iter_time_ms'] = 1e3 * r1['iter_time_s'] if r1.get('iter_time_s') is not None else None
        for T in SNAP_TIMES:
            e[f'eta_at_{T}s'] = r1.get(f'eta_at_{T}s')
            e[f'dimacs_at_{T}s'] = r1.get(f'dimacs_at_{T}s')
        e['end_to_end_min_s'] = min((r.get('end_to_end_s') for r in rs if r.get('end_to_end_s') is not None), default=None)
        e['end_to_end_max_s'] = max((r.get('end_to_end_s') for r in rs if r.get('end_to_end_s') is not None), default=None)
        for k in ('eta', 'max_abs_dimacs', 'objective_error', 'X_cone', 'S_cone', 'Z_cone', 'plato_class', 'objective_check', 'sdpa_objective',
                  'dimacs1', 'dimacs2', 'dimacs3', 'dimacs4', 'dimacs5', 'dimacs6'):
            e[k] = r1.get(f'returned_{k}')
            e['final_' + k] = r1.get(f'final_{k}')
        sts = [r['status'] for r in rs]
        e['status'] = r1['status'] if len(set(sts)) == 1 else f"{r1['status']} (reps: {', '.join(sts)})"
        e['notes'] = '; '.join(sorted({r['notes'] for r in rs if r['notes']}))
        out.append(e)
    return rows, out


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def compare(table):
    """adaptive vs fixed per dataset and in aggregate. A target counts as reached only when the independent
    revalidation agrees (status PRACTICAL_VALIDATED or STRICT_VALIDATED for the practical target, STRICT_VALIDATED
    for the strict one). Speedup = fixed time / adaptive time (> 1: adaptive faster). Where neither configuration
    reaches a target, the external max_abs_DIMACS of the returned iterates after the same wall time decides; no winner is
    declared from internal residuals."""
    by = {}
    for e in table:
        by.setdefault(e['dataset'], {})[e['config']] = e
    rows = []
    for d in sorted(by, key=lambda x: (int(INV[x]['cuadmm_vec_len']), x)):
        a, f = by[d].get('adaptive'), by[d].get('fixed')
        if not (a and f):
            continue
        base = lambda e: e['status'].split(' ')[0]
        prac = lambda e: base(e) in ('PRACTICAL_VALIDATED', 'STRICT_VALIDATED') and e.get('practical_time_s') is not None
        strict = lambda e: base(e) == 'STRICT_VALIDATED' and e.get('dimacs_1e-06_time_s') is not None
        r = dict(dataset=d, vec_len=int(INV[d]['cuadmm_vec_len']), m=int(INV[d]['constraints_m']),
                 adaptive_status=a['status'], fixed_status=f['status'],
                 adaptive_practical_s=a.get('practical_time_s') if prac(a) else None, fixed_practical_s=f.get('practical_time_s') if prac(f) else None,
                 adaptive_strict_s=a.get('dimacs_1e-06_time_s') if strict(a) else None, fixed_strict_s=f.get('dimacs_1e-06_time_s') if strict(f) else None,
                 adaptive_end_to_end_s=a.get('end_to_end_s'), fixed_end_to_end_s=f.get('end_to_end_s'),
                 adaptive_iterations=a.get('iterations'), fixed_iterations=f.get('iterations'),
                 adaptive_max_dimacs=a.get('max_abs_dimacs'), fixed_max_dimacs=f.get('max_abs_dimacs'),
                 adaptive_eta=a.get('eta'), fixed_eta=f.get('eta'),
                 adaptive_objective_error=a.get('objective_error'), fixed_objective_error=f.get('objective_error'),
                 adaptive_sigma_final=a.get('sigma_final'), adaptive_sigma_changes=a.get('sigma_changes'),
                 adaptive_ms_per_iteration=a.get('iter_time_ms'), fixed_ms_per_iteration=f.get('iter_time_ms'),
                 adaptive_reps=a['reps'], fixed_reps=f['reps'])
        for T in SNAP_TIMES:
            r[f'adaptive_dimacs_at_{T}s'] = a.get(f'dimacs_at_{T}s')
            r[f'fixed_dimacs_at_{T}s'] = f.get(f'dimacs_at_{T}s')
        pa, pf = r['adaptive_practical_s'], r['fixed_practical_s']
        if pa is not None and pf is not None:
            r['practical_speedup_adaptive'] = pf / pa if pa > 0 else None
            r['practical_winner'] = 'adaptive' if pa < pf else ('fixed' if pf < pa else 'tie')
        elif pa is not None or pf is not None:
            r['practical_winner'] = 'adaptive (only one reached)' if pa is not None else 'fixed (only one reached)'
        else:
            da, df = r['adaptive_max_dimacs'], r['fixed_max_dimacs']
            r['practical_winner'] = ('neither reached; lower external max DIMACS at the cap: ' +
                                     ('adaptive' if da < df else ('fixed' if df < da else 'tie'))) if (da is not None and df is not None) else 'neither reached'
        sa, sf = r['adaptive_strict_s'], r['fixed_strict_s']
        if sa is not None and sf is not None:
            r['strict_speedup_adaptive'] = sf / sa if sa > 0 else None
            r['strict_winner'] = 'adaptive' if sa < sf else ('fixed' if sf < sa else 'tie')
        else:
            r['strict_winner'] = ('adaptive (only one reached)' if sa is not None else ('fixed (only one reached)' if sf is not None else 'neither reached'))
        rows.append(r)
    n = len(rows)
    both_p = [r for r in rows if r['adaptive_practical_s'] is not None and r['fixed_practical_s'] is not None]
    both_s = [r for r in rows if r['adaptive_strict_s'] is not None and r['fixed_strict_s'] is not None]
    neither_p = [r for r in rows if r['adaptive_practical_s'] is None and r['fixed_practical_s'] is None]
    summ = dict(datasets=n,
                practical_reached_adaptive=sum(r['adaptive_practical_s'] is not None for r in rows),
                practical_reached_fixed=sum(r['fixed_practical_s'] is not None for r in rows),
                strict_reached_adaptive=sum(r['adaptive_strict_s'] is not None for r in rows),
                strict_reached_fixed=sum(r['fixed_strict_s'] is not None for r in rows),
                practical_both=len(both_p), practical_only_adaptive=sum(r['adaptive_practical_s'] is not None and r['fixed_practical_s'] is None for r in rows),
                practical_only_fixed=sum(r['fixed_practical_s'] is not None and r['adaptive_practical_s'] is None for r in rows), practical_neither=len(neither_p),
                strict_both=len(both_s),
                gmean_speedup_practical_adaptive_over_fixed=gmean([r['practical_speedup_adaptive'] for r in both_p]),
                gmean_speedup_strict_adaptive_over_fixed=gmean([r['strict_speedup_adaptive'] for r in both_s]),
                gmean_end_to_end_ratio_fixed_over_adaptive_on_practical_both=gmean([r['fixed_end_to_end_s'] / r['adaptive_end_to_end_s'] for r in both_p]),
                practical_both_adaptive_faster=sum(r['practical_winner'] == 'adaptive' for r in both_p),
                practical_both_fixed_faster=sum(r['practical_winner'] == 'fixed' for r in both_p),
                neither_lower_dimacs_adaptive=sum(r['practical_winner'].endswith('adaptive') for r in neither_p),
                neither_lower_dimacs_fixed=sum(r['practical_winner'].endswith('fixed') for r in neither_p),
                gmean_dimacs_ratio_fixed_over_adaptive_neither=gmean([r['fixed_max_dimacs'] / r['adaptive_max_dimacs'] for r in neither_p
                                                                      if r['fixed_max_dimacs'] and r['adaptive_max_dimacs']]),
                # convergence over equal wall time: datasets on which adaptive / fixed has the lower external max DIMACS
                **{f'lower_dimacs_at_{T}s_{w}': sum(1 for r in rows if r[f'adaptive_dimacs_at_{T}s'] is not None and r[f'fixed_dimacs_at_{T}s'] is not None and
                                                     ((r[f'adaptive_dimacs_at_{T}s'] < r[f'fixed_dimacs_at_{T}s']) == (w == 'adaptive')) and
                                                     r[f'adaptive_dimacs_at_{T}s'] != r[f'fixed_dimacs_at_{T}s'])
                   for T in SNAP_TIMES for w in ('adaptive', 'fixed')},
                gmean_iteration_time_ratio_adaptive_over_fixed=gmean([r['adaptive_ms_per_iteration'] / r['fixed_ms_per_iteration'] for r in rows
                                                                      if r['adaptive_ms_per_iteration'] and r['fixed_ms_per_iteration']]),
                timeouts_adaptive=sum(r['adaptive_status'].startswith('TIMEOUT') for r in rows),
                timeouts_fixed=sum(r['fixed_status'].startswith('TIMEOUT') for r in rows))
    return rows, summ


if __name__ == '__main__':
    os.makedirs(RES, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[2] == '--compare':
        # the adaptive-vs-fixed comparison of any stage (Stage 1: the smoke runs)
        comp, summ = compare(aggregate(sys.argv[1])[1])
        write_csv(os.path.join(RES, f'{sys.argv[1]}_comparison.csv'), comp)
        print(json.dumps(summ, indent=1))
    else:
        {'stage1': stage1, 'stage2': stage2, 'stage3': stage3}[sys.argv[1]]()
