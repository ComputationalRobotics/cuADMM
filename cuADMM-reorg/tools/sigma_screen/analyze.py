#!/usr/bin/env python3
"""analyze.py: the tables and the gate checks of the Phase-4b CPU screen (plan Section 3.5, Phase 4b).

usage: analyze.py [--root DIR] [--costs CSV] [--out-prefix NAME] <specfile> [<specfile> ...]
Reads runs/<name>/{summary.json, sigma_log.csv, validation_history.csv.gz} of every spec of the files (the format of
make_specs.py) and writes results/<prefix>_runs.csv (one row per run), results/<prefix>_settings.csv (one row per
setting x sigma_0) and results/<prefix>_report.md (the tables and the gate verdicts). Nothing is re-simulated.

Definitions
  solved         the run reached the strict stage within the cap: a validated snapshot with max|DIMACS| <= 1e-6
                 (summary first_strict_iteration in (0, cap]); it_practical: the first validated snapshot (1e-4)
  gpu_s_est      iterations x H200 seconds per iteration + snapshots x H200 seconds per snapshot of the PLATO
                 cold-start campaign (stage 3, fixed sigma, repetition 1; --costs): an estimate of what the iteration
                 count would cost on the GPU, not a measurement; an unsolved run is censored at the cap (the estimate of
                 the cap's iterations), a paused run is reported but excluded from the means
  tv_log2        sum of |log2(new / old)| over the sigma log (the plan's total variation of sigma)
  last_change    the iteration of the last sigma change; freeze_at / freeze_reason from the summary (balance_freeze)
  spike          for a change at iteration k: E_post / E_pre with E the long-horizon error of a validation snapshot
                 (max of |DIMACS 1-6|, eta_p, eta_d, eta_g and the X / dual / S cone violations; non-finite -> inf),
                 E_pre the last snapshot with iter <= k, E_post the largest E over k < iter <= k + 500;
                 spikes_ge10: the number of changes with spike >= 10
  R_max          the largest E(t) / min_{s < t} E(s) over the snapshots (rise above the running best; also fixed sigma)
  sgm            the shifted geometric mean over the nine problems of gpu_s_est: exp(mean(log(t + 10))) - 10 (shift 10 s)
Gates per setting and sigma_0 (plan, Phase 4b)
  G1 no problem lost: every problem that fixed sigma = 1 or the old rule (legacy_adaptive, same sigma_0) solves within
     the cap is solved
  G2 the changes stop and the variation is bounded: every run is frozen, or solved before any freeze; tv_log2 is reported
  G3 no change followed by a >= 10x rise of the validated error: spikes_ge10 = 0 over the setting's runs
  G4 a better shifted geometric mean of gpu_s_est than both residual_balancing (new, same sigma_0) and fixed sigma = 1
  G5 robust across sigma_0: G1-G4 hold at every sigma_0 the setting was run at (reported per setting)
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cuadmm_emu as ce  # noqa: E402

DEFAULT_ROOT = os.path.expanduser('~/cuadmm-experiments/2026-10-21_phase4b_screen')
DEFAULT_COSTS = os.path.expanduser('~/cuadmm-worktrees/long-horizon/experiments/2026-09-29_plato_cold_plain_admm_sigma/results/stage3_runs.csv')
SPIKE_W = 500
SHIFT = 10.0
E_KEYS = ['dimacs%d' % i for i in range(1, 7)] + ['primal_res', 'dual_res', 'relgap', 'X_cone_violation', 'dual_cone_violation',
                                                  'S_cone_violation']


def gpu_costs(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['config'] == 'fixed' and r['rep'] == '1':
                out[r['dataset']] = (float(r['iter_time_s']), float(r['validation_s_per_snapshot']))
    return out


def E_of(v):
    m = 0.0
    for k in E_KEYS:
        x = v.get(k, float('nan'))
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            return float('inf')
        m = max(m, abs(x))
    return m


def parse_name(name):
    ds, tag, s = name.split('_')
    assert s[0] == 's', name
    return ds, tag, s[1:]


def read_specs(files):
    specs = []
    for sf in files:
        with open(sf) as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                name, ds, rule, sig, cap, vint = line.split()
                specs.append(dict(name=name, dataset=ds, rule=rule, sigma0=sig, cap=int(cap), vint=int(vint)))
    return specs


def analyze_run(root, spec, costs):
    name, ds, cap = spec['name'], spec['dataset'], spec['cap']
    _, tag, sig0 = parse_name(name)
    d = os.path.join(root, 'runs', name)
    row = dict(name=name, dataset=ds, tag=tag, sigma0=float(sig0), rule=spec['rule'], cap=cap)
    if not os.path.exists(os.path.join(d, 'summary.json')):
        row['status'] = 'not started'
        return row
    s = json.load(open(os.path.join(d, 'summary.json')))
    sl = ce.read_gpu_csv(os.path.join(d, 'sigma_log.csv'))
    vpath = os.path.join(d, 'validation_history.csv.gz')
    vh = ce.read_gpu_csv(vpath if os.path.exists(vpath) else os.path.join(d, 'validation_history.csv'))
    row['status'] = s['stop_reason'] if s['finished'] else 'paused'
    row['finished'] = bool(s['finished'])
    row['iterations'] = s['iterations']
    fp, fs = s['first_practical_iteration'], s['first_strict_iteration']
    row['it_practical'] = fp if 0 < fp <= cap else ''
    row['it_strict'] = fs if 0 < fs <= cap else ''
    row['solved'] = bool(0 < fs <= cap)
    vit = np.array([v['iter'] for v in vh], dtype=np.int64)
    vt = np.array([v['time_s'] for v in vh], dtype=float)
    E = np.array([E_of(v) for v in vh])
    mad = np.array([v['max_abs_dimacs'] if isinstance(v['max_abs_dimacs'], (int, float)) else np.nan for v in vh], dtype=float)
    val = np.array([int(v['validated']) for v in vh]) if len(vh) else np.zeros(0, int)
    js = np.where((vit == fs) & (val == 1))[0] if row['solved'] else []
    row['cpu_s_strict'] = round(float(vt[js[0]]), 1) if len(js) else ''
    row['snapshots'] = len(vh)
    if ds in costs:
        gi, gv = costs[ds]
        it_eff = fs if row['solved'] else cap
        row['gpu_s_est'] = it_eff * gi + int(np.sum(vit <= it_eff)) * gv
        row['censored'] = not row['solved']
    else:
        row['gpu_s_est'] = ''
        row['censored'] = ''
    row['E_final'] = float(E[-1]) if len(E) else ''
    row['E_best'] = float(np.min(E)) if len(E) else ''
    row['maxdimacs_best'] = float(np.nanmin(mad)) if len(mad) and np.any(np.isfinite(mad)) else ''
    # sigma statistics
    it = np.array([r['iter'] for r in sl], dtype=np.int64)
    old = np.array([r['old_sigma'] for r in sl], dtype=float)
    new = np.array([r['new_sigma'] for r in sl], dtype=float)
    rl = np.array([r['rule'] for r in sl])
    moved = new != old
    if len(sl):
        assert old[0] == float(sig0) or tag.startswith('bf'), name  # balance_freeze clamps sigma_0
        assert np.all(old[1:] == new[:-1]), name  # the log is a chain
    row['changes'] = int(moved.sum())
    row['log_entries'] = len(sl)
    row['changes_up'] = int(np.sum(new > old))
    row['changes_down'] = int(np.sum(new < old))
    row['tv_log2'] = float(np.sum(np.abs(np.log2(new[moved] / old[moved])))) if moved.any() else 0.0
    sv = np.concatenate([[float(sig0)], new]) if len(sl) else np.array([float(sig0)])
    row['sigma_min'] = float(sv.min())
    row['sigma_max'] = float(sv.max())
    row['sigma_final'] = float(s['final_sig'])
    row['last_change_iter'] = int(it[moved][-1]) if moved.any() else ''
    row['freeze_at'] = s.get('freeze_iteration', -1)
    row['freeze_reason'] = s.get('freeze_reason', '')
    row['frozen'] = bool(row['freeze_at'] > 0)
    # spikes after the changes
    best = dict(spike=float('nan'))
    n10 = 0
    for k, o, nw, r, m in zip(it, old, new, rl, moved):
        if not m:
            continue
        pre = np.where(vit <= k)[0]
        post = np.where((vit > k) & (vit <= k + SPIKE_W))[0]
        if not len(pre) or not len(post):
            continue
        e0 = E[pre[-1]]
        e1 = float(np.max(E[post]))
        sp = e1 / e0 if e0 > 0 else float('inf')
        if sp >= 10:
            n10 += 1
        if not (sp <= best['spike']):
            best = dict(spike=sp, iter=int(k), old=o, new=nw, rule=r)
    row['spike_max'] = best['spike'] if moved.any() else ''
    row['spike_max_at'] = ('%d (%s %g->%g)' % (best['iter'], best['rule'], best['old'], best['new'])) if 'iter' in best else ''
    row['spikes_ge10'] = n10 if moved.any() else 0
    if len(E) > 1:
        rb = np.minimum.accumulate(E)
        R = E[1:] / np.maximum(rb[:-1], 1e-12)
        jr = int(np.argmax(R))
        row['R_max'] = float(R[jr])
        row['R_max_iter'] = int(vit[jr + 1])
    row['cpu_s_total'] = round(s['cpu_iter_s'] + s['cpu_validation_s'], 1)
    row['us_per_iter'] = round(s['us_per_iteration'], 1)
    row['segments'] = len(s.get('segments', []))
    return row


def sgm(values):
    v = [float(x) for x in values if x != '' and x is not None]
    if not v:
        return float('nan')
    return math.exp(sum(math.log(x + SHIFT) for x in v) / len(v)) - SHIFT


def fmt(x):
    if x == '' or x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, float):
        if math.isnan(x):
            return '—'
        if not math.isfinite(x):
            return 'inf' if x > 0 else '-inf'
        if abs(x) >= 100:
            return '%.0f' % x
        return '%.3g' % x
    if isinstance(x, int):
        return '{:,}'.format(x)
    return str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('specs', nargs='+')
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--costs', default=DEFAULT_COSTS)
    ap.add_argument('--out-prefix', default='screen')
    a = ap.parse_args()
    costs = gpu_costs(a.costs)
    specs = read_specs(a.specs)
    rows = [analyze_run(a.root, sp, costs) for sp in specs]
    os.makedirs(os.path.join(a.root, 'results'), exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(a.root, 'results', a.out_prefix + '_runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, float) else v) for k, v in r.items()})
    # settings: (tag, sigma0) -> rows
    problems = []
    for r in rows:
        if r['dataset'] not in problems:
            problems.append(r['dataset'])
    groups = {}
    for r in rows:
        groups.setdefault((r['tag'], r['sigma0']), []).append(r)

    def solved_set(tag, sigma0):
        g = groups.get((tag, sigma0), [])
        return set(r['dataset'] for r in g if r.get('solved'))

    settings = []
    for (tag, sigma0), g in sorted(groups.items(), key=lambda kv: (not kv[0][0].startswith('bf'), kv[0][0], kv[0][1])):
        st = dict(tag=tag, sigma0=sigma0, runs=len(g), finished=sum(1 for r in g if r.get('finished')),
                  solved=sum(1 for r in g if r.get('solved')))
        st['solved_problems'] = ' '.join(r['dataset'] for r in g if r.get('solved'))
        done = [r for r in g if r.get('finished')]
        st['gpu_s_est_sgm'] = sgm([r['gpu_s_est'] for r in done]) if len(done) == len(g) and g else float('nan')
        st['censored'] = sum(1 for r in done if r.get('censored') is True)
        st['tv_log2_max'] = max([r['tv_log2'] for r in done if 'tv_log2' in r] or [float('nan')])
        st['changes_max'] = max([r['changes'] for r in done if 'changes' in r] or [0])
        st['last_change_max'] = max([r['last_change_iter'] for r in done if r.get('last_change_iter', '') != ''] or [0])
        st['spikes_ge10'] = sum(r.get('spikes_ge10', 0) or 0 for r in done)
        st['spike_max'] = max([r['spike_max'] for r in done if isinstance(r.get('spike_max'), float)] or [float('nan')])
        st['frozen'] = sum(1 for r in done if r.get('frozen'))
        st['freeze_reasons'] = ' '.join(sorted(set(r['freeze_reason'] for r in done if r.get('freeze_reason'))))
        st['cpu_core_h'] = sum(r.get('cpu_s_total', 0) or 0 for r in done) / 3600.0
        if tag.startswith('bf'):
            ref = solved_set('fixed', 1.0) | solved_set('old', sigma0)
            mine = solved_set(tag, sigma0)
            lost = sorted(ref - mine)
            st['lost'] = ' '.join(lost)
            complete = len(done) == len(g) and len(g) > 0
            st['G1_no_loss'] = (not lost) if complete else ''
            # every run frozen, or solved without a freeze (a solve before K_f and before any ping-pong)
            st['G2_changes_stop'] = all(r.get('frozen') or r.get('solved') for r in done) if complete else ''
            st['G3_no_spike'] = (st['spikes_ge10'] == 0) if complete else ''
            new_sgm = groups_sgm(groups, 'new', sigma0)
            fixed_sgm = groups_sgm(groups, 'fixed', 1.0)
            st['sgm_new'] = new_sgm
            st['sgm_fixed1'] = fixed_sgm
            ok4 = complete and math.isfinite(st['gpu_s_est_sgm']) and math.isfinite(new_sgm) and math.isfinite(fixed_sgm)
            st['G4_faster'] = (st['gpu_s_est_sgm'] < new_sgm and st['gpu_s_est_sgm'] < fixed_sgm) if ok4 else ''
            st['G1-4'] = all(st[k] is True for k in ('G1_no_loss', 'G2_changes_stop', 'G3_no_spike', 'G4_faster'))
        settings.append(st)
    # robustness across sigma_0 per tag
    by_tag = {}
    for st in settings:
        by_tag.setdefault(st['tag'], []).append(st)
    for st in settings:
        if st['tag'].startswith('bf'):
            sts = by_tag[st['tag']]
            st['sigma0_values'] = ' '.join('%g' % x['sigma0'] for x in sts)
            gates = ('G1_no_loss', 'G2_changes_stop', 'G3_no_spike', 'G4_faster')
            st['G5_robust'] = all(x['G1-4'] for x in sts) if all(x[k] != '' for x in sts for k in gates) else ''
            st['sgm_mean_over_sigma0'] = (sum(x['gpu_s_est_sgm'] for x in sts) / len(sts)
                                          if all(math.isfinite(x['gpu_s_est_sgm']) for x in sts) else float('nan'))
    scols = []
    for st in settings:
        for k in st:
            if k not in scols:
                scols.append(k)
    with open(os.path.join(a.root, 'results', a.out_prefix + '_settings.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=scols, lineterminator='\n')
        w.writeheader()
        for st in settings:
            w.writerow({k: (repr(float(v)) if isinstance(v, float) else v) for k, v in st.items()})
    # the report
    md = ['# Phase-4b CPU screen: %s' % ', '.join(os.path.basename(x) for x in a.specs), '',
          'Runs: %d (%d finished, %d solved); costs: %s; definitions in analyze.py.' %
          (len(rows), sum(1 for r in rows if r.get('finished')), sum(1 for r in rows if r.get('solved')), os.path.basename(a.costs)), '',
          '## Settings (one row per setting and sigma_0)', '',
          '| setting | sigma_0 | runs done | solved | lost vs fixed 1 / old | max tv_log2 | max changes | frozen (reasons) | spikes >= 10x | '
          'max spike | sgm est. GPU s (censored) | new | fixed 1 | G1 | G2 | G3 | G4 | G5 | core-h |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for st in settings:
        md.append('| %s | %g | %d/%d | %d | %s | %s | %s | %s (%s) | %s | %s | %s (%d) | %s | %s | %s | %s | %s | %s | %s | %.2f |' % (
            st['tag'], st['sigma0'], st['finished'], st['runs'], st['solved'], fmt(st.get('lost', '')) if st.get('lost', '') != '' else ('none' if st['tag'].startswith('bf') else '—'),
            fmt(st['tv_log2_max']), fmt(st['changes_max']), st['frozen'], st['freeze_reasons'] or '—', fmt(st['spikes_ge10']), fmt(st['spike_max']),
            fmt(st['gpu_s_est_sgm']), st['censored'], fmt(st.get('sgm_new', '')), fmt(st.get('sgm_fixed1', '')), fmt(st.get('G1_no_loss', '')),
            fmt(st.get('G2_changes_stop', '')), fmt(st.get('G3_no_spike', '')), fmt(st.get('G4_faster', '')), fmt(st.get('G5_robust', '')),
            st['cpu_core_h']))
    md += ['', '## Ranking of the balance_freeze settings (mean over sigma_0 of the shifted geometric mean; G1-G3 at every sigma_0 first)', '']
    ranked = sorted((st for st in settings if st['tag'].startswith('bf') and st['sigma0'] == min(x['sigma0'] for x in by_tag[st['tag']])),
                    key=lambda st: (not all(x['G1_no_loss'] is True and x['G2_changes_stop'] is True and x['G3_no_spike'] is True for x in by_tag[st['tag']]),
                                    st.get('sgm_mean_over_sigma0', float('inf')) if math.isfinite(st.get('sgm_mean_over_sigma0', float('nan'))) else float('inf')))
    md.append('| rank | setting | sigma_0 values | G1-G3 everywhere | mean sgm est. GPU s | G4 everywhere | G5 |')
    md.append('|---|---|---|---|---|---|---|')
    for i, st in enumerate(ranked):
        sts = by_tag[st['tag']]
        g13 = all(x['G1_no_loss'] is True and x['G2_changes_stop'] is True and x['G3_no_spike'] is True for x in sts)
        g4 = all(x['G4_faster'] is True for x in sts)
        md.append('| %d | %s | %s | %s | %s | %s | %s |' % (i + 1, st['tag'], st['sigma0_values'], fmt(g13), fmt(st['sgm_mean_over_sigma0']), fmt(g4),
                                                           fmt(st['G5_robust'])))
    md += ['', '## Runs', '',
           '| run | status | it. practical | it. strict | est. GPU s | changes (log) | tv_log2 | sigma range | final | last change | freeze | '
           'spikes >= 10x | max spike (at) | R_max | E best | E final | CPU s | us/it |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r.get('status') == 'not started':
            md.append('| %s | not started | | | | | | | | | | | | | | | | |' % r['name'])
            continue
        md.append('| %s | %s | %s | %s | %s%s | %s (%s) | %s | %s – %s | %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s |' % (
            r['name'], r['status'], fmt(r['it_practical']), fmt(r['it_strict']), fmt(r['gpu_s_est']), ' (cap)' if r.get('censored') is True else '',
            fmt(r['changes']), fmt(r['log_entries']), fmt(r['tv_log2']), fmt(r['sigma_min']), fmt(r['sigma_max']), fmt(r['sigma_final']),
            fmt(r['last_change_iter']), ('%d %s' % (r['freeze_at'], r['freeze_reason'])) if r['frozen'] else '—', fmt(r['spikes_ge10']),
            fmt(r['spike_max']), r['spike_max_at'] or '—', fmt(r.get('R_max', '')), fmt(r['E_best']), fmt(r['E_final']), fmt(r['cpu_s_total']),
            fmt(r['us_per_iter'])))
    with open(os.path.join(a.root, 'results', a.out_prefix + '_report.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md[:4 + len(settings) + 4]))
    print('written: results/%s_{runs,settings}.csv and results/%s_report.md in %s' % (a.out_prefix, a.out_prefix, a.root))
    return 0


def groups_sgm(groups, tag, sigma0):
    g = groups.get((tag, sigma0), [])
    if not g or not all(r.get('finished') for r in g):
        return float('nan')
    return sgm([r['gpu_s_est'] for r in g])


if __name__ == '__main__':
    sys.exit(main())
