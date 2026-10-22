#!/usr/bin/env python3
"""gates_analyze.py: results and verdicts of the Phase-4b H200 gates G0-G3 (plan Phase 4b).

usage: gates_analyze.py [--gates DIR] [--jobs id,id,...] [--out report.md]
Reads runs/G*/<name>/{summary.json, history.csv, sigma_log.csv, validation.csv} written by the gate jobs
(tools/sigma_screen/gpu_gates.sh) and references.json (the campaign's H200 runs and the pendulum experiments), and
writes the gate report (Markdown) and gates_results.json next to the runs.

Per run: status, iterations, solve time (iteration time, the quantity the caps bound), first practical (validated
1e-4) and strict (max|DIMACS| <= 1e-6) iteration and the iteration time at that snapshot, sigma changes, total
variation sum|log2(new/old)|, freeze iteration and reason, spikes (E_post / E_pre around a sigma change: E = max of
|DIMACS 1-6|, eta_p, eta_d, eta_g, cone violations of the validation snapshots, E_pre the last snapshot at or before
the change, E_post the largest within 500 iterations after it), peak GPU memory.
Verdicts:
  G0  reproduction: strict stage reached (trto1, buck1) / practical reached (pendulum) with the reference iteration
      count (exact, else within 1%) and the validated objective (|rel| <= 1e-6 for the strict runs); times reported.
  G1  on trto1, buck1, vibra1, mater-1, mater-2, shmup1 at sigma_0 = 1: (a) no problem lost that fixed sigma = 1 or
      the old rule solves within the cap (campaign H200 runs; G0 for trto1/buck1 fixed); (b) the changes stop (frozen
      or solved) and the variation is bounded; (c) no change followed by a >= 10x rise of the validated error; (d) a
      better shifted geometric mean of the strict time (shift 10 s, unsolved censored at the cap) than residual_balancing
      (the G1 runs) and fixed sigma = 1 (campaign times; G0 for trto1/buck1).
  G2  pendulum N=80: practical time at sigma_0 in {1, 100} against 124 s (old rule) and 625 s (fixed sigma = 1), and
      against the residual_balancing runs; sensitivity to sigma_0.
  G3  mater-3..6: strict within 1800 s against the campaign's fixed (none) and old-rule (mater-3, 4, 5) runs, and the
      residual_balancing runs.
"""
import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cuadmm_emu as ce  # noqa: E402  (read_gpu_csv)

SHIFT = 10.0
SPIKE_W = 500
E_KEYS = ['dimacs%d' % i for i in range(1, 7)] + ['primal_res', 'dual_res', 'relgap', 'X_cone_violation', 'dual_cone_violation',
                                                  'S_cone_violation']


def E_of(v):
    m = 0.0
    for k in E_KEYS:
        x = v.get(k, float('nan'))
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            return float('inf')
        m = max(m, abs(x))
    return m


def sgm(values):
    return math.exp(sum(math.log(x + SHIFT) for x in values) / len(values)) - SHIFT if values else float('nan')


def read_history_times(path):
    """iteration -> iteration time (time_s column) from the 17-digit history."""
    t = {}
    with open(path) as f:
        header = f.readline().strip().split(',')
        ii, it = header.index('iter'), header.index('time_s')
        for line in f:
            p = line.split(',')
            if len(p) > it:
                t[int(p[ii])] = float(p[it])
    return t


def analyze_run(d):
    name = os.path.basename(d)
    gate = os.path.basename(os.path.dirname(d))
    row = dict(gate=gate, name=name)
    sp = os.path.join(d, 'summary.json')
    if not os.path.exists(sp):
        row['status'] = 'missing'
        return row
    s = json.loads(open(sp).readlines()[-1])
    row.update(status=s['status'], stop_reason=s.get('stop_reason'), iterations=s['iterations'], solve_time_s=s['solve_time_s'],
               time_s=s['time_s'], validation_time_s=s.get('validation_time_s'), first_practical=s['first_practical_iteration'],
               first_strict=s['first_strict_iteration'], pobj=s['pobj'], dobj=s['dobj'], sigma_final=s.get('sigma_final_scaled', s.get('sigma_final')),
               sigma_changes=s.get('sigma_changes'), freeze_iteration=s.get('freeze_iteration'), freeze_reason=s.get('freeze_reason'),
               peak_gpu_mem_mib=s.get('peak_gpu_mem_mib'), returned_best=s.get('returned_best_iterate'), policy=s.get('sigma_policy'),
               sigma0=s.get('sigma0'))
    times = read_history_times(os.path.join(d, 'history.csv')) if os.path.exists(os.path.join(d, 'history.csv')) else {}
    row['t_practical'] = times.get(row['first_practical']) if row['first_practical'] and row['first_practical'] > 0 else None
    row['t_strict'] = times.get(row['first_strict']) if row['first_strict'] and row['first_strict'] > 0 else None
    row['solved_strict'] = bool(row['first_strict'] and row['first_strict'] > 0)
    row['solved_practical'] = bool(row['first_practical'] and row['first_practical'] > 0)
    # sigma log
    sl = ce.read_gpu_csv(os.path.join(d, 'sigma_log.csv')) if os.path.exists(os.path.join(d, 'sigma_log.csv')) else []
    moved = [r for r in sl if r['new_sigma'] != r['old_sigma']]
    row['changes'] = len(moved)
    row['tv_log2'] = sum(abs(math.log2(r['new_sigma'] / r['old_sigma'])) for r in moved)
    row['last_change_iter'] = moved[-1]['iter'] if moved else None
    row['rules'] = sorted(set(r['rule'] for r in sl))
    # validation history: spikes
    vp = os.path.join(d, 'validation.csv')
    if os.path.exists(vp):
        vh = ce.read_gpu_csv(vp)
        vit = [v['iter'] for v in vh]
        E = [E_of(v) for v in vh]
        n10, worst = 0, (0.0, None)
        for r in moved:
            k = r['iter']
            pre = [i for i, it in enumerate(vit) if it <= k]
            post = [i for i, it in enumerate(vit) if k < it <= k + SPIKE_W]
            if not pre or not post:
                continue
            e0, e1 = E[pre[-1]], max(E[i] for i in post)
            spk = e1 / e0 if e0 > 0 else float('inf')
            if spk >= 10:
                n10 += 1
            if spk > worst[0]:
                worst = (spk, k)
        row['snapshots'] = len(vh)
        row['spikes_ge10'] = n10
        row['spike_max'] = worst[0]
        row['spike_max_iter'] = worst[1]
        row['E_final'] = E[-1] if E else None
        row['E_best'] = min(E) if E else None
    return row


def verdict_g0(rows, ref):
    out = []
    for r in rows:
        if r['gate'] != 'G0' or r.get('status') == 'missing':
            continue
        if r['name'].startswith('pendulum'):
            ref_it = ref['pendulum_N80']['fixed_s1']['practical_iteration']
            got = r['first_practical']
            ok_it = got > 0 and abs(got - ref_it) <= 0.01 * ref_it
            out.append(dict(run=r['name'], criterion='practical 1e-4 at %d iterations (reference %d, 622 s solve / 625 s end to end)' % (got, ref_it),
                            iterations_match=(got == ref_it), within_1pct=ok_it, time_s=r['t_practical'], solve_time_s=r['solve_time_s'],
                            passed=bool(ok_it)))
        else:
            p = r['name'].split('_')[0]
            ref_it = ref['campaign_fixed_s1'][p]['strict_iteration']
            got = r['first_strict']
            obj_ref = ref['validated_objectives_min_convention'][p]
            rel = abs(r['pobj'] - obj_ref) / max(1.0, abs(obj_ref))
            ok_it = got > 0 and abs(got - ref_it) <= 0.01 * ref_it
            out.append(dict(run=r['name'], criterion='strict stage at %d iterations (reference %d, %.1f s), objective rel. diff %.2e' % (got, ref_it, ref['campaign_fixed_s1'][p]['solve_time_s'], rel),
                            iterations_match=(got == ref_it), within_1pct=ok_it, objective_rel_diff=rel, time_s=r['t_strict'], solve_time_s=r['solve_time_s'],
                            passed=bool(ok_it and rel <= 1e-6)))
    return out


def verdict_g1(rows, ref, g0rows):
    probs = ['trto1', 'buck1', 'vibra1', 'mater-1', 'mater-2', 'shmup1']
    caps = ref['caps_s']
    bf = {p: next((r for r in rows if r['gate'] == 'G1' and r['name'] == p + '_bf_s1'), None) for p in probs}
    rb = {p: next((r for r in rows if r['gate'] == 'G1' and r['name'] == p + '_rb_s1'), None) for p in probs}
    fixed_t, old_t, must = {}, {}, set()
    for p in probs:
        f = ref['campaign_fixed_s1'][p]
        g0 = next((r for r in g0rows if r['name'] == p + '_fixed_s1'), None)
        if g0 is not None and g0.get('solved_strict'):
            fixed_t[p] = g0['t_strict']
        elif f['strict_iteration'] > 0 and f['solve_time_s'] <= caps[p]:
            fixed_t[p] = f['solve_time_s']
        else:
            fixed_t[p] = caps[p]
        o = ref['campaign_old_rule_s1'][p]
        old_t[p] = o['solve_time_s'] if o['strict_iteration'] > 0 and o['solve_time_s'] <= caps[p] else caps[p]
        if fixed_t[p] < caps[p] or old_t[p] < caps[p]:
            must.add(p)

    def t_of(r, p):
        return r['t_strict'] if r and r.get('solved_strict') and r['t_strict'] is not None and r['t_strict'] <= caps[p] else caps[p]
    done = all(bf[p] is not None and bf[p].get('status') != 'missing' for p in probs)
    lost = sorted(p for p in must if bf[p] is None or not bf[p].get('solved_strict'))
    frozen = all(bf[p] and (bf[p].get('freeze_iteration', -1) > 0 or bf[p].get('solved_strict')) for p in probs)
    tv = max((bf[p]['tv_log2'] for p in probs if bf[p]), default=float('nan'))
    spikes = sum(bf[p].get('spikes_ge10', 0) or 0 for p in probs if bf[p])
    sg_bf = sgm([t_of(bf[p], p) for p in probs]) if done else float('nan')
    sg_rb = sgm([t_of(rb[p], p) for p in probs]) if all(rb[p] for p in probs) else float('nan')
    sg_fx = sgm([fixed_t[p] for p in probs])
    sg_old = sgm([old_t[p] for p in probs])
    return dict(complete=done, must_solve=sorted(must), lost=lost, G1a_no_loss=(not lost) if done else None,
                G1b_changes_stop=frozen if done else None, tv_log2_max=tv, G1c_no_spike=(spikes == 0) if done else None, spikes_ge10=spikes,
                sgm_candidate=sg_bf, sgm_residual_balancing=sg_rb, sgm_fixed=sg_fx, sgm_old=sg_old,
                G1d_faster=(sg_bf < sg_rb and sg_bf < sg_fx) if done and math.isfinite(sg_rb) else None,
                per_problem={p: dict(candidate=t_of(bf[p], p) if bf[p] else None, candidate_solved=bool(bf[p] and bf[p].get('solved_strict')),
                                     residual_balancing=t_of(rb[p], p) if rb[p] else None, rb_solved=bool(rb[p] and rb[p].get('solved_strict')),
                                     fixed=fixed_t[p], old=old_t[p]) for p in probs})


def verdict_g2(rows, ref):
    out = {}
    for r in rows:
        if r['gate'] == 'G2' and r.get('status') != 'missing':
            out[r['name']] = dict(practical_iter=r['first_practical'], t_practical=r['t_practical'], strict_iter=r['first_strict'], t_strict=r['t_strict'],
                                  solve_time_s=r['solve_time_s'], status=r['status'], changes=r['changes'], freeze=(r.get('freeze_iteration'), r.get('freeze_reason')),
                                  spikes_ge10=r.get('spikes_ge10'))
    out['references'] = dict(old_rule_s1_end_to_end_s=ref['pendulum_N80']['old_rule_s1']['end_to_end_s'], old_rule_practical_iteration=45393,
                             fixed_s1_solve_s=ref['pendulum_N80']['fixed_s1']['solve_time_s'], fixed_s1_practical_iteration=230315)
    return out


def verdict_g3(rows, ref):
    out = {}
    for p in ['mater-3', 'mater-4', 'mater-5', 'mater-6']:
        bf = next((r for r in rows if r['gate'] == 'G3' and r['name'] == p + '_bf_s1'), None)
        rb = next((r for r in rows if r['gate'] == 'G3' and r['name'] == p + '_rb_s1'), None)
        out[p] = dict(candidate=(bf['first_strict'], bf['t_strict'], bf['status']) if bf and bf.get('status') != 'missing' else None,
                      residual_balancing=(rb['first_strict'], rb['t_strict'], rb['status']) if rb and rb.get('status') != 'missing' else None,
                      campaign_fixed=(ref['campaign_fixed_s1'][p]['strict_iteration'], ref['campaign_fixed_s1'][p]['solve_time_s']),
                      campaign_old=(ref['campaign_old_rule_s1'][p]['strict_iteration'], ref['campaign_old_rule_s1'][p]['solve_time_s']),
                      candidate_spikes=bf.get('spikes_ge10') if bf else None, candidate_freeze=(bf.get('freeze_iteration'), bf.get('freeze_reason')) if bf else None)
    return out


def usage(jobs):
    if not jobs:
        return None
    r = subprocess.run(['sacct', '-j', ','.join(jobs), '-X', '-o', 'JobID,JobName,State,ElapsedRaw,Timelimit,NodeList', '-n', '-P'], capture_output=True, text=True)
    rows, total = [], 0
    for line in r.stdout.strip().splitlines():
        jid, name, state, el, tl, node = line.split('|')[:6]
        total += int(el or 0)
        rows.append(dict(job=jid, name=name, state=state, elapsed_s=int(el or 0), timelimit=tl, node=node))
    return dict(jobs=rows, total_h200_hours=total / 3600.0)


def fmt(x):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, float):
        return ('%.3g' % x) if abs(x) < 100 else ('%.0f' % x)
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gates', default=os.path.expanduser('~/cuadmm-experiments/2026-10-21_phase4b_screen/gpu_gates'))
    ap.add_argument('--jobs', default='')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ref = json.load(open(os.path.join(a.gates, 'references.json')))
    dirs = sorted(glob.glob(os.path.join(a.gates, 'runs', 'G*', '*')))
    rows = [analyze_run(d) for d in dirs if os.path.isdir(d)]
    g0 = verdict_g0(rows, ref)
    g0rows = [r for r in rows if r['gate'] == 'G0']
    g1 = verdict_g1(rows, ref, g0rows)
    g2 = verdict_g2(rows, ref)
    g3 = verdict_g3(rows, ref)
    use = usage([j for j in a.jobs.split(',') if j])
    res = dict(runs=rows, G0=g0, G1=g1, G2=g2, G3=g3, usage=use)
    with open(os.path.join(a.gates, 'gates_results.json'), 'w') as f:
        json.dump(res, f, indent=1, default=str)
    md = ['# Phase-4b H200 gates: results', '']
    md += ['## Runs', '', '| gate | run | status | iterations | solve s | practical it (s) | strict it (s) | changes | tv_log2 | freeze | spikes >= 10x | max spike | E final | peak MiB |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r.get('status') == 'missing':
            md.append('| %s | %s | missing | | | | | | | | | | | |' % (r['gate'], r['name']))
            continue
        md.append('| %s | %s | %s | %s | %s | %s (%s) | %s (%s) | %s | %s | %s | %s | %s | %s | %s |' % (
            r['gate'], r['name'], r['status'], fmt(r['iterations']), fmt(r['solve_time_s']), fmt(r['first_practical']), fmt(r['t_practical']),
            fmt(r['first_strict']), fmt(r['t_strict']), fmt(r['changes']), fmt(r['tv_log2']),
            ('%s %s' % (r['freeze_iteration'], r['freeze_reason'])) if (r.get('freeze_iteration') or -1) > 0 else '—',
            fmt(r.get('spikes_ge10')), fmt(r.get('spike_max')), fmt(r.get('E_final')), fmt(r.get('peak_gpu_mem_mib'))))
    md += ['', '## G0 fixed-sigma reproduction', '']
    for v in g0:
        md.append('- %s: %s; iterations identical: %s; within 1%%: %s; time to the milestone %s s (solve %s s): **%s**' % (
            v['run'], v['criterion'], fmt(v['iterations_match']), fmt(v['within_1pct']), fmt(v['time_s']), fmt(v['solve_time_s']), 'pass' if v['passed'] else 'FAIL'))
    md += ['', '## G1 (sigma_0 = 1)', '', 'must-solve set (fixed sigma = 1 or the old rule within the cap): %s' % ', '.join(g1['must_solve']), '',
           '| problem | candidate strict s (solved) | residual_balancing strict s (solved) | fixed sigma = 1 s | old rule s |', '|---|---|---|---|---|']
    for p, d in g1['per_problem'].items():
        md.append('| %s | %s (%s) | %s (%s) | %s | %s |' % (p, fmt(d['candidate']), fmt(d['candidate_solved']), fmt(d['residual_balancing']), fmt(d['rb_solved']), fmt(d['fixed']), fmt(d['old'])))
    md += ['', '- G1a no problem lost: %s (lost: %s)' % (fmt(g1['G1a_no_loss']), ', '.join(g1['lost']) or 'none'),
           '- G1b changes stop (every run frozen or solved): %s; max total variation %s' % (fmt(g1['G1b_changes_stop']), fmt(g1['tv_log2_max'])),
           '- G1c no spike >= 10x: %s (%s)' % (fmt(g1['G1c_no_spike']), fmt(g1['spikes_ge10'])),
           '- G1d shifted geometric mean of the strict time: candidate %s s, residual_balancing %s s, fixed %s s, old %s s: %s' % (
               fmt(g1['sgm_candidate']), fmt(g1['sgm_residual_balancing']), fmt(g1['sgm_fixed']), fmt(g1['sgm_old']), fmt(g1['G1d_faster']))]
    md += ['', '## G2 pendulum N=80', '', '```', json.dumps(g2, indent=1, default=str), '```']
    md += ['', '## G3 mater-3..6', '', '```', json.dumps(g3, indent=1, default=str), '```']
    if use:
        md += ['', '## H200 usage', '', '| job | name | state | elapsed | limit | node |', '|---|---|---|---|---|---|']
        for j in use['jobs']:
            md.append('| %s | %s | %s | %.2f h | %s | %s |' % (j['job'], j['name'], j['state'], j['elapsed_s'] / 3600.0, j['timelimit'], j['node']))
        md.append('')
        md.append('total allocated: %.2f H200-hours (approved: at most 12.58)' % use['total_h200_hours'])
    out = a.out or os.path.join(a.gates, 'gates_report.md')
    with open(out, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    sys.exit(main())
