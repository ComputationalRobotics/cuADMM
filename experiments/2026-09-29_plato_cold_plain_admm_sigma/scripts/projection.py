#!/usr/bin/env python3
"""H200-hour projection of Stages 2 and 3 from the Stage-1 smoke runs (budgeting only; nothing here is a result).

Per dataset and sigma configuration, the solve time of a planned run is estimated as
  known       the smoke run reached the target within 120 s: that time (the trajectory is deterministic; the planned
              validation interval changes it by at most one interval)
  expected    otherwise: extrapolated from the external convergence rate over the last half of the smoke run
              (log10 of eta for the practical target, of max_abs_DIMACS for the strict target, against the solve
              clock); no progress -> the cap
  maximum     every run whose outcome is not known is counted at its cap
plus the per-run overhead measured in the smoke runs (load, CUDA and solver initialization, final validation and
saving: wall time minus solve time). Stage 3 adds repetitions 2 and 3 for runs under 10 minutes end to end. Configurations
without smoke data (the Stage-2 grid other than sigma = 1, the tuned sigma) are counted at the cap in the maximum and,
in the expected value, at the cap as well unless stated otherwise (their outcome is unknown).
usage: projection.py [--tuned-sigma-known-equal-1]"""
import argparse, csv, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

EXP = A.EXP
PLAN2 = dict(limit=900, pilot=['buck3', 'mater-3', 'shmup3', 'trto3', 'vibra3'], grid=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0])
LIMIT3 = 3600
REP_S = 600


def rate_estimate(d, key, target, cap):
    """solve time to reach `target` extrapolated from the second half of the smoke validation history"""
    vh = A.read_csv(os.path.join(d, 'validation_history.csv'))
    pts = []
    for v in vh:
        t = A.fnum(v['time_s'])
        y = max(A.fnum(v['primal_res']), A.fnum(v['dual_res']), A.fnum(v['relgap'])) if key == 'eta' else A.fnum(v['max_abs_dimacs'])
        if math.isfinite(t) and math.isfinite(y) and y > 0:
            pts.append((t, math.log10(y)))
    if len(pts) < 4:
        return cap
    for t, y in pts:
        if y <= math.log10(target):
            return t
    half = [p for p in pts if p[0] >= pts[-1][0] / 2]
    if len(half) < 3:
        return cap
    n = len(half)
    mt, my = sum(p[0] for p in half) / n, sum(p[1] for p in half) / n
    sxx = sum((p[0] - mt) ** 2 for p in half)
    if sxx <= 0:
        return cap
    slope = sum((p[0] - mt) * (p[1] - my) for p in half) / sxx
    if slope >= 0:
        return cap
    # from the best recent level (the minimum over the last half), not the noisy last point
    y0 = min(p[1] for p in half)
    t_est = pts[-1][0] + (y0 - math.log10(target)) / (-slope)
    return min(cap, t_est)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tuned-sigma-known-equal-1', action='store_true')
    a = ap.parse_args()
    rows = A.load_stage('stage1')
    by = {}
    for r in rows:
        by.setdefault(r['dataset'], {})[r['config']] = r
    over = {d: max((c.get('wall_s', 0) - (c.get('solve_time_s') or 0)) for c in cs.values()) for d, cs in by.items()}
    out, groups = [], {}

    def add(group, d, cfg, solve_known, solve_exp, cap, reps_rule):
        o = over.get(d, 10.0)
        exp = min(cap, solve_exp) + o
        mx = (solve_known if solve_known is not None else cap) + o
        reps_e = 3 if (reps_rule and exp < REP_S) else 1
        reps_m = 3 if (reps_rule and mx < REP_S) else 1
        out.append(dict(group=group, dataset=d, config=cfg, cap_s=cap, known_s=solve_known, expected_run_s=exp, max_run_s=mx, expected_reps=reps_e,
                        max_reps=reps_m, expected_s=exp * reps_e, max_s=mx * reps_m))
        g = groups.setdefault(group, [0.0, 0.0])
        g[0] += exp * reps_e
        g[1] += mx * reps_m

    for d in PLAN2['pilot']:
        for cfg in ['adaptive'] + [f'fixed_s{s:g}' for s in PLAN2['grid']]:
            base = 'adaptive' if cfg == 'adaptive' else ('fixed' if cfg == 'fixed_s1' else None)
            r = by.get(d, {}).get(base) if base else None
            known = r['first_practical_time_s'] if (r and r.get('practical_validated')) else None
            exp = known if known is not None else (rate_estimate(os.path.join(EXP, 'runs', 'stage1', r['run_id']), 'eta', 1e-4, PLAN2['limit']) if r else PLAN2['limit'])
            add('stage2 pilot', d, cfg, known, exp, PLAN2['limit'], False)
    for d in sorted(by, key=lambda x: int(A.INV[x]['cuadmm_vec_len'])):
        for cfg in ['adaptive', 'fixed', 'tuned']:
            if cfg == 'tuned' and a.tuned_sigma_known_equal_1:
                continue
            r = by[d].get('fixed' if cfg == 'tuned' else cfg)
            if cfg == 'tuned':
                add('stage3 tuned', d, cfg, None, LIMIT3, LIMIT3, True)
                continue
            known = r['first_strict_time_s'] if (r and r.get('strict_validated')) else None
            exp = known if known is not None else rate_estimate(os.path.join(EXP, 'runs', 'stage1', r['run_id']), 'dimacs', 1e-6, LIMIT3)
            add(f'stage3 {cfg}', d, cfg, known, exp, LIMIT3, True)
    A.write_csv(os.path.join(A.RES, 'budget_projection.csv'), out)
    tot_e = sum(g[0] for g in groups.values()) / 3600
    tot_m = sum(g[1] for g in groups.values()) / 3600
    lines = ['| run group | runs | expected H200-h | maximum H200-h |', '|---|---|---|---|']
    for g, (e, m) in groups.items():
        lines.append(f"| {g} | {sum(1 for o in out if o['group'] == g)} | {e / 3600:.2f} | {m / 3600:.2f} |")
    lines.append(f'| **total Stages 2-3** | {len(out)} | **{tot_e:.2f}** | **{tot_m:.2f}** |')
    txt = '\n'.join(lines)
    open(os.path.join(A.RES, 'budget_projection.md'), 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
