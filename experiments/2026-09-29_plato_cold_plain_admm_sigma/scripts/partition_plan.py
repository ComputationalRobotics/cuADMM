#!/usr/bin/env python3
"""Splits plans/stage3.json into two plans for two concurrent 1-H200 jobs: every dataset's runs (both configurations and
their repetitions) stay in one part, so each paired comparison runs back to back on the same GPU; datasets are assigned by
longest-processing-time-first on their maximum cost (Stage-1 known time of a run that reached the strict target within
120 s, times 3 repetitions when under 10 minutes; otherwise the cap), plus the measured per-run overhead. Within a part the
order of the plan is kept (datasets by increasing vec_len, configurations alternating).
Writes plans/stage3_part1.json, plans/stage3_part2.json and prints each part's maximum in hours.
usage: partition_plan.py"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

EXP = A.EXP
plan = json.load(open(os.path.join(EXP, 'plans', 'stage3.json')))
cap = plan['limit_s']
s1 = {(r['dataset'], r['config']): r for r in A.load_stage('stage1')}
cost = {}
for e in plan['runs']:
    r = s1[(e['dataset'], e['config'])]
    over = r['wall_s'] - r['solve_time_s']
    known = r['first_strict_time_s'] if r.get('strict_validated') else None
    run = (min(cap, known) if known is not None else cap) + over
    cost[e['dataset']] = cost.get(e['dataset'], 0.0) + run * (3 if run < 600 else 1)
bins = [[], []]
load = [0.0, 0.0]
for d in sorted(cost, key=lambda x: -cost[x]):
    i = 0 if load[0] <= load[1] else 1
    bins[i].append(d)
    load[i] += cost[d]
for i in (0, 1):
    part = dict(plan, part=i + 1, datasets=sorted(bins[i], key=lambda d: int(A.INV[d]['cuadmm_vec_len'])),
                runs=[e for e in plan['runs'] if e['dataset'] in bins[i]], max_h=load[i] / 3600)
    out = os.path.join(EXP, 'plans', f'stage3_part{i + 1}.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} exists (plans are frozen once written)')
    json.dump(part, open(out, 'w'), indent=1)
    print(f"part {i + 1}: {len(part['runs'])} runs, {len(bins[i])} datasets, maximum {load[i] / 3600:.2f} H200-h: {' '.join(part['datasets'])}")
print(f'total maximum {sum(load) / 3600:.2f} H200-h')
