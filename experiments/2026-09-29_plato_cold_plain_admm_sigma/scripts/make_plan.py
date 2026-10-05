#!/usr/bin/env python3
"""Builds the run plan of one stage (plans/<stage>.json). The definitions below are the predeclared protocol of
PROTOCOL.md; they are fixed before the runs of the stage.

usage: make_plan.py stage1
       make_plan.py stage2
       make_plan.py stage3 --tuned-sigma <s> [--intervals plans/validation_intervals.json]
       make_plan.py stage3 --plan-a     (reduced plan A: adaptive and fixed only, no pilot/tuned, 1800 s cap)

Every run: cold start (X = 0, y = 0, S = 0, the solver's own initialization), plain ADMM (--algorithm admm), one
H200, --max-iter 1e9 (no iteration cap: the time limit binds), external validation with the practical criteria at
1e-4 (--validate-tol 1e-4; the internal --tol 1e-4 only triggers validations).
Configurations:
  adaptive   --sigma-policy legacy_adaptive --sig 1       (the historical adaptive rules, initial sigma 1)
  fixed      --sigma-policy fixed --sig 1                 (the same initial sigma, never changed)
  tuned      --sigma-policy fixed --sig <sigma*>          (sigma* chosen on the stage-2 pilot by the frozen rule)
Initial sigma 1: the value of the upstream example for this collection (MATLAB/example_sdpt3.m solves vibra4 with
sigma = 1.0); the CLI default is 100, which is in the pilot grid.
Order: datasets by increasing vec_len; within a dataset the configurations alternate (even index adaptive first,
odd index fixed first; tuned in the middle), and repetitions interleave (A F A F ...)."""
import argparse, csv, json, math, os

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV = {r['dataset']: r for r in csv.DictReader(open(os.path.join(EXP, 'dataset_inventory.csv')))}
DATASETS = sorted(INV, key=lambda d: (int(INV[d]['cuadmm_vec_len']), d))
SIGMA0 = 1.0
PILOT = ['buck3', 'mater-3', 'shmup3', 'trto3', 'vibra3']   # the third member of every family (predeclared)
GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]         # tuned fixed-sigma grid (predeclared)
LIMITS = dict(stage1=120, stage2=900, stage3=3600)          # seconds of solve time per run
PLAN_A_LIMIT = 1800  # reduced plan A (approved by the user on 2026-09-30; PROTOCOL.md, Deviation 7)


def smoke_interval(d):
    n = int(INV[d]['cuadmm_vec_len'])
    return 100 if n < 100_000 else (200 if n < 1_000_000 else 500)


def cfg(name, sigma=None):
    if name == 'adaptive':
        return dict(config='adaptive', policy='legacy_adaptive', sigma=SIGMA0)
    if name == 'fixed':
        return dict(config='fixed', policy='fixed', sigma=SIGMA0)
    return dict(config=name, policy='fixed', sigma=sigma)


def entry(stage, i, d, c, limit, vint, strict, rep=1):
    return dict(run_id=f"{stage}_{d}_{c['config']}_r{rep}", stage=stage, dataset=d, rep=rep, time_limit_s=limit, validate_interval=vint,
                strict_dimacs_tol=strict, order=i, **c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['stage1', 'stage2', 'stage3'])
    ap.add_argument('--tuned-sigma', type=float)
    ap.add_argument('--plan-a', action='store_true')
    ap.add_argument('--intervals', default=os.path.join(EXP, 'plans', 'validation_intervals.json'))
    a = ap.parse_args()
    runs = []
    if a.stage == 'stage1':
        # smoke: all 26 datasets, adaptive and fixed at sigma 1, 2 minutes each, the strict stage on (tests the path)
        for k, d in enumerate(DATASETS):
            pair = ['adaptive', 'fixed'] if k % 2 == 0 else ['fixed', 'adaptive']
            for c in pair:
                runs.append(entry('stage1', len(runs), d, cfg(c), LIMITS['stage1'], smoke_interval(d), 1e-6))
    else:
        iv = json.load(open(a.intervals))  # per dataset, from stage 1 (validation overhead <= ~10 %)
        if a.stage == 'stage2':
            # pilot: the 5 pilot datasets, the adaptive reference and the fixed grid, 15 minutes each, stop at the
            # practical target (no strict stage); order: datasets by size, the grid alternately ascending/descending
            for k, d in enumerate([x for x in DATASETS if x in PILOT]):
                grid = GRID if k % 2 == 0 else GRID[::-1]
                cs = [cfg('adaptive')] + [cfg(f'fixed_s{s:g}', s) for s in grid]
                if k % 2:
                    cs = cs[1:] + cs[:1]
                for c in cs:
                    runs.append(entry('stage2', len(runs), d, c, LIMITS['stage2'], iv[d], 0.0))
        elif a.plan_a:
            # reduced plan A: adaptive vs fixed sigma 1 on all 26 datasets, the same 1800 s cap for both, the strict
            # stage on; the configurations alternate within a dataset (even index adaptive first, odd fixed first)
            for k, d in enumerate(DATASETS):
                for c in (['adaptive', 'fixed'] if k % 2 == 0 else ['fixed', 'adaptive']):
                    runs.append(entry('stage3', len(runs), d, cfg(c), PLAN_A_LIMIT, iv[d], 1e-6))
        else:
            if a.tuned_sigma is None:
                raise SystemExit('stage3 needs --tuned-sigma (the stage-2 selection)')
            for k, d in enumerate(DATASETS):
                order = ['adaptive', 'tuned', 'fixed'] if k % 2 == 0 else ['fixed', 'tuned', 'adaptive']
                if a.tuned_sigma == SIGMA0:
                    order = [c for c in order if c != 'tuned']  # the tuned configuration is the fixed one
                for c in order:
                    runs.append(entry('stage3', len(runs), d, cfg(c, a.tuned_sigma), LIMITS['stage3'], iv[d], 1e-6))
    os.makedirs(os.path.join(EXP, 'plans'), exist_ok=True)
    out = os.path.join(EXP, 'plans', a.stage + '.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} exists (plans are frozen once written)')
    limit = PLAN_A_LIMIT if (a.stage == 'stage3' and a.plan_a) else LIMITS[a.stage]
    json.dump(dict(stage=a.stage, plan='A (reduced; adaptive vs fixed, 1800 s)' if a.plan_a else 'as declared', sigma0=SIGMA0, pilot=PILOT, grid=GRID,
                   limit_s=limit, runs=runs), open(out, 'w'), indent=1)
    print(f'{out}: {len(runs)} runs, worst case {len(runs) * limit / 3600:.1f} h of solve time (before repetitions)')


if __name__ == '__main__':
    main()
