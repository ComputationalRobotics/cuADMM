#!/usr/bin/env python3
"""Revalidates the saved iterates of finished runs with the independent validator (scripts/dimacs_validate.py), on
the CPU allocation: runs/<stage>/<run_id>/postval_returned.json and postval_final.json (the final iterate exists
only when the solve ended at a safety limit). Skips iterates already revalidated.
usage: postvalidate.py <stage> [<stage> ...] [--jobs N]"""
import argparse, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIG = os.path.expanduser('~/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma')
TXT = os.path.expanduser('~/cuadmm-data/plato_kocvara/txt')
PY = sys.executable


def todo(stage):
    for rj in sorted(glob.glob(os.path.join(EXP, 'runs', stage, '*', 'run.json'))):
        r = json.load(open(rj))
        d = os.path.dirname(rj)
        for which in ('returned', 'final'):
            it = os.path.join(BIG, 'runs', stage, r['run_id'], which)
            out = os.path.join(d, f'postval_{which}.json')
            if os.path.exists(out) or not (os.path.exists(os.path.join(it, 'X.txt.gz')) or os.path.exists(os.path.join(it, 'X.txt'))):
                continue
            yield r, it, out


def one(job):
    r, it, out = job
    p = subprocess.run([PY, os.path.join(EXP, 'scripts', 'dimacs_validate.py'), os.path.join(TXT, r['dataset']), it, out + '.tmp',
                        '--problem-name', r['dataset']], capture_output=True, text=True,
                       env=dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
    if p.returncode == 0:
        os.replace(out + '.tmp', out)
    return r['run_id'], os.path.basename(out), p.returncode, (p.stdout + p.stderr).strip()[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stages', nargs='+')
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    jobs = [j for s in a.stages for j in todo(s)]
    with ThreadPoolExecutor(a.jobs) as ex:
        for rid, which, rc, msg in ex.map(one, jobs):
            print(f'{rid} {which}: exit {rc} {msg}', flush=True)
    print(f'{len(jobs)} iterates revalidated')


if __name__ == '__main__':
    main()
