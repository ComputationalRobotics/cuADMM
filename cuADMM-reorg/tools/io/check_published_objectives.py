#!/usr/bin/env python3
"""Runs the cases of tools/baselines/cases_formats.json with a cuADMM-reorg CLI (GPU) and compares the reported
objective with the published value of the instance (plan Phase 2b acceptance: "PLATO instances read from either
format reach their published objectives").

Each case names the problem in one format (relative to the data root), the policy, sigma0, max_iter and tol, the
published objective and `objective_sign`: +1 when the summary's pobj is in the file's sense (SDPA .dat-s / .h5
converted from it: the maximization of <F_0, Y>), -1 when the file is the minimization of the negated data (the
TXT conversions of the experiments, the SeDuMi file and its .h5 conversion). The check compares
objective_sign * pobj (and dobj) with the published value, relative to max(1, |published|).

usage: check_published_objectives.py <cuadmm_exe> <data root> <out dir> [--spec FILE] [--rtol 1e-3] [--extra "..."]
"""
import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exe')
    ap.add_argument('data_root')
    ap.add_argument('out')
    ap.add_argument('--spec', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines', 'cases_formats.json'))
    ap.add_argument('--rtol', type=float, default=1e-3, help='accepted relative deviation from the published value (tol 1e-4 runs)')
    ap.add_argument('--extra', default='', help='extra CLI options for every run')
    ap.add_argument('--only', default='', help='comma-separated case names')
    args = ap.parse_args()
    with open(args.spec) as f:
        cases = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    only = set(x for x in args.only.split(',') if x)
    rows = []
    failures = 0
    for c in cases:
        if only and c['name'] not in only:
            continue
        problem = c['problem'] if os.path.isabs(c['problem']) else os.path.join(args.data_root, c['problem'])
        summary = os.path.join(args.out, c['name'] + '.summary.json')
        log = os.path.join(args.out, c['name'] + '.log')
        if os.path.exists(summary):
            os.remove(summary)
        cmd = [args.exe, problem, '--policy', c['policy'], '--sigma0', repr(float(c['sigma0'])), '--max-iter', str(int(c['max_iter'])),
               '--tol', repr(float(c['tol'])), '--summary', summary] + args.extra.split()
        with open(log, 'w') as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        row = {'name': c['name'], 'format': c.get('format', ''), 'rc': rc, 'published': c['published_objective'], 'sign': c['objective_sign']}
        if rc == 0 and os.path.exists(summary):
            with open(summary) as f:
                s = json.loads(f.readlines()[-1])
            pobj, dobj = s['pobj'], s['dobj']
            scale = max(1.0, abs(c['published_objective']))
            row.update(pobj=pobj, dobj=dobj, iterations=s['iterations'], status=s['status'], time_s=s['time_s'],
                       rel_p=abs(c['objective_sign'] * pobj - c['published_objective']) / scale,
                       rel_d=abs(c['objective_sign'] * dobj - c['published_objective']) / scale,
                       blocks=s.get('blocks'), vec_len=s.get('vec_len'), report=s.get('conversion_report', []))
            row['ok'] = row['rel_p'] <= args.rtol and row['rel_d'] <= args.rtol
        else:
            row['ok'] = False
        failures += not row['ok']
        rows.append(row)
        print('%-20s %-7s rc=%d %s pobj=%s dobj=%s rel_p=%.2e rel_d=%.2e iters=%s status=%s' % (
            row['name'], row['format'], rc, 'OK ' if row['ok'] else 'BAD', row.get('pobj'), row.get('dobj'), row.get('rel_p', float('nan')),
            row.get('rel_d', float('nan')), row.get('iterations'), row.get('status')))
        sys.stdout.flush()
    with open(os.path.join(args.out, 'report.json'), 'w') as f:
        json.dump({'spec': args.spec, 'exe': args.exe, 'rtol': args.rtol, 'cases': rows}, f, indent=1)
    print('%d of %d cases within rtol %g' % (len(rows) - failures, len(rows), args.rtol))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
