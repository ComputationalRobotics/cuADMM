#!/usr/bin/env python3
"""check_formats.py: the GPU acceptance of the Phase 2b readers (plan: "PLATO instances read from either format reach
their published objectives"), in two parts per instance group of tools/baselines/cases_formats_gpu.json:
  objective  the run converges (internal tolerance) and sign * pobj and sign * dobj agree with the reference objective
             within --rtol (reference: a strict-validated run of the TXT conversion, see the spec's notes);
  cross      the 17-digit history of the run is compared with the group's TXT run (or `cross_base`) by
             tools/compare_runs.py: identical within `cross_rtol` (0 = bitwise), with the first differing iteration.
usage: check_formats.py <cuadmm_exe> <data root> <out dir> [--spec FILE] [--rtol 1e-3] [--only a,b]
Outputs <out dir>/<case>/{history.csv,summary.json,sigma_log.csv,stdout.log}, <out dir>/report.json; exit 1 on any failure.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(HERE, '..', 'compare_runs.py')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exe')
    ap.add_argument('data_root')
    ap.add_argument('out')
    ap.add_argument('--spec', default=os.path.join(HERE, '..', 'baselines', 'cases_formats_gpu.json'))
    ap.add_argument('--rtol', type=float, default=1e-3)
    ap.add_argument('--only', default='')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--reuse', action='store_true', help='skip the solver when history.csv and summary.json exist')
    a = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location('compare_runs', COMPARE)
    compare_runs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare_runs)
    cases = json.load(open(a.spec))
    only = set(x for x in a.only.split(',') if x)
    os.makedirs(a.out, exist_ok=True)
    rows, failures = [], 0
    for c in cases:
        if only and c['name'] not in only:
            continue
        d = os.path.join(a.out, c['name'])
        os.makedirs(d, exist_ok=True)
        reuse = a.reuse and os.path.exists(os.path.join(d, 'history.csv')) and os.path.exists(os.path.join(d, 'summary.json'))
        if not reuse:
            for f in ('history.csv', 'summary.json', 'sigma_log.csv'):
                if os.path.exists(os.path.join(d, f)):
                    os.remove(os.path.join(d, f))
        problem = c['problem'] if os.path.isabs(c['problem']) else os.path.join(a.data_root, c['problem'])
        cmd = [a.exe, problem, '--policy', c['policy'], '--sigma0', repr(float(c['sigma0'])), '--max-iter', str(int(c['max_iter'])),
               '--tol', repr(float(c['tol'])), '--history', os.path.join(d, 'history.csv'), '--summary', os.path.join(d, 'summary.json'),
               '--sigma-log', os.path.join(d, 'sigma_log.csv')]
        if reuse:
            rc = 0
        else:
            with open(os.path.join(d, 'stdout.log'), 'w') as lf:
                rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        row = dict(name=c['name'], group=c['group'], format=c.get('format', ''), check=c['check'], rc=rc, ok=True, notes=[])
        s = None
        if rc == 0 and os.path.exists(os.path.join(d, 'summary.json')):
            with open(os.path.join(d, 'summary.json')) as f:
                s = json.loads(f.readlines()[-1])
            row.update(pobj=s['pobj'], dobj=s['dobj'], iterations=s['iterations'], status=s['status'], time_s=s['time_s'],
                       blocks=s.get('blocks'), vec_len=s.get('vec_len'), report=s.get('conversion_report', []))
        else:
            row['ok'] = False
            row['notes'].append('run failed (rc %d)' % rc)
        if s is not None and 'objective' in c['check']:
            ref, sign = c['reference_objective'], c['objective_sign']
            scale = max(1.0, abs(ref))
            row['rel_p'] = abs(sign * s['pobj'] - ref) / scale
            row['rel_d'] = abs(sign * s['dobj'] - ref) / scale
            row['converged'] = bool(s.get('converged', False))
            if not (row['rel_p'] <= a.rtol and row['rel_d'] <= a.rtol):
                row['ok'] = False
                row['notes'].append('objective off: rel_p %.2e rel_d %.2e' % (row['rel_p'], row['rel_d']))
            if not row['converged']:
                row['ok'] = False
                row['notes'].append('not converged (%s)' % s.get('status'))
        if s is not None and 'cross' in c['check']:
            base = c.get('cross_base', c['group'] + '__txt')
            rep = os.path.join(d, 'cross_vs_%s.json' % base)
            r = subprocess.run([a.python, '-B', COMPARE, 'compare', os.path.join(a.out, base), d, '--rtol', repr(float(c.get('cross_rtol', 0.0))),
                                '--report', rep], capture_output=True, text=True)
            row['cross_base'] = base
            row['cross_rc'] = r.returncode
            # the verdict on the history alone (the run directories carry no solution files: the tool's own
            # verdict also counts their absence): same length, no differing iteration in any history field
            try:
                rep = json.load(open(rep))
                h = rep.get('history', {})
                hist_ok = (h.get('iterations_a') == h.get('iterations_b') and 'per_field' in h and
                           all(h['per_field'][f]['differing_iterations'] == 0 for f in compare_runs.HISTORY_FIELDS if f in h['per_field']) and
                           any(f in h['per_field'] for f in compare_runs.HISTORY_FIELDS))
                row['cross_history'] = dict(iterations=(h.get('iterations_a'), h.get('iterations_b')),
                                            first_difference=h.get('first_difference'),
                                            max_rel_diff={f: h['per_field'][f]['max_rel_diff'] for f in h.get('per_field', {})})
            except (IOError, ValueError, KeyError) as e:
                hist_ok = False
                row['cross_history'] = 'unreadable report: %s' % e
            row['cross_ok'] = bool(hist_ok)
            if not hist_ok:
                row['ok'] = False
                row['notes'].append('history differs from %s: %s' % (base, row['cross_history']))
        failures += not row['ok']
        rows.append(row)
        print('%-18s %-7s %-15s rc=%d %s it=%s status=%s pobj=%s dobj=%s %s %s' % (
            row['name'], row['format'], row['check'], rc, 'OK ' if row['ok'] else 'BAD', row.get('iterations'), row.get('status'),
            row.get('pobj'), row.get('dobj'), ('rel_p=%.1e rel_d=%.1e' % (row['rel_p'], row['rel_d'])) if 'rel_p' in row else '',
            ('cross vs %s: %s' % (row['cross_base'], 'identical' if row.get('cross_ok') else 'DIFFERENT')) if 'cross_rc' in row else ''), flush=True)
        for n in row['notes']:
            print('    ' + n, flush=True)
    with open(os.path.join(a.out, 'report.json'), 'w') as f:
        json.dump(dict(spec=a.spec, exe=a.exe, rtol=a.rtol, cases=rows), f, indent=1)
    print('%d of %d cases pass' % (len(rows) - failures, len(rows)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
