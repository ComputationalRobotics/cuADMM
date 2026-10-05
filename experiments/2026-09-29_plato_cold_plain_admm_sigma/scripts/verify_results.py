#!/usr/bin/env python3
"""Verifies the result files of a stage against the raw logs. Per run:
  1. the driver record (run.json), the solver summary (summary.json) and the solver's stdout agree on the number of
     iterations and the stop reason ("plain ADMM iterations = N", "stop reason: X");
  2. results/<stage>_runs.csv carries the same iterations, stop reason, final sigma and, for the returned iterate, the
     same max_abs_DIMACS / eta / status as the independent revalidation (postval_returned.json);
  3. the returned iterate (X, y, S) was saved, and the final iterate too whenever the solve ended at the time limit;
  4. a fixed-sigma run never changed sigma: the sigma column of the per-iteration history (history.csv.gz) is constant
     and equal to the initial sigma, and the sigma log is empty;
  5. the time limit held: solve time <= limit + the final validation + the final-iterate save callback + 5 s;
  6. TIMEOUT exactly when the practical target was not reached and the solve ended at the time limit;
  7. repetitions of a run that stopped on a validated target have identical iteration counts.
Writes logs/verification_<stage>.txt; exit status 1 if any check fails.  usage: verify_results.py <stage>"""
import csv, glob, gzip, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

EXP = A.EXP
BIG = os.path.expanduser('~/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma')


def main():
    stage = sys.argv[1]
    table = {r['run_id']: r for r in csv.DictReader(open(os.path.join(A.RES, f'{stage}_runs.csv')))}
    fails, n, lines = [], 0, []
    reps = {}
    for d in sorted(glob.glob(os.path.join(EXP, 'runs', stage, '*'))):
        if not os.path.exists(os.path.join(d, 'run.json')):
            continue
        n += 1
        r = json.load(open(os.path.join(d, 'run.json')))
        rid = r['run_id']
        bad = lambda msg: fails.append(f'{rid}: {msg}')
        s = json.loads(open(os.path.join(d, 'summary.json')).read().strip().splitlines()[-1])
        out = gzip.open(os.path.join(d, 'stdout.log.gz'), 'rt', errors='replace').read()
        m_it = re.findall(r'plain ADMM iterations = (\d+)', out)
        m_stop = re.findall(r'stop reason: (\S+)', out)
        # 1
        if r['iterations'] != s['iterations'] or not m_it or int(m_it[-1]) != s['iterations']:
            bad(f"iterations: run.json {r['iterations']}, summary {s['iterations']}, stdout {m_it[-1:] }")
        if r['stop_reason'] != s['stop_reason'] or not m_stop or m_stop[-1] != s['stop_reason']:
            bad(f"stop reason: run.json {r['stop_reason']}, summary {s['stop_reason']}, stdout {m_stop[-1:]}")
        # 2
        t = table.get(rid)
        pv = json.load(open(os.path.join(d, 'postval_returned.json'))) if os.path.exists(os.path.join(d, 'postval_returned.json')) else None
        if t is None:
            bad('missing from the results table')
        else:
            if int(t['iterations']) != s['iterations'] or t['stop_reason'] != s['stop_reason'] or float(t['sigma_final']) != s['final_sig']:
                bad('results table differs from the summary (iterations / stop reason / final sigma)')
            if pv is None:
                bad('no independent revalidation of the returned iterate')
            else:
                for k in ('max_abs_dimacs', 'eta'):
                    if not math.isclose(float(t[f'returned_{k}']), pv[k], rel_tol=1e-12, abs_tol=0):
                        bad(f'results table returned_{k} {t[f"returned_{k}"]} != revalidation {pv[k]}')
                if not (t['status'] == pv['status'] or (t['status'] == 'TIMEOUT' and pv['status'] == 'NOT_VALIDATED')):
                    bad(f"status {t['status']} vs revalidation {pv['status']}")
        # 3
        big = os.path.join(BIG, 'runs', stage, rid)
        for v in ('X', 'y', 'S'):
            if not os.path.exists(os.path.join(big, 'returned', v + '.txt.gz')):
                bad(f'returned {v} missing')
            if s['stop_reason'] == 'time_limit' and not os.path.exists(os.path.join(big, 'final', v + '.txt.gz')):
                bad(f'final {v} missing')
        # 4
        if r['policy'] == 'fixed':
            sig = set()
            with gzip.open(os.path.join(big, 'history.csv.gz'), 'rt') as f:
                for row in csv.DictReader(f):
                    sig.add(float(row['sig']))
            if sig != {r['sigma']} or s['sigma_log_entries'] != 0 or s['sigma_changes'] != 0:
                bad(f'fixed sigma changed: history sigma values {sorted(sig)[:5]}, log entries {s["sigma_log_entries"]}')
        # 5
        vh = A.read_csv(os.path.join(d, 'validation_history.csv'))
        last_val = A.fnum(vh[-1]['validation_s']) if vh else 0.0
        # the solve clock of a run stopped by the time limit also contains its final validation and the --save-final-dir
        # callback that writes the final iterate (16 s for shmup5); the check-granularity is one iteration
        if s['solve_time_s'] > r['time_limit_s'] + last_val + s['callback_time_s'] + 5.0:
            bad(f"solve time {s['solve_time_s']:.1f} s exceeds the limit {r['time_limit_s']} s + final validation {last_val:.1f} s + callback {s['callback_time_s']:.1f} s")
        # 6
        if t is not None:
            timeout_expected = (not s['practical_validated']) and s['stop_reason'] == 'time_limit'
            if (t['status'] == 'TIMEOUT') != timeout_expected:
                bad(f"TIMEOUT status {t['status']} inconsistent with practical_validated {s['practical_validated']} / stop {s['stop_reason']}")
        reps.setdefault((r['dataset'], r['config']), []).append((r['rep'], s['iterations'], s['stop_reason']))
    # 7
    for k, v in reps.items():
        its = {it for _, it, st in v if st in ('strict_validated', 'externally_validated')}
        if len(its) > 1:
            fails.append(f'{k}: validated repetitions differ in iterations {sorted(v)}')
    lines.append(f'{stage}: {n} runs checked, {len(reps)} dataset/configuration groups, {sum(len(v) for v in reps.values())} runs incl. repetitions')
    lines.append('ALL CHECKS PASSED' if not fails else f'{len(fails)} FAILURES:\n' + '\n'.join(fails))
    open(os.path.join(EXP, 'logs', f'verification_{stage}.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
