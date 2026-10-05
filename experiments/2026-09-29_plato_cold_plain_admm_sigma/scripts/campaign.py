#!/usr/bin/env python3
"""Runs the solves of a plan (plans/<stage>.json) one at a time on the GPU of the allocation (one H200 per solve).
Resumable: a run with runs/<stage>/<run_id>/run.json is skipped.

Per run: small outputs in the experiment directory (runs/<stage>/<run_id>/: run.json, summary.json,
validation_history.csv, sigma_log.csv, stdout.log.gz), large ones in BIG (history.csv.gz; returned/ and final/
iterates, gzipped). Stage 3: after rep 1 of every configuration of a dataset, the configurations whose rep 1 took
less than 10 minutes end to end get reps 2 and 3, interleaved with each other.
usage: campaign.py <plan.json> [--dry-run] [--stop-file <path>]"""
import argparse, gzip, json, os, shutil, socket, subprocess, sys, time

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIG = os.path.expanduser('~/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma')
TXT = os.path.expanduser('~/cuadmm-data/plato_kocvara/txt')
BUILD = os.path.expanduser('~/cuadmm-builds/plato_official')
EXE = os.path.join(BUILD, 'cuadmm_exe')
THREADS = 16
REP_THRESHOLD_S = 600
OOM_MARKERS = ('out of memory', 'cudaErrorMemoryAllocation', 'std::bad_alloc', 'CUDA error 2 ', 'CUSOLVER_STATUS_ALLOC_FAILED', 'oom-kill', 'Killed')


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    with open(os.path.join(EXP, 'logs', 'campaign_timeline.log'), 'a') as f:
        f.write(line + '\n')


def gpu_info():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,memory.used,memory.total,temperature.gpu,clocks.sm',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # recorded, not fatal
        return f'nvidia-smi failed: {e}'


def gzip_file(p):
    if os.path.exists(p):
        with open(p, 'rb') as fi, gzip.open(p + '.gz', 'wb', compresslevel=4) as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(p)


def run_one(e, dry):
    small = os.path.join(EXP, 'runs', e['stage'], e['run_id'])
    big = os.path.join(BIG, 'runs', e['stage'], e['run_id'])
    if os.path.exists(os.path.join(small, 'run.json')):
        return json.load(open(os.path.join(small, 'run.json')))
    cmd = [EXE, os.path.join(TXT, e['dataset']), '--algorithm', 'admm', '--sigma-policy', e['policy'], '--sig', repr(e['sigma']),
           '--tol', '1e-4', '--validate-interval', str(e['validate_interval']), '--validate-tol', '1e-4',
           '--validate-threads', str(THREADS), '--max-iter', '1000000000', '--time-limit', str(e['time_limit_s']),
           '--summary', os.path.join(small, 'summary.json'), '--validation-history', os.path.join(small, 'validation_history.csv'),
           '--sigma-log', os.path.join(small, 'sigma_log.csv'), '--history', os.path.join(big, 'history.csv'),
           '--save-solution', os.path.join(big, 'returned'), '--save-final-dir', os.path.join(big, 'final'), '--tag', e['run_id']]
    if e['strict_dimacs_tol'] > 0:
        cmd += ['--strict-dimacs-tol', repr(e['strict_dimacs_tol'])]
    if dry:
        print(' '.join(cmd))
        return None
    os.makedirs(small, exist_ok=True)
    os.makedirs(big, exist_ok=True)
    for f in ('summary.json',):  # a partial file of an interrupted attempt would be appended to
        if os.path.exists(os.path.join(small, f)):
            os.rename(os.path.join(small, f), os.path.join(small, f + '.interrupted.' + time.strftime('%H%M%S')))
    g0 = gpu_info()
    log(f"start {e['run_id']} ({e['dataset']}, {e['config']}, sigma {e['sigma']:g}, limit {e['time_limit_s']} s, validate every {e['validate_interval']})")
    t0, start = time.time(), time.strftime('%Y-%m-%dT%H:%M:%S')
    # watchdog: the solver's own limit plus a margin for loading, the final validation and saving
    watchdog = 1.5 * e['time_limit_s'] + 900
    hung = False
    with open(os.path.join(small, 'stdout.log'), 'w') as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=watchdog)
        except subprocess.TimeoutExpired:
            hung = True
            proc.kill()
            proc.wait()
    p = proc
    wall = time.time() - t0
    text = open(os.path.join(small, 'stdout.log'), errors='replace').read()
    summ = None
    if os.path.exists(os.path.join(small, 'summary.json')):
        lines = [l for l in open(os.path.join(small, 'summary.json')) if l.strip()]
        summ = json.loads(lines[-1]) if lines else None
    oom = p.returncode != 0 and (any(m in text for m in OOM_MARKERS) or p.returncode in (137, -9))
    outcome = 'completed' if (p.returncode == 0 and summ) else ('hung' if hung else ('out_of_memory' if oom else 'crashed'))
    rec = dict(e, command=cmd, start=start, end=time.strftime('%Y-%m-%dT%H:%M:%S'), wall_s=wall, exit_code=p.returncode, outcome=outcome,
               watchdog_s=watchdog, host=socket.gethostname(), gpu_before=g0, gpu_after=gpu_info(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
               stop_reason=summ.get('stop_reason') if summ else None, iterations=summ.get('iterations') if summ else None,
               solve_time_s=summ.get('solve_time_s') if summ else None)
    gzip_file(os.path.join(small, 'stdout.log'))
    gzip_file(os.path.join(small, 'validation_history.csv'))
    gzip_file(os.path.join(small, 'sigma_log.csv'))
    gzip_file(os.path.join(big, 'history.csv'))
    for sub in ('returned', 'final'):
        for v in ('X.txt', 'y.txt', 'S.txt', 'certificate.txt'):
            gzip_file(os.path.join(big, sub, v))
    json.dump(rec, open(os.path.join(small, 'run.json'), 'w'), indent=1)
    log(f"end   {e['run_id']}: {outcome}, exit {p.returncode}, wall {wall:.1f} s, stop {rec['stop_reason']}, iterations {rec['iterations']}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--stop-file', default=os.path.join(EXP, 'STOP'))
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    os.makedirs(os.path.join(EXP, 'logs'), exist_ok=True)
    if not a.dry_run:
        log(f"plan {a.plan}: {len(plan['runs'])} runs; exe {EXE}; host {socket.gethostname()}; GPU {gpu_info()}")
    queue = list(plan['runs'])
    done_rep1 = {}
    while queue:
        if os.path.exists(a.stop_file):
            log(f'stop file {a.stop_file} present: stopping before {queue[0]["run_id"]}')
            return 3
        e = queue.pop(0)
        rec = run_one(e, a.dry_run)
        if rec is None or plan['stage'] != 'stage3' or e['rep'] != 1:
            continue
        # repetitions: once every configuration of this dataset has its rep 1
        done_rep1.setdefault(e['dataset'], []).append(rec)
        n_cfg = sum(1 for r in plan['runs'] if r['dataset'] == e['dataset'] and r['rep'] == 1)
        if len(done_rep1[e['dataset']]) == n_cfg:
            short = [r for r in done_rep1[e['dataset']] if r['outcome'] == 'completed' and r['wall_s'] < REP_THRESHOLD_S]
            extra = [dict({k: v for k, v in r.items() if k in plan['runs'][0]}, rep=rep, run_id=r['run_id'][:-3] + f'_r{rep}')
                     for rep in (2, 3) for r in short]
            if extra:
                log(f"{e['dataset']}: reps 2-3 for {', '.join(r['config'] for r in short)} (rep 1 < {REP_THRESHOLD_S} s)")
            queue = extra + queue
    if not a.dry_run:
        log(f"plan {a.plan} finished")
    return 0


if __name__ == '__main__':
    sys.exit(main())
