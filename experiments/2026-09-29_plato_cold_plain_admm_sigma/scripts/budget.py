#!/usr/bin/env python3
"""H200-hours used by this campaign: allocated H200 GPUs x elapsed time of every GPU job listed in logs/gpu_jobs.txt
(sacct; idle time included), and core-hours of the CPU jobs in logs/cpu_jobs.txt. Appends to logs/budget.log.
usage: budget.py [--projected-h <additional H200-hours>]"""
import argparse, os, re, subprocess, time

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_JOBS = open(os.path.join(EXP, 'logs', 'gpu_jobs.txt')).read().split()
CPU_JOBS = open(os.path.join(EXP, 'logs', 'cpu_jobs.txt')).read().split()
LIMIT_H = 48.0


def elapsed_h(job):
    out = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job, '-o', 'ElapsedRaw,AllocTRES,State'], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip().splitlines()
    if not out:
        return 0.0, 0, 0, '?'
    raw, tres, state = out[0].split('|')
    g = re.search(r'gres/gpu=(\d+)', tres)
    c = re.search(r'cpu=(\d+)', tres)
    return int(raw) / 3600, int(g.group(1)) if g else 0, int(c.group(1)) if c else 0, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--projected-h', type=float, default=0.0)
    a = ap.parse_args()
    gpu_h, cpu_h, lines = 0.0, 0.0, []
    for j in GPU_JOBS:
        h, g, c, st = elapsed_h(j)
        gpu_h += h * g
        lines.append(f'  GPU job {j}: {h:.2f} h x {g} H200 = {h * g:.2f} H200-h ({st})')
    for j in CPU_JOBS:
        h, g, c, st = elapsed_h(j)
        cpu_h += h * c
        lines.append(f'  CPU job {j}: {h:.2f} h x {c} cores = {h * c:.1f} core-h ({st})')
    total = gpu_h + a.projected_h
    msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} used {gpu_h:.2f} H200-h; projected additional {a.projected_h:.2f}; total {total:.2f} of {LIMIT_H:g}"
           f"{'  EXCEEDS THE LIMIT: stop and ask' if total > LIMIT_H else ''}; CPU {cpu_h:.1f} core-h")
    print(msg)
    print('\n'.join(lines))
    with open(os.path.join(EXP, 'logs', 'budget.log'), 'a') as f:
        f.write(msg + '\n' + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
