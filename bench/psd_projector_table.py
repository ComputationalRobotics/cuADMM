#!/usr/bin/env python3
"""Markdown tables of docs/phase6a_projector_benchmark.md from the results.csv of cuadmm_psd_projector_bench.

usage: python -B bench/psd_projector_table.py RESULTS_DIR [--timings]
Prints (1) the per-case table (time per projection, speedup, relative error, eigenvalue violation, rejections,
Lanczos estimate quality), (2) the per-size memory table (w_evd(n) / n^2, host workspace, memory of each path),
and with --timings (3) the per-run table from timings.csv. Only the standard library is used.
"""
import csv
import math
import sys
from pathlib import Path


def f(x, fmt):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if math.isnan(v):
        return "nan"
    return format(v, fmt)


def rows_of(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def case_table(rows):
    out = []
    out.append("| n | family | FullEvd ms | Composite ms | of which Lanczos / filter / checks ms | speedup | "
               "rel. Frobenius error median / max | lambda_min(P_fp16) | rejected | Lanczos up / true (min..max) | "
               "scale / true (min..max) |")
    out.append("|---:|---|---:|---:|---|---:|---|---:|---:|---|---|")
    for r in rows:
        if r["status"] != "ok":
            out.append(f"| {r['n']} | {r['family']} | {r['status']} | | | | | | | | |")
            continue
        out.append(
            f"| {r['n']} | {r['family']} | {f(r['evd_ms_median'], '.1f')} | {f(r['comp_ms_median'], '.1f')} | "
            f"{f(r['lanczos_ms_median'], '.1f')} / {f(r['filter_ms_median'], '.1f')} / {f(r['checks_ms_median'], '.1f')} | "
            f"{f(r['speedup'], '.2f')} | {f(r['rel_fro_err_median'], '.2e')} / {f(r['rel_fro_err_max'], '.2e')} | "
            f"{f(r['lambda_min_p16'], '.2e')} | {r['rejected']}/{r['runs']} | "
            f"{f(r['up_over_true_min'], '.5f')}..{f(r['up_over_true_max'], '.5f')} | "
            f"{f(r['scale_over_true_min'], '.4f')}..{f(r['scale_over_true_max'], '.4f')} |")
    return "\n".join(out)


def memory_table(rows):
    gib = 1024.0 ** 3
    seen = {}
    for r in rows:
        if r["status"] == "ok" and r["n"] not in seen:
            seen[r["n"]] = r
    out = ["| n | w_evd(n) device bytes | w_evd(n) / n^2 (doubles) | Xsyevd host workspace bytes | FullEvd path GiB "
           "(workspace + W + info + T + P) | FullEvd path / n^2 | CompositeFilter path GiB (FP32 + FP16 workspaces + P) | "
           "CompositeFilter path / n^2 | runs per path | true ||Z||_2 (full_rank) | r+ / r- (full_rank) |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for n, r in seen.items():
        out.append(
            f"| {n} | {r['evd_ws_dev_bytes']} | {f(r['evd_ws_over_n2'], '.4f')} | {r['evd_ws_host_bytes']} | "
            f"{float(r['evd_mem_bytes']) / gib:.3f} | {f(r['evd_mem_over_n2'], '.3f')} | {float(r['comp_mem_bytes']) / gib:.3f} | "
            f"{f(r['comp_mem_over_n2'], '.3f')} | {r['reps']} | {f(r['true_norm2'], '.6f')} | {r['r_plus']} / {r['r_minus']} |")
    return "\n".join(out)


def timings_table(rows):
    out = ["| n | family | path | run | ms (cudaEvent) | wall ms | Lanczos ms | filter ms | lo | up | scale | accepted | rel. error |",
           "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        run = "warm-up" if r["warmup"] == "1" else r["rep"]
        out.append(
            f"| {r['n']} | {r['family']} | {r['path']} | {run} | {f(r['ms'], '.1f')} | {f(r['wall_ms'], '.1f')} | "
            f"{f(r['lanczos_ms'], '.1f')} | {f(r['filter_ms'], '.1f')} | {f(r['lanczos_lo'], '.6f')} | {f(r['lanczos_up'], '.6f')} | "
            f"{f(r['scale'], '.6f')} | {r['accepted']} | {f(r['rel_fro_err'], '.2e')} |")
    return "\n".join(out)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = Path(argv[1])
    rows = rows_of(d / "results.csv")
    print(case_table(rows))
    print()
    print(memory_table(rows))
    if "--timings" in argv:
        print()
        print(timings_table(rows_of(d / "timings.csv")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
