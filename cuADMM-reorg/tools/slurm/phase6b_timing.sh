#!/bin/bash
# Phase 6b acceptance (plan Section 17): time per iteration of the 6b build against the 6a build, and the measured
# peak memory against the setup plan of the 6b build, on the generator's large-block cases and the local PLATO
# problems with the largest blocks. Run inside a GPU allocation (A100, Release builds).
# usage: phase6b_timing.sh <build_6a> <build_6b> <out_dir> [max_iter]
#   CUADMM_DATA: the data root (default $HOME/cuadmm-data); CUADMM_THREADS: OpenBLAS threads of the runs
set -u
A=$1; B=$2; OUT=$3; MAXIT=${4:-200}
DATA=${CUADMM_DATA:-$HOME/cuadmm-data}
export OMP_NUM_THREADS=${CUADMM_THREADS:-16} OPENBLAS_NUM_THREADS=${CUADMM_THREADS:-16}
mkdir -p "$OUT"
CASES="generated/Large_LowRankS generated/Large_HalfRank generated/Large_LowRankX plato_kocvara/txt/shmup5 plato_kocvara/txt/shmup4 plato_kocvara/txt/buck5 plato_kocvara/txt/trto5 plato_kocvara/txt/vibra5 plato_kocvara/txt/mater-5"
{
  echo "date: $(date)"; echo "node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
  echo "build_6a: $A"; echo "build_6b: $B"; echo "max_iter: $MAXIT"; echo "threads: $OMP_NUM_THREADS"
} > "$OUT/config.txt"
for c in $CASES; do
  name=$(basename "$c")
  [ -d "$DATA/$c" ] || { echo "missing problem $DATA/$c"; continue; }
  for tag in 6a 6b; do
    bld=$A; [ "$tag" = 6b ] && bld=$B
    for rep in 1 2; do
      d="$OUT/${name}_${tag}_r$rep"; mkdir -p "$d"
      "$bld/cuadmm_exe" "$DATA/$c" --sigma-policy fixed --sig 100 --max-iter "$MAXIT" --tol 1e-300 --algorithm admm \
        --streams 15 --lobpcg 1 --summary "$d/summary.json" > "$d/stdout.txt" 2>&1
      echo "$?" > "$d/exit_code.txt"
    done
  done
done
PYTHONDONTWRITEBYTECODE=1 python3 -B -I - "$OUT" <<'EOF'
import json, os, sys, glob
out = sys.argv[1]
rows = {}
for d in sorted(glob.glob(os.path.join(out, "*_r[12]"))):
    base = os.path.basename(d)
    name, tag, rep = base.rsplit("_", 2)
    sj = os.path.join(d, "summary.json")
    if not os.path.exists(sj):
        rows.setdefault(name, {}).setdefault(tag, []).append(None)
        continue
    s = json.load(open(sj))
    it = s.get("iterations") or 0
    t = s.get("admm_phase_time_s") or s.get("solve_time_s") or 0.0
    rows.setdefault(name, {}).setdefault(tag, []).append({"ms_per_iter": 1000.0 * t / it if it else float("nan"), "peak": s.get("peak_gpu_mem_mib"),
                                                          "plan": s.get("memory_plan_mib"), "iterations": it})
lines = ["| case | 6a ms/iter (best of 2) | 6b ms/iter (best of 2) | 6b / 6a | 6b peak MiB | 6b plan MiB | peak / plan |", "|---|---|---|---|---|---|---|"]
for name, tags in rows.items():
    def best(tag):
        xs = [x for x in tags.get(tag, []) if x]
        return min(xs, key=lambda x: x["ms_per_iter"]) if xs else None
    a, b = best("6a"), best("6b")
    if not a or not b:
        lines.append("| %s | failed run (see the stdout files) | | | | | |" % name)
        continue
    ratio = b["ms_per_iter"] / a["ms_per_iter"] if a["ms_per_iter"] else float("nan")
    pp = (b["peak"] / b["plan"]) if (b.get("peak") and b.get("plan")) else float("nan")
    lines.append("| %s | %.2f | %.2f | %.3f | %.1f | %.1f | %.3f |" % (name, a["ms_per_iter"], b["ms_per_iter"], ratio, b.get("peak") or 0, b.get("plan") or 0, pp))
open(os.path.join(out, "table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
echo "=== done ($(date))"
