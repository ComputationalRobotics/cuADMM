#!/bin/bash
# screen.sh <specfile> [max concurrent tasks (32)] [task time limit (00:15:00)]
# The Phase-4b CPU screen driver: submits one Slurm array task per spec of <specfile> whose run has not finished
# (runs/<name>/summary.json with "finished": true), each a 1-core job that runs or resumes the CPU replica
# (cuadmm_emu.py v1.2) with a wall limit 75 s under the task's time limit and checkpoints every 120 s. Call the script
# again for the same spec file to resume the runs a task did not finish. Never run the replica on a login node.
#
# Layout (SCREEN_ROOT, default ~/cuadmm-experiments/2026-10-21_phase4b_screen): runs/<name>/, logs/, specs/, results/,
# and the pinned copy cuadmm_emu.py (+ cuadmm_emu.sha256) taken from this directory on the first call; a later call
# whose repository copy differs warns and keeps using the pinned copy, so that every run of the screen uses one code.
# Environment: SCREEN_ROOT, SCREEN_DATA (PLATO TXT directories), SCREEN_PYTHON, SCREEN_PARTITION (seas_compute),
# SCREEN_CONSTRAINT (cascadelake: the sigma study's nodes, for comparable microseconds per iteration), SCREEN_MEM (4G).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${SCREEN_ROOT:-$HOME/cuadmm-experiments/2026-10-21_phase4b_screen}
DATA=${SCREEN_DATA:-$HOME/cuadmm-data/plato_kocvara/txt}
PY=${SCREEN_PYTHON:-$HOME/cuadmm-analysis-env/bin/python}
PARTITION=${SCREEN_PARTITION:-seas_compute}
CONSTRAINT=${SCREEN_CONSTRAINT:-cascadelake}
MEM=${SCREEN_MEM:-4G}
if [ $# -lt 1 ]; then sed -n 2,14p "$0"; exit 1; fi
specfile=$1; maxpar=${2:-32}; tlimit=${3:-00:15:00}
mkdir -p "$ROOT/runs" "$ROOT/logs" "$ROOT/specs" "$ROOT/results"
if [ ! -f "$ROOT/cuadmm_emu.py" ]; then
  cp "$HERE/cuadmm_emu.py" "$ROOT/cuadmm_emu.py"
  (cd "$ROOT" && sha256sum cuadmm_emu.py > cuadmm_emu.sha256)
  echo "pinned $HERE/cuadmm_emu.py as $ROOT/cuadmm_emu.py ($(cut -c1-16 "$ROOT/cuadmm_emu.sha256"))"
elif ! cmp -s "$HERE/cuadmm_emu.py" "$ROOT/cuadmm_emu.py"; then
  echo "WARNING: $HERE/cuadmm_emu.py differs from the pinned copy $ROOT/cuadmm_emu.py; the pinned copy is used" >&2
fi
stamp=$(date +%Y%m%d-%H%M%S)
pending=$ROOT/specs/$(basename "$specfile" .txt).pending.$stamp
: > "$pending"
total=0
while read -r name ds rule sig cap vint; do
  [[ -z "${name:-}" || "$name" == \#* ]] && continue
  total=$((total + 1))
  out=$ROOT/runs/$name
  if [ -f "$out/summary.json" ] && grep -q '"finished": true' "$out/summary.json"; then continue; fi
  echo "$name $ds $rule $sig $cap $vint" >> "$pending"
done < "$specfile"
n=$(wc -l < "$pending")
echo "$(date +%FT%T) $specfile: $total runs, $n pending"
if [ "$n" -eq 0 ]; then rm -f "$pending"; exit 0; fi
secs=$(echo "$tlimit" | awk -F: '{ if (NF == 3) print $1 * 3600 + $2 * 60 + $3; else print $1 * 60 + $2 }')
wall=$((secs - 75))
jobid=$(sbatch --parsable --array=0-$((n - 1))%$maxpar -p "$PARTITION" ${CONSTRAINT:+--constraint=$CONSTRAINT} -c 1 --mem="$MEM" -t "$tlimit" \
  -J "p4b_$(basename "$specfile" .txt)" -o "$ROOT/logs/slurm_%A_%a.out" \
  --export=ALL,SCREEN_PENDING="$pending",SCREEN_ROOT="$ROOT",SCREEN_DATA="$DATA",SCREEN_PYTHON="$PY",SCREEN_WALL="$wall" \
  "$HERE/screen_task.sh")
echo "$(date +%FT%T) submitted array job $jobid ($n tasks, at most $maxpar at a time, $tlimit each, wall limit $wall s): $pending"
echo "$jobid $pending" >> "$ROOT/specs/submissions.txt"
