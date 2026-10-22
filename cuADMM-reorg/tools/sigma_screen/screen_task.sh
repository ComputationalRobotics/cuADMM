#!/bin/bash
# One task of the Phase-4b screen array (submitted by screen.sh): runs or resumes the spec on line
# SLURM_ARRAY_TASK_ID + 1 of SCREEN_PENDING with the CPU replica pinned in SCREEN_ROOT.
set -u
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SCREEN_PENDING")
read -r name ds rule sig cap vint <<< "$line"
out=$SCREEN_ROOT/runs/$name
log=$SCREEN_ROOT/logs/$name.log
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -f "$out/summary.json" ] && grep -q '"finished": true' "$out/summary.json"; then
  echo "$(date +%FT%T) $name already finished" >> "$log"
  exit 0
fi
if [ -f "$out/state.npz" ]; then
  args="resume $out --wall-limit $SCREEN_WALL --checkpoint-every 120"
else
  args="run $SCREEN_DATA/$ds --out $out --rule $rule --sig $sig --tol 1e-4 --max-iter $cap --validate-interval $vint --validate-tol 1e-4 --strict-dimacs-tol 1e-6 --history-stride 10 --gzip-validation --wall-limit $SCREEN_WALL --checkpoint-every 120"
fi
echo "$(date +%FT%T) $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?}: $args" >> "$log"
"$SCREEN_PYTHON" -B "$SCREEN_ROOT/cuadmm_emu.py" $args >> "$log" 2>&1
rc=$?
echo "$(date +%FT%T) exit $rc" >> "$log"
exit $rc
