#!/bin/bash
# The defaults of cuADMM 2.0 on one problem: usage: examples/run_defaults.sh <build dir> <problem> [out dir]
set -u
BLD=$1; PROBLEM=$2; OUT=${3:-out_defaults}
mkdir -p "$OUT"
"$BLD/cuadmm_exe" "$PROBLEM" --tol 1e-6 --verbosity 1 \
  --summary "$OUT/summary.json" --history "$OUT/history.csv" --save-solution "$OUT/solution" --sigma-log "$OUT/sigma_log.csv"
