#!/bin/bash
# The arithmetic of main (bit for bit the reference builds on the baseline cases): usage: run_bitwise_main.sh <build dir> <problem> [out dir]
set -u
BLD=$1; PROBLEM=$2; OUT=${3:-out_main}
mkdir -p "$OUT"
"$BLD/cuadmm_exe" "$PROBLEM" --psd-warmup none --psd-final-product gemm_compat --psd-batched-final-product gemm_compat \
  --policy residual_balancing --sigma0 100 --tol 1e-6 \
  --summary "$OUT/summary.json" --history "$OUT/history.csv" --save-solution "$OUT/solution"
