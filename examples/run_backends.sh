#!/bin/bash
# The normal-equation backends on one problem (cudss needs a build with CUADMM_WITH_CUDSS): usage: run_backends.sh <build dir> <problem> [out dir]
set -u
BLD=$1; PROBLEM=$2; OUT=${3:-out_backends}
mkdir -p "$OUT"
for backend in cholmod dense cudss; do
  "$BLD/cuadmm_exe" "$PROBLEM" --backend $backend --tol 1e-6 --max-iter 2000 --summary "$OUT/summary_$backend.json" > "$OUT/stdout_$backend.txt" 2>&1 \
    && echo "$backend: $(grep -o '"iterations": [0-9]*' "$OUT/summary_$backend.json"), $(grep -o '"linsys_backend": "[^"]*"' "$OUT/summary_$backend.json")" \
    || echo "$backend: failed (see $OUT/stdout_$backend.txt)"
done
