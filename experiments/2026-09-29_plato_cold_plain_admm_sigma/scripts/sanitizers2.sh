#!/bin/bash
# Second sanitizer pass (after stage0.sh sanitizers was stopped during initcheck; see PROTOCOL.md, Deviations):
#  1. initcheck on the two large-block cases with cuSOLVER's internal tridiagonalization kernels excluded
#     (--kernel-name-exclude kns=sytrd: sytrd4_gpu and epilogue<sytrd_params...>, both launched by cusolverDnXsyevd).
#     All 100 printed reports of the unfiltered buck4 run came from sytrd4_gpu; with only sytrd4_gpu excluded (first
#     attempt, interrupted when the session ended) all 100 printed reports came from epilogue<sytrd_params...>. The
#     filtered run shows whether any other kernel (ours, cuBLAS, cuSPARSE, the rest of cuSOLVER) reads
#     uninitialized memory. Each case has a 30-minute timeout (exit 124 = timed out).
#  2. synccheck on all five cases (not reached by the first pass)
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
B=$HOME/cuadmm-builds/plato_official
TXT=$HOME/cuadmm-data/plato_kocvara/txt
BIG=$HOME/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma
CS=$(dirname $(which nvcc))/compute-sanitizer
OUT=$EXP/logs/sanitizers; mkdir -p $OUT $BIG/sanitizers
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
args() {
  local o=$BIG/sanitizers/$1_$2
  case $1 in
    gtest_plain) echo "./tests --gtest_filter=PlainAdmmOnly*:DimacsValidator.*:StrictStage.*" ;;
    trto2_adaptive_strict) echo "./cuadmm_exe $TXT/trto2 --algorithm admm --sigma-policy legacy_adaptive --sig 1 --tol 1e-4 --max-iter 400 --validate-interval 100 --validate-threads 4 --strict-dimacs-tol 1e-6 --sigma-log $o.sigma.csv --save-final-dir $o.final --save-solution $o.ret" ;;
    mater2_fixed) echo "./cuadmm_exe $TXT/mater-2 --algorithm admm --sigma-policy fixed --sig 1 --tol 1e-4 --max-iter 300 --validate-interval 100 --validate-threads 4 --strict-dimacs-tol 1e-6" ;;
    buck4_adaptive) echo "./cuadmm_exe $TXT/buck4 --algorithm admm --sigma-policy legacy_adaptive --sig 1 --tol 1e-4 --max-iter 200 --validate-interval 100 --validate-threads 4" ;;
    trto5_fixed_lobpcg) echo "./cuadmm_exe $TXT/trto5 --algorithm admm --sigma-policy fixed --sig 1 --tol 1e-4 --max-iter 250 --validate-interval 125 --validate-threads 4" ;;
  esac
}
run() {  # tool case label extra...
  local tool=$1 c=$2 label=$3; shift 3
  t0=$(date +%s)
  timeout 1800 $CS --tool $tool "$@" --error-exitcode 99 $(args $c $tool$label) > $OUT/${tool}${label}__$c.log 2>&1
  rc=$?
  echo "$tool$label $c: exit $rc, $(( $(date +%s) - t0 )) s | $(grep -E 'ERROR SUMMARY|LEAK SUMMARY' $OUT/${tool}${label}__$c.log | tr '\n' ' ') | $(grep -E 'Solver ended|PASSED|FAILED' $OUT/${tool}${label}__$c.log | tail -1)"
}
cd $B
for c in buck4_adaptive trto5_fixed_lobpcg; do
  run initcheck $c _excl_sytrd --kernel-name-exclude kns=sytrd
done
for c in gtest_plain trto2_adaptive_strict mater2_fixed buck4_adaptive trto5_fixed_lobpcg; do
  run synccheck $c ""
done
echo "== sanitizers2 done"
