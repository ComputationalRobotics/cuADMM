#!/bin/bash
# Stage 0 on the H200 node: environment capture, device test, official build of the committed branch (flags of
# build_version.sh), the complete test suite, compute-sanitizer and the bit-identity comparison with 09687d2.
# usage: stage0.sh [env|build|tests|sanitizers|compare|all]
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
SRC=$(cd $EXP/../.. && pwd)
BLD=$HOME/cuadmm-builds/plato_official
REF=$HOME/cuadmm-builds/hybrid_official    # 09687d2, the hybrid campaign's official build (same flags)
TXT=$HOME/cuadmm-data/plato_kocvara/txt
BIG=$HOME/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma
CS=$(dirname $(which nvcc))/compute-sanitizer
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
what=${1:-all}
mkdir -p $EXP/logs/stage0 $BIG

if [ $what = env ] || [ $what = all ]; then
  bash $EXP/scripts/capture_env.sh $EXP/env
  $HOME/cuadmm-tools/ctx > $EXP/logs/stage0/device_test.txt 2>&1; cat $EXP/logs/stage0/device_test.txt
fi

if [ $what = build ] || [ $what = all ]; then
  # the tree must be clean in every source path (the experiment directory is untracked data)
  if [ -n "$(git -C $SRC status --porcelain -- src include test CMakeLists.txt psd_projection MATLAB examples)" ]; then
    echo "source tree not clean"; git -C $SRC status --short; exit 1
  fi
  [ -e $BLD ] && { echo "$BLD exists: the official build is frozen; not rebuilding"; exit 1; }
  JOBS=16 bash $EXP/scripts/build_version.sh official $SRC $BLD || exit 1
fi

if [ $what = tests ] || [ $what = all ]; then
  cd $BLD && ./tests > $EXP/logs/stage0/official_full_suite.log 2>&1
  echo "official test suite exit code $?: $(grep -E '^\[  PASSED|^\[  FAILED|tests ran' $EXP/logs/stage0/official_full_suite.log | tr '\n' ' ')"
  grep -E '^\[  FAILED|^\[  SKIPPED' $EXP/logs/stage0/official_full_suite.log | head -20
fi

if [ $what = sanitizers ] || [ $what = all ]; then
  OUT=$EXP/logs/sanitizers; mkdir -p $OUT $BIG/sanitizers
  $CS --version | head -2 > $OUT/compute_sanitizer_version.txt
  CASES=(gtest_plain trto2_adaptive_strict mater2_fixed buck4_adaptive trto5_fixed_lobpcg)
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
  cd $BLD
  for tool in memcheck initcheck synccheck; do
    for c in "${CASES[@]}"; do
      extra=""; [ $tool = memcheck ] && extra="--leak-check full"
      t0=$(date +%s)
      $CS --tool $tool $extra --error-exitcode 99 $(args $c $tool) > $OUT/${tool}__$c.log 2>&1
      rc=$?
      echo "$tool $c: exit $rc, $(( $(date +%s) - t0 )) s | $(grep -E 'ERROR SUMMARY|LEAK SUMMARY' $OUT/${tool}__$c.log | tr '\n' ' ') | $(grep -E 'Solver ended|PASSED|FAILED|lobpcg' $OUT/${tool}__$c.log | tail -1)"
    done
  done
fi

if [ $what = compare ] || [ $what = all ]; then
  bash $EXP/scripts/compare_builds.sh
fi
echo "== stage0 $what done"
