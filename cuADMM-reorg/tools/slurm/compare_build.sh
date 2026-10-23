#!/bin/bash
# Bitwise comparison of a cuADMM-reorg build with the three reference builds on every baseline case (plan "bitwise"
# acceptance of the structural phases), then the test suite of that build. Run on the GPU node.
# usage: compare_build.sh <build dir> <config name> <tag> [kind]
#   kind: how tools/compare_runs.py drives the build's CLI: legacy@pre0 (the copied CLI from Pre-0 on) or reorg (Phase 2+)
# Outputs: $CUADMM_BASELINES/<config>/compare/<tag>/{fixed,legacy_adaptive,residual_balancing,sgs}/, tests.log
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
source "$HERE/tools/slurm/gpu_env.sh"
BLD=$1; CONFIG=$2; TAG=$3; KIND=${4:-legacy@pre0}
OUT=$CUADMM_BASELINES/$CONFIG/compare/$TAG
mkdir -p "$OUT"
CMP="$PYTHON -B $HERE/tools/compare_runs.py"
SRC_OF_BUILD=$(grep -m1 '^CMAKE_HOME_DIRECTORY' "$BLD/CMakeCache.txt" | cut -d= -f2)
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; nvidia-smi -L
  echo "build: $BLD from $SRC_OF_BUILD ($(git -C "$SRC_OF_BUILD" rev-parse HEAD), $(git -C "$SRC_OF_BUILD" status --porcelain -- . | wc -l) local changes)"
  echo "threads: $OMP_NUM_THREADS"; echo "kind: $KIND"; echo "extra_b: ${CUADMM_EXTRA_B:-}"; echo "rtol: ${CUADMM_RTOL:-0}"
} > "$OUT/config.txt"
cat "$OUT/config.txt"
R9=legacy@9bb4ba5:$CUADMM_REF_BUILDS/ref-9bb4ba5/cuadmm_exe
R2=legacy@2d8a755:$CUADMM_REF_BUILDS/ref-2d8a755/cuadmm_exe
RS=legacy@2d8a755-sgs:$CUADMM_REF_BUILDS/ref-2d8a755-sgs/cuadmm_exe
NEW=$KIND:$BLD/cuadmm_exe
# CUADMM_EXTRA_B: extra CLI flags of the new build (e.g. the compat products of Phase 6b); CUADMM_RTOL: within-rounding comparison
EXTRA=()
[ -n "${CUADMM_EXTRA_B:-}" ] && EXTRA+=(--extra-b "$CUADMM_EXTRA_B")
[ -n "${CUADMM_RTOL:-}" ] && EXTRA+=(--rtol "$CUADMM_RTOL")
rc=0
$CMP batch --spec "$HERE/tools/baselines/cases_fixed.json" --problem-root "$CUADMM_DATA" --build-a $R2 --build-b $NEW "${EXTRA[@]}" --out "$OUT/fixed" || rc=1
$CMP batch --spec "$HERE/tools/baselines/cases_legacy_adaptive.json" --problem-root "$CUADMM_DATA" --build-a $R9 --build-b $NEW "${EXTRA[@]}" --out "$OUT/legacy_adaptive" || rc=1
$CMP batch --spec "$HERE/tools/baselines/cases_residual_balancing.json" --problem-root "$CUADMM_DATA" --build-a $R2 --build-b $NEW "${EXTRA[@]}" --out "$OUT/residual_balancing" || rc=1
$CMP batch --spec "$HERE/tools/baselines/cases_sgs.json" --problem-root "$CUADMM_DATA" --build-a $RS --build-b $NEW "${EXTRA[@]}" --out "$OUT/sgs" || rc=1
echo "=== comparison exit $rc ($(date))"
"$HERE/tools/slurm/run_tests.sh" "$BLD" "$OUT/tests.log"
echo "=== tests exit $? ($(date))"
exit $rc
