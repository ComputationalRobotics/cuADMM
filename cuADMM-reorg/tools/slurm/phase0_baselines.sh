#!/bin/bash
# Phase 0 GPU job (plan Phase 0, developer prompt "Phase 0: skeleton and safety net"), run on the GPU node:
#   1. record the machine configuration;
#   2. generate the known-optimum TXT problems (cuadmm_gen_problems) next to the PLATO TXT problems;
#   3. self-checks: each reference build against itself (acceptance: zero differences);
#   4. baselines: the reference runs of tools/baselines/cases_*.json, each run twice (run-to-run determinism), and
#      the fixed-sigma cases also against the Phase 0 copy (legacy@c23e67e), which must match 2d8a755.
# usage: phase0_baselines.sh <phase0 build dir> [config name]
# Outputs: $CUADMM_BASELINES/<config>/{config.txt,selfcheck,fixed,legacy_adaptive,residual_balancing,sgs}/
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
source "$HERE/tools/slurm/gpu_env.sh"
BLD=$1
CONFIG=${2:-a100-sxm4-80gb_cuda12.9_gcc12.2_ss7.10.1}
OUT=$CUADMM_BASELINES/$CONFIG
mkdir -p "$OUT"
CMP="$PYTHON -B $HERE/tools/compare_runs.py"

echo "=== 1. configuration ($(date))"
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; nvidia-smi -L; nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv
  echo "cuda: $(nvcc --version | tail -1)"; echo "gcc: $(gcc --version | head -1)"; echo "cmake: $(cmake --version | head -1)"
  echo "suitesparse: $(grep -E '^#define SUITESPARSE_(MAIN|SUB|SUBSUB)_VERSION' $HOME/cuadmm-env/include/suitesparse/SuiteSparse_config.h | awk '{print $3}' | tr '\n' '.')"
  echo "cholmod: $(grep -m3 'define CHOLMOD_MAIN_VERSION\|define CHOLMOD_SUB_VERSION\|define CHOLMOD_SUBSUB_VERSION' $HOME/cuadmm-env/include/suitesparse/cholmod.h | awk '{print $3}' | tr '\n' '.')"
  echo "openblas: $(ls $HOME/cuadmm-env/lib | grep -m1 openblasp)"
  echo "threads: OMP_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
  echo "cuda arch: sm_80 (-gencode arch=compute_80,code=sm_80 for the references; 80-real for cuADMM-reorg)"
  echo "build type: RelWithDebInfo"
  for d in ref-9bb4ba5 ref-2d8a755 ref-2d8a755-sgs; do echo "reference $d: $(git -C $HOME/cuadmm-worktrees/$d rev-parse HEAD) + $(git -C $HOME/cuadmm-worktrees/$d diff --stat | tail -1)"; done
  SRC_OF_BUILD=$(grep -m1 '^CMAKE_HOME_DIRECTORY' "$BLD/CMakeCache.txt" | cut -d= -f2)
  echo "phase0 build: $BLD from $SRC_OF_BUILD ($(git -C "$SRC_OF_BUILD" rev-parse HEAD), $(git -C "$SRC_OF_BUILD" status --porcelain -- . | wc -l) local changes)"
  echo "scripts: $HERE ($(git -C $HERE rev-parse HEAD))"
} > "$OUT/config.txt" 2>&1
cat "$OUT/config.txt"

echo "=== 2. problems ($(date))"
mkdir -p "$CUADMM_DATA/generated" "$CUADMM_DATA/plato"
"$BLD/cuadmm_gen_problems" "$CUADMM_DATA/generated" > "$OUT/gen_problems.log" 2>&1 || { cat "$OUT/gen_problems.log"; exit 1; }
tail -3 "$OUT/gen_problems.log"
for p in trto1 mater-1; do
  [ -e "$CUADMM_DATA/plato/$p" ] || ln -s "$CUADMM_DATA/plato_kocvara/txt/$p" "$CUADMM_DATA/plato/$p"
done
(cd "$CUADMM_DATA" && find generated -type f | sort | xargs sha256sum) > "$OUT/generated_sha256.txt"

R9=legacy@9bb4ba5:$CUADMM_REF_BUILDS/ref-9bb4ba5/cuadmm_exe
R2=legacy@2d8a755:$CUADMM_REF_BUILDS/ref-2d8a755/cuadmm_exe
RS=legacy@2d8a755-sgs:$CUADMM_REF_BUILDS/ref-2d8a755-sgs/cuadmm_exe
P0=legacy@c23e67e:$BLD/cuadmm_exe

# tolerance 1e-300: the reference CLI rejects --tol 0; 1e-300 is never reached, so the runs stop at --max-iter and the
# hidden tau reduction of the sGS path (errRd < tol) never fires (plan Phase 4 criterion)
echo "=== 3. self-checks ($(date))"
rc=0
$CMP selfcheck --build $R9 --problem "$CUADMM_DATA/generated/PSD_Free" --out "$OUT/selfcheck/9bb4ba5_fixed" --policy fixed --sigma0 100 --max-iter 1500 --tol 1e-300 || rc=1
$CMP selfcheck --build $R9 --problem "$CUADMM_DATA/generated/PSD_Free" --out "$OUT/selfcheck/9bb4ba5_legacy" --policy legacy_adaptive --sigma0 100 --max-iter 6000 --tol 1e-300 || rc=1
$CMP selfcheck --build $R2 --problem "$CUADMM_DATA/generated/PSD_Free" --out "$OUT/selfcheck/2d8a755_fixed" --policy fixed --sigma0 100 --max-iter 1500 --tol 1e-300 || rc=1
$CMP selfcheck --build $R2 --problem "$CUADMM_DATA/generated/PSD_Free" --out "$OUT/selfcheck/2d8a755_balancing" --policy residual_balancing --sigma0 100 --max-iter 6000 --tol 1e-300 || rc=1
$CMP selfcheck --build $RS --problem "$CUADMM_DATA/generated/PSD_Free" --out "$OUT/selfcheck/2d8a755-sgs_sgs" --policy fixed --sigma0 100 --max-iter 3000 --tol 1e-300 --scheme sgs_admm || rc=1
$CMP selfcheck --build $R2 --problem "$CUADMM_DATA/plato/trto1" --out "$OUT/selfcheck/2d8a755_trto1" --policy fixed --sigma0 1 --max-iter 3000 --tol 1e-300 || rc=1
echo "self-checks exit: $rc"

echo "=== 4. baselines ($(date))"
# A = the reference (the stored baseline is <case>/a), B = the Phase 0 copy for fixed sigma, else the reference again
$CMP batch --spec "$HERE/tools/baselines/cases_fixed.json" --problem-root "$CUADMM_DATA" --build-a $R2 --build-b $P0 --out "$OUT/fixed" || rc=1
$CMP batch --spec "$HERE/tools/baselines/cases_legacy_adaptive.json" --problem-root "$CUADMM_DATA" --build-a $R9 --build-b $R9 --out "$OUT/legacy_adaptive" || rc=1
$CMP batch --spec "$HERE/tools/baselines/cases_residual_balancing.json" --problem-root "$CUADMM_DATA" --build-a $R2 --build-b $R2 --out "$OUT/residual_balancing" || rc=1
$CMP batch --spec "$HERE/tools/baselines/cases_sgs.json" --problem-root "$CUADMM_DATA" --build-a $RS --build-b $RS --out "$OUT/sgs" || rc=1
echo "=== done ($(date)) exit $rc"
exit $rc
