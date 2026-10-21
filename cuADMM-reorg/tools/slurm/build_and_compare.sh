#!/bin/bash
# Clean evidence for one commit: a worktree and a build directory of its own (never rebuilt while a GPU job uses it),
# the build on the CPU allocation, then tools/slurm/compare_build.sh and the tests on the GPU allocation.
# usage: build_and_compare.sh <commit> <tag> <config> <kind> [branch-or-worktree-dir]
#   CUADMM_CPU_JOB, CUADMM_GPU_JOB: Slurm job ids of the allocations; CUADMM_THREADS: thread count of the config
set -u
COMMIT=$1; TAG=$2; CONFIG=$3; KIND=$4
REPO=$HOME/cuADMM
WT=$HOME/cuadmm-worktrees/reorg-$COMMIT
BLD=$HOME/cuadmm-builds/reorg-$COMMIT
if [ ! -d "$WT" ]; then
  git -C "$REPO" worktree add --detach "$WT" "$COMMIT" || exit 2
fi
echo "=== build $COMMIT in $BLD ($(date))"
srun --jobid=$CUADMM_CPU_JOB --overlap -n 1 -c 16 bash -c "source \$HOME/cuadmm-tools/env.sh; cd $WT/cuADMM-reorg && BUILD_JOBS=16 tools/build.sh $BLD 2>&1 | tail -3" || exit 2
echo "=== compare and test on the GPU ($(date))"
OUT=$HOME/cuadmm-baselines/$CONFIG
mkdir -p "$OUT"
srun --jobid=$CUADMM_GPU_JOB --overlap -n 1 -c ${CUADMM_THREADS:-8} bash -c "export CUADMM_THREADS=${CUADMM_THREADS:-8}; cd $WT/cuADMM-reorg && tools/slurm/compare_build.sh $BLD $CONFIG $TAG $KIND > $OUT/compare_$TAG.log 2>&1; echo \"compare exit \$?\""
echo "identical: $(grep -c 'RESULT: IDENTICAL' $OUT/compare_$TAG.log), different: $(grep -c 'RESULT: DIFFERENT' $OUT/compare_$TAG.log), failed runs: $(grep -c 'FAILED (' $OUT/compare_$TAG.log)"
grep -E "tests ran|^\[  (PASSED|FAILED) \]" $OUT/compare/$TAG/tests.log | tail -6
echo "=== done ($(date))"
