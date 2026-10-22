#!/bin/bash
# Second pinned configuration (the full A100-SXM4-80GB on partition gpu, 16 threads): the Phase 0 reference baselines
# and the comparison of one cuADMM-reorg commit, in one allocation of under one hour.
# usage: full_a100_session.sh <gpu job id> <commit> <tag> [kind]
set -u
JOB=$1; COMMIT=$2; TAG=$3; KIND=${4:-reorg}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=a100-sxm4-80gb_cuda12.9_gcc12.2_ss7.10.1_t16
WT=$HOME/cuadmm-worktrees/reorg-$COMMIT
BLD=$HOME/cuadmm-builds/reorg-$COMMIT
mkdir -p $HOME/cuadmm-baselines/$CONFIG
echo "=== references and Phase 0 copy on the full A100 ($(date))"
srun --jobid=$JOB --overlap -n 1 -c 16 bash -c "export CUADMM_THREADS=16; cd $HERE && tools/slurm/phase0_baselines.sh \$HOME/cuadmm-builds/reorg-phase0 $CONFIG > \$HOME/cuadmm-baselines/$CONFIG/phase0_baselines.log 2>&1; echo \"baselines exit \$?\""
L=$HOME/cuadmm-baselines/$CONFIG/phase0_baselines.log
echo "baselines: identical $(grep -c 'RESULT: IDENTICAL' $L), different $(grep -c 'RESULT: DIFFERENT' $L), failed runs $(grep -c 'FAILED (' $L)"
echo "=== commit $COMMIT against the references on the full A100 ($(date))"
srun --jobid=$JOB --overlap -n 1 -c 16 bash -c "export CUADMM_THREADS=16; cd $WT/cuADMM-reorg && tools/slurm/compare_build.sh $BLD $CONFIG $TAG $KIND > \$HOME/cuadmm-baselines/$CONFIG/compare_$TAG.log 2>&1; echo \"compare exit \$?\""
L=$HOME/cuadmm-baselines/$CONFIG/compare_$TAG.log
echo "compare: identical $(grep -c 'RESULT: IDENTICAL' $L), different $(grep -c 'RESULT: DIFFERENT' $L), failed runs $(grep -c 'FAILED (' $L)"
grep -E "tests ran|^\[  (PASSED|FAILED) \]" $HOME/cuadmm-baselines/$CONFIG/compare/$TAG/tests.log | tail -4
echo "=== done ($(date))"
