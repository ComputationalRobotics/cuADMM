#!/bin/bash
# Self-contained H200 batch job for one plan (ends when the plan ends; resumable: finished runs are skipped).
# usage: sbatch -t <limit> -J <name> scripts/gpu_job_plan.sh plans/<stage>.json
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH --exclude=holygpu8a12204
#SBATCH -o /n/home00/yukuanwei/cuadmm-tools/slurm_logs/%x-%j.out
set -u
EXP=/n/home00/yukuanwei/cuadmm-worktrees/plato-cold/experiments/2026-09-29_plato_cold_plain_admm_sigma
PY=$HOME/cuadmm-analysis-env/bin/python
PLAN=$1
source $HOME/cuadmm-tools/env.sh
echo "start $(date -Iseconds) host $(hostname) job $SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES plan $PLAN"
$HOME/cuadmm-tools/ctx > $EXP/logs/stage0/device_test_$SLURM_JOB_ID.txt 2>&1
cat $EXP/logs/stage0/device_test_$SLURM_JOB_ID.txt
grep -q "malloc: no error" $EXP/logs/stage0/device_test_$SLURM_JOB_ID.txt || { echo "device test failed: stopping"; exit 2; }
ALLOC_CMD="sbatch scripts/gpu_job_plan.sh $PLAN (job $SLURM_JOB_ID)" bash $EXP/scripts/capture_env.sh $EXP/env/job_$SLURM_JOB_ID
base=$(basename $PLAN .json)
cd $EXP && $PY scripts/campaign.py $PLAN >> $EXP/logs/${base}_driver.log 2>&1
echo "driver exit $? $(date -Iseconds)"
tail -3 $EXP/logs/${base}_driver.log
echo "end $(date -Iseconds)"
