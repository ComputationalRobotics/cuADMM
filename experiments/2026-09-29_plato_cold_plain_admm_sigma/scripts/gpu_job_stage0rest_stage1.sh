#!/bin/bash
# Self-contained H200 batch job (it ends when the work ends, so no idle allocation and no dependence on an attached
# session): device test and environment capture, the missing sanitizer cases (scripts/sanitizers2.sh), then Stage 1
# (plans/stage1.json, 52 smoke runs of 120 s).
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH --exclude=holygpu8a12204
#SBATCH -J cuadmm-plato-s1
#SBATCH -o /n/home00/yukuanwei/cuadmm-tools/slurm_logs/%x-%j.out
set -u
EXP=/n/home00/yukuanwei/cuadmm-worktrees/plato-cold/experiments/2026-09-29_plato_cold_plain_admm_sigma
PY=$HOME/cuadmm-analysis-env/bin/python
source $HOME/cuadmm-tools/env.sh
echo "start $(date -Iseconds) host $(hostname) job $SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$HOME/cuadmm-tools/ctx > $EXP/logs/stage0/device_test_$SLURM_JOB_ID.txt 2>&1
cat $EXP/logs/stage0/device_test_$SLURM_JOB_ID.txt
grep -q "malloc: no error" $EXP/logs/stage0/device_test_$SLURM_JOB_ID.txt || { echo "device test failed: stopping"; exit 2; }
ALLOC_CMD="sbatch scripts/gpu_job_stage0rest_stage1.sh (job $SLURM_JOB_ID)" bash $EXP/scripts/capture_env.sh $EXP/env/job_$SLURM_JOB_ID
echo "== sanitizers (missing cases) $(date -Iseconds)"
bash $EXP/scripts/sanitizers2.sh > $EXP/logs/stage0/sanitizers_runner_part3.log 2>&1
cat $EXP/logs/stage0/sanitizers_runner_part3.log
echo "== stage 1 $(date -Iseconds)"
cd $EXP && $PY scripts/campaign.py plans/stage1.json > $EXP/logs/stage1_driver.log 2>&1
echo "stage 1 driver exit $? $(date -Iseconds)"
tail -3 $EXP/logs/stage1_driver.log
echo "end $(date -Iseconds)"
