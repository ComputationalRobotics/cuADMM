#!/bin/bash
# CPU-only follow-up of the Stage-1 GPU job (submitted with --dependency=afterany:<gpu job>): the independent
# revalidation of every saved iterate, the Stage-1 analysis (validation intervals) and the budget ledger.
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -J cuadmm-plato-s1-cpu
#SBATCH -o /n/home00/yukuanwei/cuadmm-tools/slurm_logs/%x-%j.out
set -u
EXP=/n/home00/yukuanwei/cuadmm-worktrees/plato-cold/experiments/2026-09-29_plato_cold_plain_admm_sigma
PY=$HOME/cuadmm-analysis-env/bin/python
cd $EXP
echo "start $(date -Iseconds) host $(hostname) job $SLURM_JOB_ID"
$PY scripts/postvalidate.py stage1 --jobs 4 > logs/stage1_postvalidate.log 2>&1
echo "postvalidate exit $?: $(tail -1 logs/stage1_postvalidate.log)"
$PY scripts/analyze.py stage1 > logs/stage1_analysis.log 2>&1
echo "analyze exit $?"
cat logs/stage1_analysis.log
$PY scripts/budget.py
echo "end $(date -Iseconds)"
