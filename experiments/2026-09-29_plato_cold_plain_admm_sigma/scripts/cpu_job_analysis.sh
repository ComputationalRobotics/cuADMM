#!/bin/bash
# CPU-only follow-up of a stage's GPU job (--dependency=afterany:<gpu job>): independent revalidation of every saved
# iterate, the stage analysis and the budget ledger.  usage: sbatch scripts/cpu_job_analysis.sh <stage>
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /n/home00/yukuanwei/cuadmm-tools/slurm_logs/%x-%j.out
set -u
EXP=/n/home00/yukuanwei/cuadmm-worktrees/plato-cold/experiments/2026-09-29_plato_cold_plain_admm_sigma
PY=$HOME/cuadmm-analysis-env/bin/python
ST=$1
cd $EXP
echo "start $(date -Iseconds) host $(hostname) job $SLURM_JOB_ID stage $ST"
$PY scripts/postvalidate.py $ST --jobs 4 > logs/${ST}_postvalidate.log 2>&1
echo "postvalidate exit $?: $(tail -1 logs/${ST}_postvalidate.log)"
$PY scripts/analyze.py $ST > logs/${ST}_analysis.log 2>&1
echo "analyze exit $?"
cat logs/${ST}_analysis.log
$PY scripts/budget.py
echo "end $(date -Iseconds)"
