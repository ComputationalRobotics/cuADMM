#!/bin/bash
# CPU-only follow-up of the two Stage-3 GPU jobs (--dependency=afterany:<part1>:<part2>): independent revalidation of
# every returned and final iterate, the Stage-3 analysis, plots, report tables and workbook, and the budget ledger.
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J cuadmm-plato-s3-cpu
#SBATCH -o /n/home00/yukuanwei/cuadmm-tools/slurm_logs/%x-%j.out
set -u
EXP=/n/home00/yukuanwei/cuadmm-worktrees/plato-cold/experiments/2026-09-29_plato_cold_plain_admm_sigma
PY=$HOME/cuadmm-analysis-env/bin/python
cd $EXP
echo "start $(date -Iseconds) host $(hostname) job $SLURM_JOB_ID"
$PY scripts/postvalidate.py stage3 --jobs 4 > logs/stage3_postvalidate.log 2>&1
echo "postvalidate exit $?: $(tail -1 logs/stage3_postvalidate.log)"
$PY scripts/analyze.py stage3 > logs/stage3_analysis.log 2>&1; echo "analyze exit $?"
$PY scripts/plots.py stage3 > logs/stage3_plots.log 2>&1; echo "plots exit $?"
$PY scripts/make_report.py > logs/stage3_report.log 2>&1; echo "make_report exit $?"; tail -3 logs/stage3_report.log
$PY scripts/budget.py
echo "end $(date -Iseconds)"
