#!/bin/bash
# Sanitizer coverage of the LOBPCG path (psd_projection's lobpcg), used in Stage 1 by shmup4 and shmup5 but by none of the
# earlier sanitizer cases (LOBPCG calls = 0 there). Case: shmup4, fixed sigma 1, 120 iterations (LOBPCG from the first
# rank analysis, the full-EVD re-evaluation at iteration 100), validation every 60. memcheck (leak check), synccheck,
# and the complete unfiltered initcheck tally (--print-limit 0, every report attributed to its kernel and host frames).
# Self-contained H200 batch job; ends when done.
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 02:30:00
#SBATCH --exclude=holygpu8a12204
#SBATCH -J cuadmm-plato-sanlob
#SBATCH -o /n/home00/yukuanwei/cuadmm-tools/slurm_logs/%x-%j.out
set -u
EXP=/n/home00/yukuanwei/cuadmm-worktrees/plato-cold/experiments/2026-09-29_plato_cold_plain_admm_sigma
OUT=$EXP/logs/sanitizers; D=$EXP/scripts/diag
B=$HOME/cuadmm-builds/plato_official; TXT=$HOME/cuadmm-data/plato_kocvara/txt
source $HOME/cuadmm-tools/env.sh
CS=$(dirname $(which nvcc))/compute-sanitizer
echo "start $(date -Iseconds) host $(hostname) job $SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$HOME/cuadmm-tools/ctx | tee $EXP/logs/stage0/device_test_$SLURM_JOB_ID.txt
grep -q "malloc: no error" $EXP/logs/stage0/device_test_$SLURM_JOB_ID.txt || { echo "device test failed"; exit 2; }
cd $B
CASE="./cuadmm_exe $TXT/shmup4 --algorithm admm --sigma-policy fixed --sig 1 --tol 1e-4 --max-iter 120 --validate-interval 60 --validate-threads 4"
for tool in memcheck synccheck; do
  extra=""; [ $tool = memcheck ] && extra="--leak-check full"
  t0=$(date +%s)
  timeout 2700 $CS --tool $tool $extra --error-exitcode 99 $CASE > $OUT/${tool}__shmup4_fixed_lobpcg.log 2>&1
  rc=$?
  echo "$tool shmup4_fixed_lobpcg: exit $rc, $(( $(date +%s) - t0 )) s | $(grep -E 'ERROR SUMMARY|LEAK SUMMARY' $OUT/${tool}__shmup4_fixed_lobpcg.log | tr '\n' ' ') | $(grep -E 'LOBPCG calls|Solver ended' $OUT/${tool}__shmup4_fixed_lobpcg.log | tr '\n' ' ')"
done
t0=$(date +%s)
timeout 3600 $CS --tool initcheck --print-limit 0 $CASE 2>&1 | tee >(grep -E "LOBPCG calls|Solver ended" > $OUT/initcheck__shmup4_fixed_lobpcg.solver_lines.txt) | awk -f $D/initcheck_tally.awk > $EXP/logs/diag/initcheck_tally__shmup4_fixed_lobpcg_120it.txt
echo "initcheck (complete tally) shmup4_fixed_lobpcg: pipe status ${PIPESTATUS[*]}, $(( $(date +%s) - t0 )) s"
sleep 2; cat $OUT/initcheck__shmup4_fixed_lobpcg.solver_lines.txt; cut -c1-300 $EXP/logs/diag/initcheck_tally__shmup4_fixed_lobpcg_120it.txt
echo "end $(date -Iseconds)"
