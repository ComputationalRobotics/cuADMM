#!/bin/bash
# Diagnosis of the initcheck reports (PROTOCOL.md, Deviation 4). Self-contained H200 batch job; ends when done.
#  1. syevd_workspace_test: cusolverDnXsyevd as cuADMM calls it, n = 673 (buck4) and 1761 (trto5), device workspace,
#     W and info unwritten / zero / NaN / garbage, 3 repetitions each: are the eigenvalues and eigenvectors
#     bit-identical for every fill?
#  2. the same program under initcheck (n = 673, unwritten vs zero-filled workspace): do the reports concern the
#     caller-provided workspace/outputs?
#  3. complete, unfiltered initcheck tallies of the cuADMM cases (--print-limit 0, streamed through
#     initcheck_tally.awk, nothing excluded): buck4 exactly as the planned case (200 iterations, adaptive, validation
#     every 100) and trto5 (fixed sigma, 3 iterations with validation every iteration: the planned 250 iterations take
#     hours under initcheck). Every report is attributed to its kernel and its named host frames.
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH --exclude=holygpu8a12204
#SBATCH -J cuadmm-plato-diag
#SBATCH -o /n/home00/yukuanwei/cuadmm-tools/slurm_logs/%x-%j.out
set -u
EXP=/n/home00/yukuanwei/cuadmm-worktrees/plato-cold/experiments/2026-09-29_plato_cold_plain_admm_sigma
D=$EXP/scripts/diag
OUT=$EXP/logs/diag; mkdir -p $OUT
BIN=$HOME/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma/diag; mkdir -p $BIN
B=$HOME/cuadmm-builds/plato_official
TXT=$HOME/cuadmm-data/plato_kocvara/txt
source $HOME/cuadmm-tools/env.sh
CS=$(dirname $(which nvcc))/compute-sanitizer
echo "start $(date -Iseconds) host $(hostname) job $SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$HOME/cuadmm-tools/ctx | tee $OUT/device_test_$SLURM_JOB_ID.txt
grep -q "malloc: no error" $OUT/device_test_$SLURM_JOB_ID.txt || { echo "device test failed"; exit 2; }
nvcc -O2 -arch=sm_90 -o $BIN/syevd_workspace_test $D/syevd_workspace_test.cu -lcusolver -lcublas > $OUT/syevd_workspace_test_build.log 2>&1 || { cat $OUT/syevd_workspace_test_build.log; exit 3; }
echo "== 1. workspace fills $(date -Iseconds)"
for n in 673 1761; do
  for m in none zero nan garbage; do $BIN/syevd_workspace_test $n $m 3; done
done > $OUT/syevd_workspace_fills.txt 2>&1
grep RESULT $OUT/syevd_workspace_fills.txt
for n in 673 1761; do
  echo "n=$n distinct (hashW,hashV) over the 4 fills: $(grep "RESULT n $n " $OUT/syevd_workspace_fills.txt | awk '{print $7, $9}' | sort -u | wc -l)"
done | tee $OUT/syevd_workspace_fills_verdict.txt
echo "== 2. the test program under initcheck $(date -Iseconds)"
for m in none zero; do
  timeout 1800 $CS --tool initcheck --print-limit 0 $BIN/syevd_workspace_test 673 $m 2 2>&1 | awk -f $D/initcheck_tally.awk > $OUT/initcheck_tally__syevd_test_673_$m.txt
  echo "initcheck syevd_workspace_test 673 $m: pipe status ${PIPESTATUS[*]}"; cut -c1-240 $OUT/initcheck_tally__syevd_test_673_$m.txt
done
echo "== 3. complete unfiltered tallies of the cuADMM cases $(date -Iseconds)"
cd $B
t0=$(date +%s)
timeout 4500 $CS --tool initcheck --print-limit 0 ./cuadmm_exe $TXT/buck4 --algorithm admm --sigma-policy legacy_adaptive --sig 1 --tol 1e-4 --max-iter 200 --validate-interval 100 --validate-threads 4 2>&1 \
  | awk -f $D/initcheck_tally.awk > $OUT/initcheck_tally__buck4_adaptive_200it.txt
echo "buck4 (200 it): pipe status ${PIPESTATUS[*]}, $(( $(date +%s) - t0 )) s"; cut -c1-300 $OUT/initcheck_tally__buck4_adaptive_200it.txt
t0=$(date +%s)
timeout 4500 $CS --tool initcheck --print-limit 0 ./cuadmm_exe $TXT/trto5 --algorithm admm --sigma-policy fixed --sig 1 --tol 1e-4 --max-iter 3 --validate-interval 1 --validate-threads 4 2>&1 \
  | awk -f $D/initcheck_tally.awk > $OUT/initcheck_tally__trto5_fixed_3it.txt
echo "trto5 (3 it): pipe status ${PIPESTATUS[*]}, $(( $(date +%s) - t0 )) s"; cut -c1-300 $OUT/initcheck_tally__trto5_fixed_3it.txt
echo "end $(date -Iseconds)"
