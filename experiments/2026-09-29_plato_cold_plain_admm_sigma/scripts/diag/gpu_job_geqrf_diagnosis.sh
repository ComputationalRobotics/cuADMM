#!/bin/bash
# Diagnosis of the initcheck reports in cuSOLVER's geqr2_smem_domino_fast under cusolverDnDgeqrf, called from
# psd_projection's lobpcg (PROTOCOL.md, Deviation 6): lobpcg's QR step with the caller's tau/workspace/devInfo unwritten,
# zeroed, NaN-filled or garbage-filled (shapes of shmup4: n = 1681 and 1680, m = 21, and m = 20; the first LOBPCG
# iteration, where Delta_X_k = X_k, and a generic full-rank block), then the same under initcheck (complete tally).
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH --exclude=holygpu8a12204
#SBATCH -J cuadmm-plato-diag2
#SBATCH -o /n/home00/yukuanwei/cuadmm-tools/slurm_logs/%x-%j.out
set -u
EXP=/n/home00/yukuanwei/cuadmm-worktrees/plato-cold/experiments/2026-09-29_plato_cold_plain_admm_sigma
D=$EXP/scripts/diag; OUT=$EXP/logs/diag
BIN=$HOME/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma/diag; mkdir -p $BIN $OUT
source $HOME/cuadmm-tools/env.sh
CS=$(dirname $(which nvcc))/compute-sanitizer
echo "start $(date -Iseconds) host $(hostname) job $SLURM_JOB_ID"
$HOME/cuadmm-tools/ctx | tee $OUT/device_test_$SLURM_JOB_ID.txt
grep -q "malloc: no error" $OUT/device_test_$SLURM_JOB_ID.txt || { echo "device test failed"; exit 2; }
nvcc -O2 -arch=sm_90 -o $BIN/geqrf_workspace_test $D/geqrf_workspace_test.cu -lcusolver -lcublas > $OUT/geqrf_workspace_test_build.log 2>&1 || { cat $OUT/geqrf_workspace_test_build.log; exit 3; }
for nm in "1681 21" "1680 20"; do
  for kind in first generic; do
    for mode in none zero nan garbage; do $BIN/geqrf_workspace_test $nm $mode $kind 3; done
  done
done > $OUT/geqrf_workspace_fills.txt 2>&1
grep RESULT $OUT/geqrf_workspace_fills.txt
for nm in "1681 21" "1680 20"; do for kind in first generic; do
  echo "n,m=$nm $kind: distinct (hashR,hashQ) over the 4 fills: $(grep "RESULT n ${nm% *} m ${nm#* } kind $kind " $OUT/geqrf_workspace_fills.txt | awk '{print $11, $13}' | sort -u | wc -l); all stable: $(grep "RESULT n ${nm% *} m ${nm#* } kind $kind " $OUT/geqrf_workspace_fills.txt | awk '{print $15}' | sort -u | tr '\n' ' ')"
done; done | tee $OUT/geqrf_workspace_fills_verdict.txt
for mode in none zero; do
  timeout 1200 $CS --tool initcheck --print-limit 0 $BIN/geqrf_workspace_test 1681 21 $mode first 2 2>&1 | awk -f $D/initcheck_tally.awk > $OUT/initcheck_tally__geqrf_test_1681_21_$mode.txt
  echo "initcheck geqrf_workspace_test 1681 21 $mode first: pipe status ${PIPESTATUS[*]}"; cut -c1-240 $OUT/initcheck_tally__geqrf_test_1681_21_$mode.txt
done
echo "end $(date -Iseconds)"
