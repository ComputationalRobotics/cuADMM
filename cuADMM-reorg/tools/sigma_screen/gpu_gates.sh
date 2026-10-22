#!/bin/bash
# gpu_gates.sh: writes the sbatch files of the Phase-4b GPU gates (plan Phase 4b: G0-G3) and prints the projected
# H200-hours. NOTHING IS SUBMITTED by this script: the jobs run only with the user's approval (given on 2026-10-22
# for at most 12.58 H200-hours); submit with `sbatch <file>`, G0 first.
#
# usage: gpu_gates.sh <cuadmm_exe (sm_90 build)> "<flags of the chosen balance_freeze setting>" [gates dir]
#   e.g. gpu_gates.sh ~/cuadmm-builds/reorg-f10e149-sm90/cuadmm_exe "--factor 1.2 --step-schedule halve_on_reversal --freeze-bias 4"
#
# Protocol of every run (the PLATO cold-start campaign's): plain ADMM, tau 1.618, tolerance 1e-4, external validation
# every 50 iterations at 1e-4, strict stage max|DIMACS| <= 1e-6, time cap on the iteration time, 17-digit history,
# summary, sigma log, validation history. Every policy parameter is given explicitly, so that a change of a default
# cannot alter a comparison.
#   G0  fixed-sigma reproduction on the H200: trto1, buck1 (600 s caps) and pendulum N=80 (900 s) at fixed sigma = 1
#       against the campaign's H200 runs (9bb4ba5: trto1 strict at 60,150 iterations / 23.3 s, buck1 119,850 / 68.9 s)
#       and the pendulum reference (fixed sigma = 1 plain ADMM: practical 1e-4 at 230,315 iterations, 625 s).
#   G1  trto1, buck1 (600 s), vibra1, mater-1, mater-2, shmup1 (1800 s) x {balance_freeze candidate, residual_balancing}
#       at sigma_0 = 1; the fixed-sigma and old-rule comparators are the campaign's H200 runs (reused, G0 validates
#       the reproduction; trto1/buck1 fixed are re-run by G0).
#   G2  pendulum N=80 at sigma_0 in {1, 100} x {candidate, residual_balancing} (900 s) against 118-124 s (old rule)
#       and 625 s (fixed sigma = 1) of the pendulum experiments.
#   G3  mater-3 to mater-6 x {candidate, residual_balancing} (1800 s, 128 GB); comparators: the campaign's runs.
set -u
EXE=${1:?cuadmm_exe}; SETTING=${2:?"balance_freeze flags"}
GATES=${3:-${SCREEN_ROOT:-$HOME/cuadmm-experiments/2026-10-21_phase4b_screen}/gpu_gates}
DATA=${CUADMM_DATA:-$HOME/cuadmm-data}/plato_kocvara/txt
PENDULUM=${PENDULUM:-$HOME/cuADMM/examples/pendulum/TXT/N=80_licols}
PARTITION=${GATES_PARTITION:-gpu_h200}
GRES=${GATES_GRES:-gpu:nvidia_h200:1}
EXCLUDE=${GATES_EXCLUDE:-holygpu8a12204}   # a GPU of this node refused the CUDA context (G1, 2026-10-22); the campaign excluded it too
VTHREADS=${GATES_VALIDATION_THREADS:-8}
COMMON="--scheme admm --tau 1.618 --tol 1e-4 --validation-interval 50 --validation-tol 1e-4 --strict-dimacs-tol 1e-6 --validation-threads $VTHREADS --max-iter 100000000 --psd-medium-streams 15 --psd-lowrank-enabled 1"
mkdir -p "$GATES"
total_s=0

run() {  # run <gate> <name> <problem dir> <policy flags> <cap s>
  local gate=$1 name=$2 prob=$3 flags=$4 cap=$5
  local out=$GATES/runs/$gate/$name
  cat >> "$GATES/$gate.sbatch" <<EOR
mkdir -p $out
echo "=== $name start \$(date +%FT%T) cap $cap s: $prob"
echo "$EXE $prob $flags $COMMON --time-limit $cap" > $out/command.txt
$EXE $prob $flags $COMMON --time-limit $cap --history $out/history.csv --summary $out/summary.json --sigma-log $out/sigma_log.csv --validation-history $out/validation.csv --tag $gate/$name > $out/stdout.log 2>&1
echo "=== $name exit \$? \$(date +%FT%T)"
EOR
  total_s=$((total_s + cap))
}

header() {  # header <gate> <time limit> <mem>
  local gate=$1 tl=$2 mem=$3
  cat > "$GATES/$gate.sbatch" <<EOH
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH --gres=$GRES
#SBATCH -c 16
#SBATCH --mem=$mem
#SBATCH -t $tl
#SBATCH -J p4b-$gate
#SBATCH --exclude=$EXCLUDE
#SBATCH -o $GATES/logs/$gate-%j.out
# Phase-4b GPU gate $gate (plan Phase 4b); written by tools/sigma_screen/gpu_gates.sh; approved 2026-10-22.
set -u
mkdir -p $GATES/logs
source \$HOME/cuadmm-tools/env.sh
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
echo "=== gate $gate job \${SLURM_JOB_ID:-?} on \$(hostname) \$(date +%FT%T)"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
$EXE --version
sha256sum $EXE
EOH
}

FIXED="--policy fixed --sigma0 1"
RB="--policy residual_balancing --factor 2 --win-ratio 1.35 --guard 1 --residual-weight 1 --sigma-min 1e-3 --sigma-max 1e3"
BF="--policy balance_freeze $SETTING --freeze-value band_centre --freeze-iteration 10000 --pingpong-window 8 --pingpong-net 2 --pingpong-min-iter 1000 --win-ratio 1.35 --guard 1 --residual-weight 1 --sigma-min 1e-3 --sigma-max 1e3 --schedule 200:10,1000:25,5000:50,10000:200,inf:1000"

header G0 00:45:00 96G
run G0 trto1_fixed_s1 $DATA/trto1 "$FIXED" 600
run G0 buck1_fixed_s1 $DATA/buck1 "$FIXED" 600
run G0 pendulum_N80_fixed_s1 $PENDULUM "$FIXED" 900
g0=$total_s; total_s=0

header G1 05:00:00 96G
for p in trto1 buck1; do run G1 ${p}_bf_s1 $DATA/$p "$BF --sigma0 1" 600; done
for p in vibra1 mater-1 mater-2 shmup1; do run G1 ${p}_bf_s1 $DATA/$p "$BF --sigma0 1" 1800; done
for p in trto1 buck1; do run G1 ${p}_rb_s1 $DATA/$p "$RB --sigma0 1" 600; done
for p in vibra1 mater-1 mater-2 shmup1; do run G1 ${p}_rb_s1 $DATA/$p "$RB --sigma0 1" 1800; done
g1=$total_s; total_s=0

header G2 01:10:00 96G
for s in 1 100; do run G2 pendulum_N80_bf_s$s $PENDULUM "$BF --sigma0 $s" 900; done
for s in 1 100; do run G2 pendulum_N80_rb_s$s $PENDULUM "$RB --sigma0 $s" 900; done
g2=$total_s; total_s=0

header G3 04:20:00 128G
for p in mater-3 mater-4 mater-5 mater-6; do run G3 ${p}_bf_s1 $DATA/$p "$BF --sigma0 1" 1800; done
for p in mater-3 mater-4 mater-5 mater-6; do run G3 ${p}_rb_s1 $DATA/$p "$RB --sigma0 1" 1800; done
g3=$total_s; total_s=0

all=$((g0 + g1 + g2 + g3))
printf "written to %s (not submitted)\n" "$GATES"
printf "%-4s %-62s %8s\n" gate runs "cap h"
printf "%-4s %-62s %8.2f\n" G0 "trto1, buck1, pendulum N=80 at fixed sigma = 1" $(echo "$g0 / 3600" | bc -l)
printf "%-4s %-62s %8.2f\n" G1 "6 problems x {balance_freeze candidate, residual_balancing}" $(echo "$g1 / 3600" | bc -l)
printf "%-4s %-62s %8.2f\n" G2 "pendulum N=80, sigma_0 in {1, 100} x {candidate, residual_balancing}" $(echo "$g2 / 3600" | bc -l)
printf "%-4s %-62s %8.2f\n" G3 "mater-3..6 x {candidate, residual_balancing}" $(echo "$g3 / 3600" | bc -l)
printf "%-4s %-62s %8.2f  (job limits 0:45 + 5:00 + 1:10 + 4:20 = 11.25 h; approved: at most 12.58 H200-hours)\n" all "sum of the caps" $(echo "$all / 3600" | bc -l)
