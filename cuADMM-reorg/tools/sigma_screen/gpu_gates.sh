#!/bin/bash
# gpu_gates.sh: prepares the Phase-4b GPU gates of the plan (G0-G3) as sbatch scripts and prints the projected
# GPU-hours. NOTHING IS SUBMITTED: H200 jobs and GPU jobs over one hour need the user's approval (developer
# prompt); submit with `sbatch <file>` after it.
#
# usage: gpu_gates.sh <cuadmm_exe> "<flags of the chosen balance_freeze setting>" [gates dir]
#   e.g. gpu_gates.sh ~/cuadmm-builds/reorg-<sha>/cuadmm_exe "--factor 2 --step-schedule constant --freeze-iteration 10000"
#
# Gates (plan, Phase 4b), one sbatch file each, every run with the campaign protocol (tolerance 1e-4, validation
# every 50 iterations, strict stage max|DIMACS| <= 1e-6, time caps; 17-digit histories, summaries, sigma logs):
#   G0  fixed-sigma reproduction: trto1, buck1 and pendulum N=80 at fixed sigma = 1 (caps 600, 600, 900 s): the
#       iterates must match the Phase 0 references (the build is bitwise), the times are the H200 reference of this
#       build against the PLATO campaign (9bb4ba5) and the paper replication (625 s for the pendulum).
#   G1  trto1, buck1 (600 s caps), vibra1, mater-1, mater-2, shmup1 (1800 s) with balance_freeze, residual_balancing
#       and fixed sigma = 1, sigma_0 = 1.
#   G2  pendulum N=80 at sigma_0 in {1, 100} with balance_freeze and residual_balancing (900 s caps) against 124 s
#       (old rule) and 625 s (fixed sigma = 1) of the paper replication.
#   G3  mater-3 to mater-6 with balance_freeze and residual_balancing (1800 s caps, 128 GB).
# Problem data: PLATO TXT directories under $CUADMM_DATA/plato_kocvara/txt, the pendulum under $PENDULUM.
set -u
EXE=${1:?cuadmm_exe}; SETTING=${2:?"balance_freeze flags"}
GATES=${3:-${SCREEN_ROOT:-$HOME/cuadmm-experiments/2026-10-21_phase4b_screen}/gpu_gates}
DATA=${CUADMM_DATA:-$HOME/cuadmm-data}/plato_kocvara/txt
PENDULUM=${PENDULUM:-$HOME/cuADMM/examples/pendulum/TXT/N=80_licols}
PARTITION=${GATES_PARTITION:-gpu_h200}
GRES=${GATES_GRES:-gpu:nvidia_h200:1}
COMMON="--tol 1e-4 --validation-interval 50 --validation-tol 1e-4 --strict-dimacs-tol 1e-6 --max-iter 100000000"
mkdir -p "$GATES"
total_s=0

# run <gate> <name> <problem dir> <policy flags> <cap s>: one solver run inside the gate's job script
run() {
  local gate=$1 name=$2 prob=$3 flags=$4 cap=$5
  local out=$GATES/runs/$gate/$name
  cat >> "$GATES/$gate.sbatch" <<EOR
mkdir -p $out
echo "=== $name (\$(date)) cap $cap s"
$EXE $prob $flags $COMMON --time-limit $cap --history $out/history.csv --summary $out/summary.json --sigma-log $out/sigma_log.csv --validation-history $out/validation.csv --tag $gate/$name > $out/stdout.log 2>&1
echo "exit \$? (\$(date))"
EOR
  total_s=$((total_s + cap))
}

# header <gate> <time limit> <mem>
header() {
  local gate=$1 tl=$2 mem=$3
  cat > "$GATES/$gate.sbatch" <<EOH
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH --gres=$GRES
#SBATCH -c 16
#SBATCH --mem=$mem
#SBATCH -t $tl
#SBATCH -J p4b-$gate
#SBATCH -o $GATES/logs/$gate-%j.out
# Phase-4b GPU gate $gate (plan Phase 4b); prepared by tools/sigma_screen/gpu_gates.sh, submitted only after approval.
set -u
mkdir -p $GATES/logs
source \$HOME/cuadmm-tools/env.sh
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
nvidia-smi -L
$EXE --version
EOH
}

FIXED="--policy fixed --sigma0 1"
BF="--policy balance_freeze $SETTING"
RB="--policy residual_balancing"

g0=0
header G0 00:45:00 96G
run G0 trto1_fixed_s1 $DATA/trto1 "$FIXED" 600
run G0 buck1_fixed_s1 $DATA/buck1 "$FIXED" 600
run G0 pendulum_N80_fixed_s1 $PENDULUM "$FIXED" 900
g0=$total_s; total_s=0

header G1 07:30:00 96G
for p in trto1 buck1; do
  run G1 ${p}_bf_s1 $DATA/$p "$BF --sigma0 1" 600
  run G1 ${p}_rb_s1 $DATA/$p "$RB --sigma0 1" 600
  run G1 ${p}_fixed_s1 $DATA/$p "$FIXED" 600
done
for p in vibra1 mater-1 mater-2 shmup1; do
  run G1 ${p}_bf_s1 $DATA/$p "$BF --sigma0 1" 1800
  run G1 ${p}_rb_s1 $DATA/$p "$RB --sigma0 1" 1800
  run G1 ${p}_fixed_s1 $DATA/$p "$FIXED" 1800
done
g1=$total_s; total_s=0

header G2 01:15:00 96G
for s in 1 100; do
  run G2 pendulum_N80_bf_s$s $PENDULUM "$BF --sigma0 $s" 900
  run G2 pendulum_N80_rb_s$s $PENDULUM "$RB --sigma0 $s" 900
done
g2=$total_s; total_s=0

header G3 04:15:00 128G
for p in mater-3 mater-4 mater-5 mater-6; do
  run G3 ${p}_bf_s1 $DATA/$p "$BF --sigma0 1" 1800
  run G3 ${p}_rb_s1 $DATA/$p "$RB --sigma0 1" 1800
done
g3=$total_s; total_s=0

all=$((g0 + g1 + g2 + g3))
printf "prepared in %s (not submitted)\n" "$GATES"
printf "%-4s %-60s %8s\n" gate runs "cap h"
printf "%-4s %-60s %8.2f\n" G0 "trto1, buck1, pendulum N=80 at fixed sigma = 1" $(echo "$g0 / 3600" | bc -l)
printf "%-4s %-60s %8.2f\n" G1 "6 problems x {balance_freeze, residual_balancing, fixed 1}" $(echo "$g1 / 3600" | bc -l)
printf "%-4s %-60s %8.2f\n" G2 "pendulum N=80, sigma_0 in {1, 100} x {balance_freeze, rb}" $(echo "$g2 / 3600" | bc -l)
printf "%-4s %-60s %8.2f\n" G3 "mater-3..6 x {balance_freeze, residual_balancing}" $(echo "$g3 / 3600" | bc -l)
printf "%-4s %-60s %8.2f  (plan: at most 12.6 H200-hours; runs that converge stop before their cap)\n" all "sum of the caps" $(echo "$all / 3600" | bc -l)
