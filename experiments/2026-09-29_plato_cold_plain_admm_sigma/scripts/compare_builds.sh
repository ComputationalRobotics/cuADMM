#!/bin/bash
# Bit identity of the plain-ADMM-only build (plato_official) with the 09687d2 build (hybrid_official) run as pure plain
# ADMM, from a cold start, sigma 1, fixed and legacy_adaptive, 6,000 iterations (past the adaptive schedule and the
# first Monitor1 updates at k > 5000), on four PLATO instances. Both builds: Release, sm_90, deterministic CSR_ALG2 SpMV.
#  A  without validation (internal tolerance 1e-14, unreachable): the returned iterate (the best-KKT iterate of the run,
#     the solver's rule without validation) and the per-iteration history without its time column must be identical;
#  B  with external validation every 1,000 iterations (criteria 1e-14, unreachable) and checkpoints at 3,000 and 6,000:
#     the checkpoint iterates of the two builds must be identical, and each build's history must equal its history of
#     A (the validator reads the iterate and never changes the trajectory).
# Writes logs/stage0/compare_builds.txt; exit status 0 only if everything is identical.
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
NEW=$HOME/cuadmm-builds/plato_official/cuadmm_exe
REF=$HOME/cuadmm-builds/hybrid_official/cuadmm_exe
TXT=$HOME/cuadmm-data/plato_kocvara/txt
OUT=$HOME/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma/compare_builds_v2
REP=$EXP/logs/stage0/compare_builds.txt
mkdir -p $OUT $EXP/logs/stage0
{
  echo "date: $(date -Iseconds) host: $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "new (plain-ADMM-only, 21464b1 tree; library identical to d18da08): $NEW exe $(sha256sum $NEW | cut -c1-16) lib $(sha256sum $(dirname $NEW)/libcuadmm_lib.so | cut -c1-16)"
  echo "ref (09687d2, hybrid campaign official build): $REF exe $(sha256sum $REF | cut -c1-16) lib $(sha256sum $(dirname $REF)/libcuadmm_lib.so | cut -c1-16)"
} > $REP
notime() {
  python3 - "$1" <<'EOF'
import csv, sys
r = csv.reader(open(sys.argv[1]))
h = next(r)
keep = [i for i, c in enumerate(h) if not c.lower().startswith('time')]
w = csv.writer(sys.stdout)
w.writerow([h[i] for i in keep])
for row in r:
    w.writerow([row[i] for i in keep])
EOF
}
same() { cmp -s "$1" "$2" && echo identical || echo DIFFERENT; }
for ds in trto3 vibra2 mater-2 shmup2; do
  for pol in fixed legacy_adaptive; do
    common="$TXT/$ds --algorithm admm --sigma-policy $pol --sig 1 --tol 1e-14 --max-iter 6000"
    for b in new ref; do
      exe=$NEW; [ $b = ref ] && exe=$REF
      d=$OUT/${ds}_${pol}_${b}_A; mkdir -p $d
      $exe $common --save-solution $d/sol --history $d/history.csv --summary $d/summary.json > $d/stdout.log 2>&1
      echo "A $b $ds $pol exit $?" >> $REP
      notime $d/history.csv > $d/history.notime.csv
      d=$OUT/${ds}_${pol}_${b}_B; mkdir -p $d
      $exe $common --validate-interval 1000 --validate-tol 1e-14 --validate-threads 8 --checkpoint-iters 3000,6000 --checkpoint-dir $d/ckpt \
        --history $d/history.csv --summary $d/summary.json > $d/stdout.log 2>&1
      echo "B $b $ds $pol exit $?" >> $REP
      notime $d/history.csv > $d/history.notime.csv
    done
    P=$OUT/${ds}_${pol}
    line="$ds $pol | A returned X,y,S:"
    for v in X y S; do line="$line $v $(same ${P}_new_A/sol/$v.txt ${P}_ref_A/sol/$v.txt)"; done
    line="$line | A history: $(same ${P}_new_A/history.notime.csv ${P}_ref_A/history.notime.csv)"
    for k in 3000 6000; do
      line="$line | B iter $k X,y,S:"
      for v in X y S; do line="$line $v $(same ${P}_new_B/ckpt/iter_$k/$v.txt ${P}_ref_B/ckpt/iter_$k/$v.txt)"; done
    done
    line="$line | B history new vs ref: $(same ${P}_new_B/history.notime.csv ${P}_ref_B/history.notime.csv)"
    line="$line | history A vs B: new $(same ${P}_new_A/history.notime.csv ${P}_new_B/history.notime.csv), ref $(same ${P}_ref_A/history.notime.csv ${P}_ref_B/history.notime.csv)"
    line="$line | iterations, final sigma: $(python3 -c "import json,sys; print([(json.loads(open(f).readlines()[-1])['iterations'], json.loads(open(f).readlines()[-1])['final_sig']) for f in sys.argv[1:]])" ${P}_new_A/summary.json ${P}_ref_A/summary.json ${P}_new_B/summary.json ${P}_ref_B/summary.json)"
    echo "$line" >> $REP
  done
done
# every comparison and every run's exit status go into the report; the verdict is read back from it
nbad=$(grep -c -E "DIFFERENT|exit [1-9]" $REP)
echo "overall: $([ $nbad = 0 ] && echo 'ALL IDENTICAL' || echo "DIFFERENCES OR FAILED RUNS FOUND ($nbad lines)")" >> $REP
cat $REP
[ $nbad = 0 ]
