#!/bin/bash
# Final bundles of the campaign (run on a CPU node after the analysis and the commit):
#  1. results/iterate_manifest.sha256: SHA-256 of every saved iterate (returned/ and final/ X, y, S of every run), which
#     stay on the cluster in ~/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma/runs/ (several GB)
#  2. plato_cold_plain_admm_sigma_bundle.tar.gz: the experiment directory (protocols, data provenance, scripts, plans,
#     every run's summary, driver record, validation history, sigma log, stdout and revalidation, the results tables,
#     plots, workbook, logs, and the iterate manifest)
#  3. plato_cold_plain_admm_sigma_histories.tar: the per-iteration histories (history.csv.gz) of every run (already
#     gzipped, so the tar is not compressed again)
# Each with a .sha256 file; bundle 2 and its checksum are also copied to $HOME.
# usage: make_bundle.sh [--manifest-only]
set -eu
EXP=/n/home00/yukuanwei/cuadmm-worktrees/plato-cold/experiments/2026-09-29_plato_cold_plain_admm_sigma
BIG=$HOME/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma
OUT=$BIG/bundle
mkdir -p $OUT
cd $BIG
find runs -type f \( -path '*/returned/*' -o -path '*/final/*' \) | sort | xargs sha256sum > $EXP/results/iterate_manifest.sha256
echo "iterate manifest: $(wc -l < $EXP/results/iterate_manifest.sha256) files, $(find runs -type f \( -path '*/returned/*' -o -path '*/final/*' \) -printf '%s\n' | awk '{s+=$1} END {printf "%.2f GB", s/1e9}')"
[ "${1:-}" = "--manifest-only" ] && exit 0
N=plato_cold_plain_admm_sigma
tar -czf $OUT/${N}_bundle.tar.gz -C $(dirname $EXP) --exclude='__pycache__' --exclude='STOP' $(basename $EXP)
find runs -name history.csv.gz | sort > $OUT/${N}_histories.list
tar -cf $OUT/${N}_histories.tar -T $OUT/${N}_histories.list
(cd $OUT && sha256sum ${N}_bundle.tar.gz > ${N}_bundle.tar.gz.sha256 && sha256sum ${N}_histories.tar > ${N}_histories.tar.sha256)
cp $OUT/${N}_bundle.tar.gz $OUT/${N}_bundle.tar.gz.sha256 $HOME/
ls -la $OUT
cat $OUT/*.sha256
