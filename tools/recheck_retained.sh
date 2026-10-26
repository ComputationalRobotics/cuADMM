#!/bin/bash
# Rechecks retained comparison outputs with the current tools/compare_runs.py (no solver run):
# usage: tools/recheck_retained.sh <compare directory of a tag> [rtol]
# For every <group>/<case>/{a,b} under the directory, prints the verdict (IDENTICAL, DIFFERENT, INVALID) and writes
# <dir>/recheck.csv (case, verdict, exit code, first difference). Exit code: the worst verdict (0, 1, 2).
set -u
DIR=$1; RTOL=${2:-0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$HOME/cuadmm-analysis-env/bin/python}
OUT=$DIR/recheck.csv
echo "case,verdict,exit,first_difference" > "$OUT"
worst=0; n=0; ni=0; nd=0; nv=0
for c in "$DIR"/*/*/; do
  [ -d "$c/a" ] && [ -d "$c/b" ] || continue
  name=$(basename "$(dirname "$c")")/$(basename "$c")
  log=$(PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$HERE/tools/compare_runs.py" compare "$c/a" "$c/b" --rtol "$RTOL" 2>&1); rc=$?
  verdict=$(echo "$log" | grep -o -E "RESULT: [A-Z]+" | sed 's/RESULT: //'); [ -z "$verdict" ] && verdict=ERROR
  fd=$(echo "$log" | grep -E "FIRST DIFFERENCE|INVALID:|ITERATION ID MISMATCH" | head -1 | tr ',' ';')
  echo "$name,$verdict,$rc,$fd" >> "$OUT"
  n=$((n+1)); case $rc in 0) ni=$((ni+1));; 1) nd=$((nd+1));; *) nv=$((nv+1));; esac
  [ $rc -gt $worst ] && worst=$rc
done
echo "recheck of $DIR (rtol $RTOL): $n cases, $ni identical, $nd different, $nv invalid or error -> $OUT"
exit $worst
