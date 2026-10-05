#!/bin/bash
# Recomputes the geqrf workspace-fill verdict from the raw RESULT lines (the verdict printed by job 49476960 used the
# wrong awk fields, 10/12/14 instead of 11/13/15, and is not used).
F=$(dirname "$0")/../../logs/diag/geqrf_workspace_fills.txt
for nm in "1681 21" "1680 20"; do for kind in first generic; do
  L=$(grep "RESULT n ${nm% *} m ${nm#* } kind $kind " $F)
  echo "n,m=$nm $kind: fills $(echo "$L" | wc -l); distinct (hashR,hashQ): $(echo "$L" | awk '{print $11, $13}' | sort -u | wc -l); stable across reps: $(echo "$L" | awk '{print $15}' | sort -u | tr '\n' ' ')"
done; done
