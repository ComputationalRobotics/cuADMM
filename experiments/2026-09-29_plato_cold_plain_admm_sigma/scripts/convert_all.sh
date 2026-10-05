#!/bin/bash
# Converts the 26 Kocvara instances (SDPA, from the zips) to cuADMM TXT and runs the independent checks; compares the
# 16 instances also mirrored on PLATO with the Kocvara originals. CPU node.
EXP=$(cd "$(dirname "$0")/.." && pwd); D=$HOME/cuadmm-data/plato_kocvara; PY=$HOME/cuadmm-analysis-env/bin/python
mkdir -p $D/txt $EXP/logs/conversion
for fam in buck mater shmup trto vibra; do
  for member in $(unzip -Z1 $D/raw/$fam.zip); do
    p=${member%.dat-s}; src="zip:$D/raw/$fam.zip::$member"
    t0=$(date +%s.%N)
    $PY $EXP/scripts/sdpa_to_cuadmm.py "$src" $D/txt/$p > $EXP/logs/conversion/$p.convert.log 2>&1; rc=$?
    t1=$(date +%s.%N)
    if [ $rc -eq 0 ]; then $PY $EXP/scripts/check_conversion.py "$src" $D/txt/$p > $EXP/logs/conversion/$p.check.log 2>&1; rc2=$?; else rc2=-1; fi
    t2=$(date +%s.%N)
    echo "$p convert_rc=$rc check_rc=$rc2 convert_s=$(echo "$t1 - $t0" | bc) check_s=$(echo "$t2 - $t1" | bc) | $(tail -1 $EXP/logs/conversion/$p.check.log 2>/dev/null | cut -c1-220)"
  done
done
echo "== PLATO mirror vs Kocvara originals"
for f in $D/plato_mirror/*.dat-s.gz; do
  p=$(basename $f .dat-s.gz); fam=$(echo $p | sed 's/[-0-9]*$//'); member=$p.dat-s
  a=$(zcat $f | sha256sum | cut -c1-16); b=$(unzip -p $D/raw/$fam.zip $member | sha256sum | cut -c1-16)
  if [ "$a" = "$b" ]; then echo "$p: identical bytes ($a)"; else
    echo "$p: bytes differ (plato $a, kocvara $b): $($PY - <<PYEOF
import gzip, zipfile, re
def ent(t):
    L=[l for l in t.splitlines() if l.strip() and l.lstrip()[0] not in '"*']
    num=re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
    m=int(float(num.findall(L[0])[0])); nb=int(float(num.findall(L[1])[0])); v=num.findall(' '.join(L[2:]))
    return m, [int(float(x)) for x in v[:nb]], [float(x) for x in v[nb:nb+m]], sorted(tuple(float(x) for x in v[nb+m+5*k:nb+m+5*k+5]) for k in range((len(v)-nb-m)//5))
A=ent(gzip.open('$f','rt').read()); B=ent(zipfile.ZipFile('$D/raw/$fam.zip').read('$member').decode())
print('parsed content identical' if A==B else 'PARSED CONTENT DIFFERS: m %s/%s blocks %s c %s entries %d/%d'%(A[0],B[0],A[1]==B[1],A[2]==B[2],len(A[3]),len(B[3])))
PYEOF
)"; fi
done
