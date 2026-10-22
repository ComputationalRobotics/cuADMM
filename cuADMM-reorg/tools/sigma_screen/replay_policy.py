#!/usr/bin/env python3
"""replay_policy.py: drive the balance_freeze rule of the CPU replica (cuadmm_emu.py v1.2, BalanceFreezeState) with the
residual sequence of a reference history, and write the sigma of every iteration and the sigma log. The C++ policy
(src/admm/penalty_policy.cc) is checked against the output in test/penalty_policy_test.hpp
(test/data/golden/balance_freeze_cases.csv): the two transcriptions of plan Section 3.5 must agree bit for bit.

usage: replay_policy.py <history.csv> <out.csv> <case> [<case> ...]
  case: <sigma0>@balance_freeze[:key=value,...] (the keys are the option names of penalty.*; see cuadmm_emu.parse_rule)
out.csv columns: case,iter,sigma,log   (sigma: the value returned for the next iteration, shortest round-trip repr;
log: the rule names of the sigma-log entries recorded at that iteration joined by ';', else empty; the freeze entry is
recorded even when sigma does not move)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuadmm_emu as ce  # noqa: E402


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    history, out = sys.argv[1], sys.argv[2]
    cases = sys.argv[3:]
    rows = []
    with open(history) as f:
        header = f.readline().strip().split(',')
        col = {name: i for i, name in enumerate(header)}
        ip = col.get('eta_p', col.get('errRp'))
        idd = col.get('eta_d', col.get('errRd'))
        ig = col.get('eta_g', col.get('relgap'))
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            rows.append((int(parts[0]), float(parts[ip]), float(parts[idd]), float(parts[ig])))
    with open(out, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['case', 'iter', 'sigma', 'log'])
        for case in cases:
            sigma0_text, spec = case.split('@', 1)
            rule = ce.parse_rule(spec)
            st = ce.BalanceFreezeState(rule, float(sigma0_text))
            for it, ep, ed, eg in rows:
                n = len(st.log)
                sig = st.update(it, ep, ed, eg)
                logged = ';'.join(e['rule'] for e in st.log[n:])
                w.writerow([case, it, repr(sig), logged])
            print('%s: %d iterations, %d log entries, frozen %s at %d (%s), final sigma %r' %
                  (case, len(rows), len(st.log), st.frozen, st.freeze_at, st.freeze_reason, st.sig))
    return 0


if __name__ == '__main__':
    sys.exit(main())
