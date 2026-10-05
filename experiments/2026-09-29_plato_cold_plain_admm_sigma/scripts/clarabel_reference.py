#!/usr/bin/env python3
"""Independent check of the conversion's orientation and sign: solves the CONVERTED cuADMM data (min <C,X> s.t.
A(X) = b, X in K, read back from the TXT files) with the interior-point solver Clarabel through CVXPY (not MOSEK), and
compares -<C,X*> (the SDPA-convention optimal value) with Kocvara's published PENSDP objective.
Writes data/clarabel_reference.csv.  usage: clarabel_reference.py <problem> [<problem> ...]"""
import csv, json, math, os, sys, time
import numpy as np
import scipy.sparse as sp
import cvxpy as cp

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT = os.path.expanduser('~/cuadmm-data/plato_kocvara/txt')
REF = json.load(open(os.path.join(EXP, 'data', 'published_objectives.json')))


def load(d):
    blk = [(l.split()[0], int(l.split()[1])) for l in open(os.path.join(d, 'blk.txt')) if l.strip()]
    m = int(open(os.path.join(d, 'con_num.txt')).read().split()[0])
    n = sum(k * (k + 1) // 2 if t == 's' else k for t, k in blk)
    At = np.loadtxt(os.path.join(d, 'At.txt'), ndmin=2)
    A = sp.csr_matrix((At[:, 2], (At[:, 1].astype(int), At[:, 0].astype(int))), shape=(m, n))
    def vec(name, size):
        v = np.zeros(size)
        if os.path.getsize(os.path.join(d, name)):
            x = np.loadtxt(os.path.join(d, name), ndmin=2)
            np.add.at(v, x[:, 0].astype(int), x[:, 2])
        return v
    return blk, m, n, A, vec('b.txt', m), vec('C.txt', n)


def solve(name):
    blk, m, n, A, b, C = load(os.path.join(TXT, name))
    parts, cons = [], []
    for t, k in blk:
        if t == 's':
            Xb = cp.Variable((k, k), symmetric=True)
            cons.append(Xb >> 0)
            # svec of Xb: column-major upper triangle, off-diagonals times sqrt(2), from cp.vec (column-major)
            rows, cols, vals = [], [], []
            idx = 0
            for j in range(k):
                for i in range(j + 1):
                    rows.append(idx)
                    cols.append(j * k + i)
                    vals.append(1.0 if i == j else math.sqrt(2.0))
                    idx += 1
            T = sp.csr_matrix((vals, (rows, cols)), shape=(k * (k + 1) // 2, k * k))
            parts.append(T @ cp.vec(Xb, order='F'))
        else:
            xb = cp.Variable(k)
            cons.append(xb >= 0)
            parts.append(xb)
    v = cp.hstack(parts)
    prob = cp.Problem(cp.Minimize(C @ v), cons + [A @ v == b])
    t0 = time.time()
    prob.solve(solver=cp.CLARABEL, verbose=False)
    dt = time.time() - t0
    ref = REF.get(name, {}).get('objective')
    val = -prob.value if prob.value is not None else None  # SDPA convention
    return dict(problem=name, status=prob.status, clarabel_value_cuadmm_convention=prob.value, sdpa_convention=val,
                published=ref, published_digits=REF.get(name, {}).get('digits'),
                rel_diff=(abs(val - ref) / (1 + abs(ref))) if (val is not None and ref is not None) else None, solve_s=dt, m=m, vec_len=n)


if __name__ == '__main__':
    out = os.path.join(EXP, 'data', 'clarabel_reference.csv')
    rows = []
    for name in sys.argv[1:]:
        r = solve(name)
        rows.append(r)
        print(json.dumps(r), flush=True)
    new = not os.path.exists(out)
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
