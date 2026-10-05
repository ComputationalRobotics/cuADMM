#!/usr/bin/env python3
"""Standalone validator of a returned cuADMM iterate (independent of the C++ ExternalValidator in the solver).

Reads the converted problem (blk.txt, At.txt, b.txt, C.txt, con_num.txt, dimacs_blocks.txt) and an iterate (X.txt,
y.txt, S.txt; unscaled, as written by --save-solution / --save-final-dir), and computes with NumPy (dense eigenvalues
per block):
  cuADMM measures   eta_p = ||A(X)-b||/(1+||b||), eta_d = ||A*y+S-C||/(1+||C||), eta_g, eta = max
                    X cone max_b max(0,-lmin(X_b))/(1+||X_b||_F); S and Z = C - A*y cones /(1+||C||)
  DIMACS (7th Challenge, z = S) err1..err6, max_abs_DIMACS, and err4/err6 for z = C - A*y
  objectives        <C,X>, b^T y (cuADMM), -<C,X> (SDPA convention) and the relative objective error against the
                    reference of data/reference_objectives.json, with its uncertainty
  status            PRACTICAL (eta <= 1e-4 and every normalized cone violation <= 1e-4, finite) and STRICT
                    (max_abs_DIMACS <= 1e-6 and objective error <= 1e-6, finite; the objective part is "undecidable"
                    when the reference uncertainty exceeds 1e-6)
usage: dimacs_validate.py <problem dir> <iterate dir> <out.json> [--problem-name NAME]"""
import argparse, gzip, json, math, os
import numpy as np
import scipy.sparse as sp

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRACTICAL, STRICT = 1e-4, 1e-6


def vec_file(path, n=None):
    for p in (path, path + '.gz'):
        if os.path.exists(p):
            with (gzip.open(p, 'rt') if p.endswith('.gz') else open(p)) as f:
                v = np.array([float(l) for l in f if l.strip()])
            if n is not None and v.size != n:
                raise SystemExit(f'{p}: {v.size} entries, expected {n}')
            return v
    raise SystemExit(f'missing {path}')


def load_problem(d):
    blk = [(l.split()[0], int(l.split()[1])) for l in open(os.path.join(d, 'blk.txt')) if l.strip()]
    kp = os.path.join(d, 'dimacs_blocks.txt')
    kinds = [l.split()[0] for l in open(kp) if l.strip()] if os.path.exists(kp) else ['s' if t == 's' else 'l' for t, _ in blk]
    m = int(open(os.path.join(d, 'con_num.txt')).read().split()[0])
    n = sum(k * (k + 1) // 2 if t == 's' else k for t, k in blk)
    At = np.loadtxt(os.path.join(d, 'At.txt'), ndmin=2)
    A = sp.csr_matrix((At[:, 2], (At[:, 1].astype(int), At[:, 0].astype(int))), shape=(m, n))
    def sv(name, size):
        v = np.zeros(size)
        if os.path.getsize(os.path.join(d, name)):
            x = np.loadtxt(os.path.join(d, name), ndmin=2)
            np.add.at(v, x[:, 0].astype(int), x[:, 2])
        return v
    return blk, kinds, m, n, A, sv('b.txt', m), sv('C.txt', n)


def smat(v, k):
    # svec order: column by column (j), rows i <= j; index j(j+1)/2 + i
    ju = np.repeat(np.arange(k), np.arange(1, k + 1))
    iu = np.concatenate([np.arange(j + 1) for j in range(k)]) if k else np.zeros(0, int)
    M = np.zeros((k, k))
    M[iu, ju] = v / np.where(iu == ju, 1.0, math.sqrt(2.0))
    return M + np.triu(M, 1).T


def evaluate(blk, kinds, A, b, C, X, y, S):
    Z = C - A.T @ y
    rp = A @ X - b
    rd = A.T @ y + S - C
    off, lminX, lminS, lminZ, frobX, rdK_psd, rd_lin2 = 0, [], [], [], [], 0.0, 0.0
    maxC = 0.0
    for (t, k), kind in zip(blk, kinds):
        L = k * (k + 1) // 2 if t == 's' else k
        xb, sb, zb, rb, cb = X[off:off + L], S[off:off + L], Z[off:off + L], rd[off:off + L], C[off:off + L]
        if t == 's':
            lminX.append(float(np.linalg.eigvalsh(smat(xb, k))[0]))
            lminS.append(float(np.linalg.eigvalsh(smat(sb, k))[0]))
            lminZ.append(float(np.linalg.eigvalsh(smat(zb, k))[0]))
            maxC = max(maxC, float(np.max(np.abs(smat(cb, k)))) if k else 0.0)
        else:
            lminX.append(float(np.min(xb)))
            lminS.append(float(np.min(sb)))
            lminZ.append(float(np.min(zb)))
            maxC = max(maxC, float(np.max(np.abs(cb))) if L else 0.0)
        frobX.append(float(np.linalg.norm(xb)))
        if kind == 's':
            rdK_psd += float(np.linalg.norm(rb))
        else:
            rd_lin2 += float(rb @ rb)
        off += L
    nb2, nC2, binf = float(np.linalg.norm(b)), float(np.linalg.norm(C)), float(np.max(np.abs(b)))
    pobj, dobj = float(C @ X), float(b @ y)
    den = 1 + abs(pobj) + abs(dobj)
    finite = all(np.all(np.isfinite(v)) for v in (X, y, S))
    r = dict(finite=bool(finite), pobj=pobj, dobj=dobj, sdpa_objective=-pobj, sdpa_dual_objective=-dobj,
             eta_p=float(np.linalg.norm(rp)) / (1 + nb2), eta_d=float(np.linalg.norm(rd)) / (1 + nC2), eta_g=abs(pobj - dobj) / den,
             X_cone=max(max(0.0, -l) / (1 + f) for l, f in zip(lminX, frobX)), S_cone=max(max(0.0, -l) for l in lminS) / (1 + nC2),
             Z_cone=max(max(0.0, -l) for l in lminZ) / (1 + nC2), min_eig_X=min(lminX), min_eig_S=min(lminS), min_eig_Z=min(lminZ),
             X_negative_blocks=int(sum(1 for l in lminX if l < 0)))
    r['eta'] = max(r['eta_p'], r['eta_d'], r['eta_g'])
    e = [float(np.linalg.norm(rp)) / (1 + binf), max(0.0, -min(lminX)) / (1 + binf),
         (rdK_psd + math.sqrt(rd_lin2)) / (1 + maxC), max(0.0, -min(lminS)) / (1 + maxC), (pobj - dobj) / den, float(X @ S) / den]
    r.update(dimacs=e, max_abs_dimacs=max(abs(x) for x in e), err6_defined=e[1] == 0 and e[3] == 0,
             err4_z=max(0.0, -min(lminZ)) / (1 + maxC), err6_z=float(X @ Z) / den, norm_b_inf=binf, norm_C_inf=maxC)
    return r


def classify(r, refrow):
    ok = lambda x: x is not None and math.isfinite(x)
    practical = r['finite'] and all(ok(r[k]) for k in ('eta', 'X_cone', 'S_cone', 'Z_cone')) and r['eta'] <= PRACTICAL and \
        max(r['X_cone'], r['S_cone'], r['Z_cone']) <= PRACTICAL
    obj_err = unc = None
    if refrow:
        ref = refrow['reference']
        obj_err = abs(r['sdpa_objective'] - ref) / (1 + abs(ref))
        unc = refrow['reference_uncertainty_rel']
    dimacs_ok = r['finite'] and ok(r['max_abs_dimacs']) and r['max_abs_dimacs'] <= STRICT
    if obj_err is None:
        obj_state = 'no reference'
    elif unc > STRICT:
        obj_state = f'undecidable (reference uncertainty {unc:.1e})'
    else:
        obj_state = 'pass' if obj_err <= STRICT else 'fail'
    strict = practical and dimacs_ok and (obj_state == 'pass' or obj_state.startswith('undecidable'))
    status = 'STRICT_VALIDATED' if strict else ('PRACTICAL_VALIDATED' if practical else 'NOT_VALIDATED')
    plato = 'clean' if r['max_abs_dimacs'] <= 1e-4 else ('a' if r['max_abs_dimacs'] <= 1e-2 else 'f')
    return dict(practical_ok=bool(practical), dimacs_strict_ok=bool(dimacs_ok), objective_error=obj_err, reference_uncertainty=unc,
                objective_check=obj_state, status=status, plato_class=plato,
                strict_note='objective reference uncertain' if (strict and obj_state.startswith('undecidable')) else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('problem_dir')
    ap.add_argument('iterate_dir')
    ap.add_argument('out')
    ap.add_argument('--problem-name', default=None)
    a = ap.parse_args()
    blk, kinds, m, n, A, b, C = load_problem(a.problem_dir)
    X, y, S = vec_file(os.path.join(a.iterate_dir, 'X.txt'), n), vec_file(os.path.join(a.iterate_dir, 'y.txt'), m), vec_file(os.path.join(a.iterate_dir, 'S.txt'), n)
    r = evaluate(blk, kinds, A, b, C, X, y, S)
    name = a.problem_name or os.path.basename(os.path.normpath(a.problem_dir))
    refs = json.load(open(os.path.join(EXP, 'data', 'reference_objectives.json')))
    r.update(classify(r, refs.get(name)), problem=name, reference=refs.get(name, {}).get('reference'))
    json.dump(r, open(a.out, 'w'), indent=1)
    print(json.dumps({k: r[k] for k in ('problem', 'status', 'plato_class', 'eta', 'max_abs_dimacs', 'X_cone', 'S_cone', 'Z_cone', 'objective_error',
                                        'objective_check')}))


if __name__ == '__main__':
    main()
