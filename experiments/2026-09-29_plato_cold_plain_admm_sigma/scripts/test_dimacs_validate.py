#!/usr/bin/env python3
"""Cross-checks of scripts/dimacs_validate.py (the standalone validator), written to logs/validator_crosscheck.json.

1. svec/smat: a random symmetric matrix packed by the definition (index j(j+1)/2 + i, off-diagonals times sqrt 2) and
   unpacked again.
2. Hand-computed problem: the same data as the C++ test DimacsValidator.HandComputedSmallProblem (a 2 x 2 PSD block, a
   2-entry nonnegative block and a 1 x 1 block that SDPA declares PSD), written in the cuADMM TXT format; every DIMACS
   error, eta and the cone measures are recomputed here from full matrices with plain loops.
3. An independent solver: Clarabel (interior point, through CVXPY; not MOSEK) solves converted PLATO instances; its
   primal X (variables), dual y (equality multipliers) and dual S (cone multipliers) go through the validator, which
   must report DIMACS errors at the interior-point accuracy, the right sign of err5 and an objective equal to the
   reference. y is taken with the sign for which A*y + S = C (CVXPY's sign convention for equality duals is recorded).
usage: test_dimacs_validate.py [problem ...]   (default: trto1 buck1 vibra1 shmup1 mater-1)"""
import json, math, os, sys, tempfile
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dimacs_validate as dv

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT = os.path.expanduser('~/cuadmm-data/plato_kocvara/txt')
R2 = math.sqrt(2.0)
out = {}
fails = []


def check(name, got, want, tol):
    ok = bool(abs(got - want) <= tol * max(1.0, abs(want)))
    if not ok:
        fails.append(f'{name}: got {got!r} want {want!r}')
    return dict(got=got, want=want, ok=ok)


def svec(M):
    k = M.shape[0]
    return np.array([M[i, j] * (1.0 if i == j else R2) for j in range(k) for i in range(j + 1)])


# 1. svec round trip
rng = np.random.default_rng(7)
B = rng.standard_normal((6, 6))
B = B + B.T
out['svec_roundtrip_max_err'] = float(np.max(np.abs(dv.smat(svec(B), 6) - B)))
if out['svec_roundtrip_max_err'] > 1e-15:
    fails.append('svec round trip')

# 2. hand-computed problem (the C++ test's data)
with tempfile.TemporaryDirectory() as d:
    open(os.path.join(d, 'blk.txt'), 'w').write('s 2\nl 2\nl 1\n')
    open(os.path.join(d, 'dimacs_blocks.txt'), 'w').write('s\nl\ns\n')
    open(os.path.join(d, 'con_num.txt'), 'w').write('2\n')
    # At (svec_idx con_idx val): A1 = [[1,1],[1,0]] (+ lp0 = 1), A2 = [[0,0],[0,2]] (+ lp1 = -1, 1x1 = 1)
    open(os.path.join(d, 'At.txt'), 'w').write(f'0 0 1\n1 0 {R2!r}\n3 0 1\n2 1 2\n4 1 -1\n5 1 1\n')
    open(os.path.join(d, 'b.txt'), 'w').write('0 0 1\n1 0 2\n')
    open(os.path.join(d, 'C.txt'), 'w').write(f'0 0 1\n1 0 {0.5 * R2!r}\n2 0 3\n3 0 2\n4 0 -4\n5 0 0.25\n')
    blk, kinds, m, n, A, b, C = dv.load_problem(d)
X = np.array([2.0, R2, -1.0, 3.0, -0.5, 0.1])
y = np.array([0.5, -1.0])
S = np.array([1.0, 0.0, -2.0, 0.3, 0.2, -0.05])
r = dv.evaluate(blk, kinds, A, b, C, X, y, S)
Xm, Sm, Cm = np.array([[2, 1], [1, -1.]]), np.array([[1, 0], [0, -2.]]), np.array([[1, .5], [.5, 3]])
A1, A2 = np.array([[1, 1], [1, 0.]]), np.array([[0, 0], [0, 2.]])
xl, sl, cl, a1l, a2l = [3, -0.5, 0.1], [0.3, 0.2, -0.05], [2, -4, 0.25], [1, 0, 0], [0, -1, 1]
ip = lambda P, Q: sum(P[i][j] * Q[i][j] for i in range(2) for j in range(2))
AX = [ip(A1, Xm) + sum(p * q for p, q in zip(a1l, xl)), ip(A2, Xm) + sum(p * q for p, q in zip(a2l, xl))]
rp = [AX[0] - 1, AX[1] - 2]
Rm = y[0] * A1 + y[1] * A2 + Sm - Cm
rl = [y[0] * a1l[i] + y[1] * a2l[i] + sl[i] - cl[i] for i in range(3)]
pobj = ip(Cm, Xm) + sum(p * q for p, q in zip(cl, xl))
dobj = 1 * y[0] + 2 * y[1]
den = 1 + abs(pobj) + abs(dobj)
lmin = lambda P: 0.5 * (P[0][0] + P[1][1]) - math.sqrt(0.25 * (P[0][0] - P[1][1]) ** 2 + P[0][1] ** 2)
binf, cinf = 2.0, 4.0
RF = math.sqrt(sum(Rm[i][j] ** 2 for i in range(2) for j in range(2)))
want = [math.hypot(*rp) / (1 + binf), max(0, -min(lmin(Xm), *xl)) / (1 + binf),
        (RF + abs(rl[2]) + math.hypot(rl[0], rl[1])) / (1 + cinf),  # the 1x1 block is PSD for DIMACS: |r| by itself
        max(0, -min(lmin(Sm), *sl)) / (1 + cinf), (pobj - dobj) / den, (ip(Xm, Sm) + sum(p * q for p, q in zip(xl, sl))) / den]
normC = math.sqrt(sum(Cm[i][j] ** 2 for i in range(2) for j in range(2)) + sum(c * c for c in cl))
normb = math.sqrt(1 + 4)
nrd = math.sqrt(RF ** 2 + sum(v * v for v in rl))
hand = {f'err{i + 1}': check(f'err{i + 1}', r['dimacs'][i], want[i], 1e-13) for i in range(6)}
hand['eta_p'] = check('eta_p', r['eta_p'], math.hypot(*rp) / (1 + normb), 1e-13)
hand['eta_d'] = check('eta_d', r['eta_d'], nrd / (1 + normC), 1e-13)
hand['eta_g'] = check('eta_g', r['eta_g'], abs(pobj - dobj) / den, 1e-13)
hand['S_cone'] = check('S_cone', r['S_cone'], max(0, -lmin(Sm), 0.05) / (1 + normC), 1e-13)
fx = [math.sqrt(sum(Xm[i][j] ** 2 for i in range(2) for j in range(2))), math.hypot(3, -0.5), 0.1]
hand['X_cone'] = check('X_cone', r['X_cone'], max(max(0, -lmin(Xm)) / (1 + fx[0]), 0.5 / (1 + fx[1]), 0), 1e-13)
hand['max_abs_dimacs'] = check('max_abs', r['max_abs_dimacs'], max(abs(w) for w in want), 1e-13)
out['hand_computed'] = hand

# 3. Clarabel solutions of converted PLATO instances
names = sys.argv[1:] or ['trto1', 'buck1', 'vibra1', 'shmup1', 'mater-1']
if names:
    import cvxpy as cp
    refs = json.load(open(os.path.join(EXP, 'data', 'reference_objectives.json')))
out['clarabel'] = {}
for name in names:
    blk, kinds, m, n, A, b, C = dv.load_problem(os.path.join(TXT, name))
    parts, cones, vars_ = [], [], []
    for t, k in blk:
        if t == 's':
            V = cp.Variable((k, k), symmetric=True)
            cones.append(V >> 0)
            ii = [(i, j) for j in range(k) for i in range(j + 1)]
            T = sp.csr_matrix(([1.0 if i == j else R2 for i, j in ii], (range(len(ii)), [j * k + i for i, j in ii])), shape=(len(ii), k * k))
            parts.append(T @ cp.vec(V, order='F'))
        else:
            V = cp.Variable(k)
            cones.append(V >= 0)
            parts.append(V)
        vars_.append((t, k, V))
    v = cp.hstack(parts)
    eq = A @ v == b
    prob = cp.Problem(cp.Minimize(C @ v), cones + [eq])
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    Xs = np.concatenate([svec(V.value) if t == 's' else V.value for t, k, V in vars_])
    Ss = np.concatenate([svec(c.dual_value) if t == 's' else c.dual_value for (t, k, V), c in zip(vars_, cones)])
    ydual = np.asarray(eq.dual_value).ravel()
    res = {}
    for sign in (+1, -1):
        rr = dv.evaluate(blk, kinds, A, b, C, Xs, sign * ydual, Ss)
        res[sign] = rr
    sign = min(res, key=lambda s: res[s]['dimacs'][2])
    rr = res[sign]
    rr.update(dv.classify(rr, refs.get(name)))
    rec = dict(status=prob.status, y_sign_vs_cvxpy_dual=sign, dimacs=rr['dimacs'], max_abs_dimacs=rr['max_abs_dimacs'], eta=rr['eta'],
               X_cone=rr['X_cone'], S_cone=rr['S_cone'], Z_cone=rr['Z_cone'], sdpa_objective=rr['sdpa_objective'], objective_error=rr['objective_error'],
               err3_other_sign=res[-sign]['dimacs'][2], validator_status=rr['status'])
    out['clarabel'][name] = rec
    # an interior-point solution: every DIMACS error small (the solver's tolerance is 1e-10; allow for its scaling),
    # the wrong sign of y must give a large dual residual, and the objective must match the reference
    if not (rr['max_abs_dimacs'] < 1e-6 and res[-sign]['dimacs'][2] > 1e-3 and rr['objective_error'] < 1e-6):
        fails.append(f'clarabel {name}: {rec}')
    print(name, json.dumps(rec), flush=True)
out['failures'] = fails
out['all_ok'] = not fails
json.dump(out, open(os.path.join(EXP, 'logs', 'validator_crosscheck.json'), 'w'), indent=1)
print('ALL OK' if not fails else 'FAILURES:\n' + '\n'.join(fails))
sys.exit(1 if fails else 0)
