#!/usr/bin/env python3
"""cuadmm_emu.py -- CPU (NumPy/SciPy) replica of cuADMM's plain two-block ADMM solve, for small problems.

Replicates SDPSolver::init + SDPSolver::solve of cuADMM at main = 2d8a755 (src/solver.cu) with switch_admm = 0
(plain ADMM from the first iteration), the CLI defaults of src/main.cu (EIG_FP64 projections), a cold or warm start,
and the validation-aware stopping of the PLATO campaign (--validate-interval, --validate-tol, --strict-dimacs-tol).
Line numbers below refer to src/solver.cu at 2d8a755 unless stated otherwise.

Sigma rules ("rule"):
  old    9bb4ba5 (= campaign build 21464b1, identical src/ and include/): schedule with factor 2, change if
         prim_win > 1.35 dual_win (or the reverse), sigma in [1e-6, 1e6], Monitor1 "chasing" correction for k > 5000
         every 25 iterations (include/cuadmm/monitors.h, identical at both commits)
  new    2d8a755: the same schedule, change only if the winner's count exceeds max(1, 1.35 * the other's),
         sigma in [1e-3, 1e3], Monitor1 disabled (solver.cu:2052-2085, 1015-1016)
  fixed  SigmaPolicy::FIXED: sigma = sigma_0 for every iteration
  A custom rule can be passed as a SigmaRule (guard, ratio, bounds, monitor on/off, freeze_after, deadband).
  Variants that are NOT in any cuADMM commit (v1.1, for the rule comparison of 2026-10-19):
  oldnom1   old without Monitor1 (guard 0, [1e-6, 1e6], schedule only)
  newfreeze new, but sigma never changes after iteration 20,000: the schedule check runs only for k <= 20,000, so the
            last possible change is at k = 19,001 (the new sigma is used from k = 19,002 on)
  newdb     new plus a magnitude dead-band D = 10 on the schedule decision. Every iteration i adds
            log(max(errRp_i, 1e-300)) - log(max(errRd_i, 1e-300)) to a window sum; at every schedule check k the window
            is W = (k', k] with k' the previous check (k' = 0 for k = 1), and rho_W = exp(mean of the window sum) is the
            geometric mean of errRp/errRd over W. A change chosen by the counter test of `new` is applied only if
            rho_W < 1/D for an increase (primal residual smaller: the primal side "wins") or rho_W > D for a decrease;
            otherwise sigma AND the counters are left unchanged (as when the counter test fails), and the blocked attempt
            is counted (deadband_blocked_up / _down). The window sum is reset at every check, changed or not.

Changes in v1.1 (2026-10-19, rule comparison; old / new / fixed trajectories are unchanged: ../emu-run/test_emu_rules.py):
  - the variants above (SigmaRule.freeze_after, SigmaRule.deadband; RULES 'oldnom1', 'newfreeze', 'newdb');
  - bug fix: validation_history time_s lagged the CPU clock by up to --checkpoint-every seconds, because the iteration
    time of the current segment (kept in a local accumulator) was added to cpu_iter_s only at checkpoints; time_s now
    includes it (only the time_s column changes; no iterate, sigma or flag depends on it);
  - --gzip-validation writes validation_history.csv.gz instead of validation_history.csv;
  - every CLI segment (run / resume) records host, CPU model, iterations and seconds in summary.json "segments".

What is exact and what is not (validation against the GPU logs: validation_table.csv in this directory):
  - every elementwise operation, scalar factor and the order of the operations inside one iteration are those of
    solve(); the sigma rules, the counters and Monitor1 are bit-exact functions of the residual sequence;
  - not bit-exact: reductions (cuBLAS nrm2, cuSPARSE SpMV/SpVV summation order), FMA contraction by nvcc, the
    y-step solve (CHOLMOD simplicial LDL^T with a fill-reducing permutation vs. a dense Cholesky here), and the
    eigensolvers (cuSOLVER syevjBatched (n <= 32) / syevd (32 < n <= 1000) vs. LAPACK dsyevd here). These give
    relative differences of about 1e-16 to 1e-14 per iteration, which the iteration carries forward.
  - blocks larger than 1000 (LOBPCG / large-matrix path of the GPU code) are projected with a full eigh here.

Anatomy of iteration `iter` = k (1, 2, ...; the C++ loop at solver.cu:1280 runs iter = 1 .. max_iter + 1):
  Step 0 (1285-1400)  termination / validation of the iterate after done = iter - 1 steps, with the CURRENT sigma (the
                      one set at the end of iteration iter - 1); triggers: every `interval` completed iterations, a
                      downward crossing of stop_tol by max(errRp, errRd, relgap), the cap; stop only on a validated
                      snapshot (strict stage: validated and max_abs_DIMACS <= strict_dimacs_tol). The internal residual
                      alone never stops a validating solve; the practical test is ExternalValidator's `validated`
                      (eta_p, eta_d, eta_g, X / dual / S cone violations <= tol), max_abs_DIMACS is used only by the
                      strict stage.
  y-step (1622-1637)  y = (AA^T + 1e-15 I)^{-1} (Rp / sigma - A (S - C)), Rp = b - A X and S of the previous iteration
  (1648-1660)         Xinput = -(A^T y - C + X (1/sigma))
  (1665-1863)         S = Pi(Xinput): PSD blocks by eigendecomposition, 'l' max(., 0), 'u' 0
  (1931-1949)         Rd = A^T y - C + S; X += tau sigma Rd, tau = 1.618 (the errRd < tol reduction max(1.618, tau/1.1)
                      is a no-op in plain ADMM; every GPU history has tau = 1.618)
  (1954-1975)         Rp = b - A X; errRp, errRd, pobj, dobj, relgap of the NEW iterate (original units)
  (2001-2005)         feasratio = errRp / errRd of the new iterate; prim_win += 1 if feasratio < 1 else dual_win += 1
                      (the counters are incremented BEFORE the check of the same iteration, so at k = 1 one of them is 1)
  (2049-2073)         if k is a schedule iteration (k <= 200: k % 10 == 1; <= 1000: % 25; <= 5000: % 50; <= 10000: % 200;
                      else % 1000 == 1): factor sigscale = max(2 exp(-k/5e4), 2) = 2 exactly; the winner's counter is
                      reset (the loser's is NOT); the new sigma is used from the y-step of iteration k + 1
  (9bb4ba5:2076-2083) old rule only: if k > 5000 and k % 25 == 0, Monitor1.push(pobj, dobj, errRp, errRd, relgap) and,
                      once its 20-entry queues are full, sigma = chase_update_sig(sigma) (100 or 0.01 or unchanged; the
                      queues are reset whenever a chase pattern is detected, even if sigma does not change)

Usage (CLI):
  python cuadmm_emu.py run <problem_dir> --rule old --sig 1 --out <dir> [--max-iter N] [--pause-at N]
                           [--wall-limit S] [--validate-interval 50] [--tol 1e-4] [--strict-dimacs-tol 1e-6]
                           [--no-stop] [--history-stride K] [--history-until N]
  python cuadmm_emu.py resume <dir> [--pause-at N] [--wall-limit S] [--max-iter N]
Python:
  import cuadmm_emu as ce
  p = ce.Problem('/n/home00/yukuanwei/cuadmm-data/plato_kocvara/txt/buck1')
  e = ce.Emulator(p, rule='old', sig0=1.0, validate_interval=50, strict_dimacs_tol=1e-6)
  e.run(pause_at=1000)          # runs iterations 1..1000, then returns (state kept; call run() again to continue)
  e.sigma_log, e.validations    # lists of dicts with the GPU CSV columns
  e.save_state('state.npz'); e2 = ce.Emulator.load_state('state.npz')
"""
import os

# single-threaded BLAS/LAPACK: deterministic reductions, and safe on shared nodes (set before numpy is imported)
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import collections
import csv
import json
import math
import re
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

__version__ = '1.2 (2026-10-21, cuADMM-reorg Phase 4b: balance_freeze)'


def _sqrt_newton(x, curr, prev):
    # include/cuadmm/kernels.h:159-165: constexpr Newton iteration (gives 1.414213562373095, one ulp below sqrt(2))
    while curr != prev:
        curr, prev = 0.5 * (curr + x / curr), curr
    return curr


SQRT2 = _sqrt_newton(2.0, 2.0, 0.0)  # kernels.h:167
SQRT2INV = 1.0 / SQRT2  # kernels.h:168
VAL_IS2 = 1.0 / math.sqrt(2.0)  # external_validation.cu:44 (svec -> matrix in the validator)
SMALL_MAT_LIMIT, MEDIUM_MAT_LIMIT = 32, 1000  # matrix_sizes.h


# --------------------------------------------------------------------------------------------------------------------
# Problem input (src/problem.cu, src/utils/io.cu)
# --------------------------------------------------------------------------------------------------------------------
_BLK_TV = re.compile(r'\s*([a-zA-Z])\s+(-?\d+)\s*')
_BLK_V = re.compile(r'\s*(-?\d+)\s*')


def read_blk(path):
    """io.cu read_blk: lines "s 10" / "l 5" / "u 3", or "10" (= "s 10"); other lines are ignored."""
    out = []
    with open(path) as f:
        for line in f.read().split('\n'):
            m = _BLK_TV.fullmatch(line)
            if m:
                out.append((m.group(1), int(m.group(2))))
                continue
            m = _BLK_V.fullmatch(line)
            if m:
                out.append(('s', int(m.group(1))))
    return out


def _read_triplets(path):
    """`while (file >> row >> col >> val)`: whitespace-separated triplets (int, int, double)."""
    tok = open(path).read().split()
    k = len(tok) // 3
    rows = np.array(tok[0:3 * k:3], dtype=np.int64)
    cols = np.array(tok[1:3 * k:3], dtype=np.int64)
    vals = np.array([float(t) for t in tok[2:3 * k:3]], dtype=np.float64)
    return rows, cols, vals


def nrm2(v):
    # cublasDnrm2 (summation order not replicated)
    return math.sqrt(float(np.dot(v, v)))


def _div(a, b):
    # IEEE division of two doubles (Python raises on /0)
    if b == 0.0:
        if a == 0.0 or a != a:
            return float('nan')
        return math.copysign(float('inf'), a) * math.copysign(1.0, b)
    return a / b


class Problem:
    """Solver input in the cuADMM text format (Problem::from_txt).

    At.txt: "svec_index constraint value" (At is vec_len x m, CSC by constraint after COO_to_CSC);
    b.txt: "i 0 value"; C.txt: "svec_index 0 value"; blk.txt; con_num.txt; optional dimacs_blocks.txt.
    svec: per PSD block, the upper triangle column by column, (1,1), (1,2), (2,2), (1,3), ..., the off-diagonal
    entries multiplied by sqrt(2) (block_structure.cu:240-256, vec_mat_conversion.cu)."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        d = self.path
        self.name = os.path.basename(os.path.normpath(d))
        self.blk = read_blk(os.path.join(d, 'blk.txt'))
        if not self.blk:
            raise ValueError('empty blk.txt')
        for t, n in self.blk:
            if t not in 'sul' or n <= 0:
                raise ValueError(f'bad block {t} {n}')
        # con_num.txt is read as a double and stored in a vector<int> (io.cu read_dense_vector_data)
        self.m = int(float(open(os.path.join(d, 'con_num.txt')).read().split()[0]))
        self.offsets, off = [], 0
        for t, n in self.blk:
            self.offsets.append(off)
            off += n * (n + 1) // 2 if t == 's' else n
        self.n = off  # vec_len
        rows, cols, vals = _read_triplets(os.path.join(d, 'At.txt'))
        if rows.size and (rows.min() < 0 or rows.max() >= self.n or cols.min() < 0 or cols.max() >= self.m):
            raise ValueError('At.txt index out of range')
        # COO_to_CSC: sort by (constraint, svec index); duplicates are kept as separate entries (std::sort is not
        # stable, so the order among duplicates is unspecified; the PLATO data have none, see meta.json)
        order = np.lexsort((rows, cols))
        self.At_row = rows[order]
        self.At_col = cols[order]
        self.At_val = vals[order]
        self.At_colptr = np.zeros(self.m + 1, dtype=np.int64)
        np.add.at(self.At_colptr, self.At_col + 1, 1)
        self.At_colptr = np.cumsum(self.At_colptr)
        self.has_duplicates = len(set(zip(self.At_row.tolist(), self.At_col.tolist()))) != self.At_row.size
        bi, bc, bv = _read_triplets(os.path.join(d, 'b.txt'))
        ci, cc, cv = _read_triplets(os.path.join(d, 'C.txt'))
        if (bi.size and (bi.min() < 0 or bi.max() >= self.m)) or (ci.size and (ci.min() < 0 or ci.max() >= self.n)):
            raise ValueError('b.txt / C.txt index out of range')
        self.b_idx, self.b_val = bi, bv  # sparse vectors, file order (DeviceSparseVector)
        self.C_idx, self.C_val = ci, cv
        if len(set(bi.tolist())) != bi.size or len(set(ci.tolist())) != ci.size:
            # cusparseAxpby / SpVV with repeated indices are not replicated
            raise ValueError('repeated indices in b.txt or C.txt are not supported')
        kp = os.path.join(d, 'dimacs_blocks.txt')
        if os.path.exists(kp):
            kinds = [l.split()[0] for l in open(kp) if l.split()]
            if len(kinds) != len(self.blk):
                raise ValueError('dimacs_blocks.txt does not match blk.txt')
            self.dimacs_kinds = kinds
        else:
            self.dimacs_kinds = ['s' if t == 's' else 'l' for t, _ in self.blk]
        # dense copies of the original b and C (ExternalValidator: b_[idx] += val)
        self.b_dense = np.zeros(self.m)
        np.add.at(self.b_dense, self.b_idx, self.b_val)
        self.C_dense = np.zeros(self.n)
        np.add.at(self.C_dense, self.C_idx, self.C_val)

    def summary(self):
        return dict(name=self.name, m=self.m, vec_len=self.n, nnz_At=int(self.At_val.size), blocks=self.blk)


# --------------------------------------------------------------------------------------------------------------------
# External validation (src/external_validation.cu at 2d8a755, identical at 9bb4ba5)
# --------------------------------------------------------------------------------------------------------------------
def _ld_sumsq(v):
    a = np.asarray(v, dtype=np.longdouble)
    return np.sum(a * a)


def _ld_dot(a, b):
    return np.sum(np.asarray(a, dtype=np.longdouble) * np.asarray(b, dtype=np.longdouble))


def _ld_norm2(v):
    # norm2(): long double accumulation of the squares, sqrt in double
    return math.sqrt(float(_ld_sumsq(v)))


def _svec_min_eig(v, off, n):
    # svec_min_eig: matrix from the svec (off-diagonals * 1/sqrt(2)), LAPACK dsyevd jobz='N', uplo='U'
    L = n * (n + 1) // 2
    seg = np.asarray(v[off:off + L], dtype=np.float64)
    ii, jj = _tri_index(n)  # column ii, row jj <= ii
    M = np.zeros((n, n))
    x = np.where(ii == jj, seg, seg * VAL_IS2)
    M[jj, ii] = x
    M[ii, jj] = x
    try:
        w = sla.eigh(M, lower=False, eigvals_only=True, driver='evd', check_finite=False)
    except Exception:
        return float('nan')
    return float(w[0])


_TRI_CACHE = {}


def _tri_index(n):
    t = _TRI_CACHE.get(n)
    if t is None:
        ii = np.repeat(np.arange(n), np.arange(1, n + 1))  # column index i
        jj = np.concatenate([np.arange(i + 1) for i in range(n)])  # row index j <= i
        t = (ii, jj)
        _TRI_CACHE[n] = t
    return t


class ExternalValidator:
    """ExternalValidator::evaluate on the ORIGINAL data and the UNSCALED iterate."""

    def __init__(self, prob):
        self.p = prob
        p = prob
        # A (m x n) with the original values, for AX and A^T y (the C++ loops over the CSC of At)
        self.A = sp.csr_matrix((p.At_val, p.At_row, p.At_colptr), shape=(p.m, p.n))
        self.At = self.A.T.tocsr()
        self.b = p.b_dense
        self.C = p.C_dense
        self.norm_b = _ld_norm2(self.b)
        self.norm_C = _ld_norm2(self.C)
        self.norm_b_inf = float(np.max(np.abs(self.b))) if self.b.size else 0.0
        cinf = 0.0
        for (t, n), off in zip(p.blk, p.offsets):
            if t == 's':
                ii, jj = _tri_index(n)
                seg = self.C[off:off + n * (n + 1) // 2]
                vals = np.where(ii == jj, np.abs(seg), np.abs(seg / math.sqrt(2.0)))
                if vals.size:
                    cinf = max(cinf, float(vals.max()))
            else:
                if n:
                    cinf = max(cinf, float(np.max(np.abs(self.C[off:off + n]))))
        self.norm_C_inf = cinf

    def evaluate(self, X, y, S, tol=1e-4):
        p = self.p
        nan = float('nan')
        r = dict(finite=bool(np.all(np.isfinite(X)) and np.all(np.isfinite(S)) and np.all(np.isfinite(y))))
        if not r['finite']:
            r.update(validated=False, failed='non-finite')
            return r
        AX = self.A @ X
        Aty = self.At @ y
        rp = AX - self.b
        rd = Aty + S - self.C
        Z = self.C - Aty
        pobj_ld = _ld_dot(self.C, X)
        dobj_ld = _ld_dot(self.b, y)
        pobj, dobj = float(pobj_ld), float(dobj_ld)
        r['pobj'], r['dobj'] = pobj, dobj
        r['primal_res'] = _ld_norm2(rp) / (1 + self.norm_b)
        r['dual_res'] = _ld_norm2(rd) / (1 + self.norm_C)
        r['relgap'] = abs(float(pobj_ld - dobj_ld)) / (1 + abs(pobj) + abs(dobj))
        r['kkt_full'] = max(r['primal_res'], max(r['dual_res'], r['relgap']))
        nb = len(p.blk)
        lmin_X, lmin_Z, lmin_S, frob_X = [0.0] * nb, [0.0] * nb, [0.0] * nb, [0.0] * nb
        eig_ok = [True] * nb
        for k, ((t, n), off) in enumerate(zip(p.blk, p.offsets)):
            if t == 's':
                L = n * (n + 1) // 2
                frob_X[k] = math.sqrt(float(_ld_sumsq(X[off:off + L])))
                lmin_X[k] = _svec_min_eig(X, off, n)
                lmin_Z[k] = _svec_min_eig(Z, off, n)
                lmin_S[k] = _svec_min_eig(S, off, n)
                eig_ok[k] = all(math.isfinite(v) for v in (lmin_X[k], lmin_Z[k], lmin_S[k]))
            elif t == 'l':
                frob_X[k] = math.sqrt(float(_ld_sumsq(X[off:off + n])))
                lmin_X[k] = float(np.min(X[off:off + n]))
                lmin_Z[k] = float(np.min(Z[off:off + n]))
                lmin_S[k] = float(np.min(S[off:off + n]))
        worst_X = worst_X_normb = worst_Z = worst_S = 0.0
        min_X = min_Z = min_S = float('inf')
        all_eig_ok = True
        neg_blocks, worst_blk, cert = 0, -1, 0.0
        for k, (t, n) in enumerate(p.blk):
            if t == 'u':
                continue
            all_eig_ok = all_eig_ok and eig_ok[k]
            min_X = min(min_X, lmin_X[k])
            min_Z = min(min_Z, lmin_Z[k])
            min_S = min(min_S, lmin_S[k])
            worst_S = max(worst_S, max(0.0, -lmin_S[k]) / (1 + self.norm_C))
            if lmin_X[k] < 0:
                neg_blocks += 1
            vX = max(0.0, -lmin_X[k]) / (1 + frob_X[k])
            if vX > worst_X:
                worst_X, worst_blk = vX, k
            worst_X_normb = max(worst_X_normb, max(0.0, -lmin_X[k]) / (1 + self.norm_b))
            worst_Z = max(worst_Z, max(0.0, -lmin_Z[k]) / (1 + self.norm_C))
            if t == 's':
                cert += n * min(0.0, lmin_Z[k])
            else:
                off = p.offsets[k]
                cert += float(np.sum(np.minimum(0.0, Z[off:off + n])))
        r['X_cone_violation'] = worst_X if all_eig_ok else nan
        r['X_cone_violation_normb'] = worst_X_normb if all_eig_ok else nan
        r['dual_cone_violation'] = worst_Z if all_eig_ok else nan
        r['S_cone_violation'] = worst_S if all_eig_ok else nan
        r['min_eig_X'], r['min_eig_Z'], r['min_eig_S'] = min_X, min_Z, min_S
        r['X_negative_blocks'] = neg_blocks
        r['X_worst_block'] = worst_blk
        # DIMACS errors (z = S) with the norms of the original block kinds
        lin2, psd_sum = np.longdouble(0), 0.0
        for k, ((t, n), off) in enumerate(zip(p.blk, p.offsets)):
            L = n * (n + 1) // 2 if t == 's' else n
            bl = _ld_sumsq(rd[off:off + L])
            if p.dimacs_kinds[k] == 's':
                psd_sum += math.sqrt(float(bl))
            else:
                lin2 += bl
        xs = _ld_dot(X, S)
        xz = _ld_dot(X, Z)
        rp2 = _ld_norm2(rp)
        rdK = psd_sum + math.sqrt(float(lin2))
        den = 1 + abs(pobj) + abs(dobj)
        e = [rp2 / (1 + self.norm_b_inf),
             max(0.0, -min_X) / (1 + self.norm_b_inf) if all_eig_ok else nan,
             rdK / (1 + self.norm_C_inf),
             max(0.0, -min_S) / (1 + self.norm_C_inf) if all_eig_ok else nan,
             float(pobj_ld - dobj_ld) / den,
             float(xs) / den]
        r['dimacs'] = e
        r['err4_z'] = max(0.0, -min_Z) / (1 + self.norm_C_inf) if all_eig_ok else nan
        r['err6_z'] = float(xz) / den
        r['err6_defined'] = e[1] == 0.0 and e[3] == 0.0
        mx = 0.0
        for v in e:
            mx = max(mx, abs(v)) if (math.isfinite(v) and math.isfinite(mx)) else nan
        r['max_abs_dimacs'] = mx
        r['certificate_lower_bound_conditional'] = dobj + cert
        failed = []
        le = lambda x, tl: x <= tl  # False for NaN (fail closed)
        for ok, name in ((le(r['primal_res'], tol), 'primal'), (le(r['dual_res'], tol), 'dual'), (le(r['relgap'], tol), 'gap'),
                         (le(r['X_cone_violation'], tol), 'X_cone'), (le(r['dual_cone_violation'], tol), 'dual_cone'),
                         (le(r['S_cone_violation'], tol), 'S_cone')):
            if not ok:
                failed.append(name)
        if not all_eig_ok:
            failed.append('eig')
        r['failed'] = ','.join(failed)
        r['validated'] = not failed
        return r


def merit(r):
    if not r.get('finite'):
        return float('inf')
    m = 0.0
    for k in ('primal_res', 'dual_res', 'relgap', 'X_cone_violation', 'dual_cone_violation', 'S_cone_violation'):
        x = r.get(k, float('nan'))
        if not math.isfinite(x):
            return float('inf')
        m = max(m, x)
    return m


# --------------------------------------------------------------------------------------------------------------------
# Sigma rules
# --------------------------------------------------------------------------------------------------------------------
class SigmaRule:
    """Plain-ADMM sigma update of solve() (solver.cu:2000-2085).

    guard: the winner's count must exceed max(guard, ratio * other) (0 at 9bb4ba5, 1 at 2d8a755).
    freeze_after: if set, the schedule check runs only for k <= freeze_after (no change afterwards; not in cuADMM).
    deadband: if set (D > 1), a change chosen by the counter test needs the window geometric mean of errRp/errRd to be
              < 1/D (increase) or > D (decrease); see the module docstring (not in cuADMM)."""

    def __init__(self, name, adaptive=True, guard=0.0, ratio=1.35, sigmin=1e-6, sigmax=1e6, monitor1=True, freeze_after=None,
                 deadband=None, freeze=False, factor=2.0, step_schedule='constant', freeze_iteration=10000, pingpong_window=8,
                 pingpong_net=2, pingpong_min_iter=1000, freeze_value='band_centre', freeze_bias=1.0, clamp_sigma0=None):
        self.name, self.adaptive, self.guard, self.ratio = name, adaptive, float(guard), float(ratio)
        self.sigmin, self.sigmax, self.monitor1 = float(sigmin), float(sigmax), bool(monitor1)
        self.freeze_after = None if freeze_after is None else int(freeze_after)
        self.deadband = None if deadband is None else float(deadband)
        # v1.2: balance_freeze (cuADMM-reorg src/admm/penalty_policy.cc, ResidualBalancingPenalty with freeze enabled)
        self.freeze = bool(freeze)
        self.factor = float(factor)
        self.step_schedule = str(step_schedule)
        self.freeze_iteration = int(freeze_iteration) if freeze_iteration is not None and int(freeze_iteration) >= 0 else 10000
        self.pingpong_window = int(pingpong_window)
        self.pingpong_net = int(pingpong_net)
        self.pingpong_min_iter = int(pingpong_min_iter)
        self.freeze_value = str(freeze_value)
        self.freeze_bias = float(freeze_bias)
        self.clamp_sigma0 = bool(freeze) if clamp_sigma0 is None else bool(clamp_sigma0)
        if self.step_schedule not in ('constant', 'halve_on_reversal') or self.freeze_value not in ('last', 'band_centre'):
            raise ValueError('unknown step_schedule or freeze_value')

    def to_dict(self):
        d = dict(name=self.name, adaptive=self.adaptive, guard=self.guard, ratio=self.ratio, sigmin=self.sigmin,
                 sigmax=self.sigmax, monitor1=self.monitor1, freeze_after=self.freeze_after, deadband=self.deadband)
        if self.freeze:
            d.update(freeze=True, factor=self.factor, step_schedule=self.step_schedule, freeze_iteration=self.freeze_iteration,
                     pingpong_window=self.pingpong_window, pingpong_net=self.pingpong_net, pingpong_min_iter=self.pingpong_min_iter,
                     freeze_value=self.freeze_value, freeze_bias=self.freeze_bias, clamp_sigma0=self.clamp_sigma0)
        return d

    @staticmethod
    def from_dict(d):
        return SigmaRule(**d)  # v1.0 checkpoints have no freeze_after / deadband: the defaults (None) apply


RULES = {
    'old': lambda: SigmaRule('old', True, 0.0, 1.35, 1e-6, 1e6, True),     # 9bb4ba5 (solver.cu:1015-1016, 2052-2083)
    'new': lambda: SigmaRule('new', True, 1.0, 1.35, 1e-3, 1e3, False),    # 2d8a755 (solver.cu:1015-1016, 2052-2085)
    'fixed': lambda: SigmaRule('fixed', False, 0.0, 1.35, 1e-3, 1e3, False),  # SigmaPolicy::FIXED
    # variants for the rule comparison (not in any cuADMM commit)
    'oldnom1': lambda: SigmaRule('oldnom1', True, 0.0, 1.35, 1e-6, 1e6, False),
    'newfreeze': lambda: SigmaRule('newfreeze', True, 1.0, 1.35, 1e-3, 1e3, False, freeze_after=20000),
    'newdb': lambda: SigmaRule('newdb', True, 1.0, 1.35, 1e-3, 1e3, False, deadband=10.0),
}
DB_TINY = 1e-300  # floor of errRp / errRd inside the dead-band logarithms


def balance_freeze_rule(name='balance_freeze', factor=2.0, step_schedule='constant', freeze_iteration=10000, pingpong_window=8,
                        pingpong_net=2, pingpong_min_iter=1000, freeze_value='band_centre', freeze_bias=1.0, sigmin=1e-3,
                        sigmax=1e3, ratio=1.35, guard=1.0):
    """The planned rule of cuADMM-reorg (plan Section 3.5): residual balancing with the guard, then a freeze."""
    return SigmaRule(name, True, guard, ratio, sigmin, sigmax, False, freeze=True, factor=factor, step_schedule=step_schedule,
                     freeze_iteration=freeze_iteration, pingpong_window=pingpong_window, pingpong_net=pingpong_net,
                     pingpong_min_iter=pingpong_min_iter, freeze_value=freeze_value, freeze_bias=freeze_bias)


RULES['balance_freeze'] = lambda: balance_freeze_rule()


class BalanceFreezeState:
    """The state and the update of balance_freeze, one call per iteration (the counters, the freeze, the sigma log); a
    transcription of ResidualBalancingPenalty (freeze enabled) of cuADMM-reorg src/admm/penalty_policy.cc, so that the
    C++ policy and this replica can be checked against each other on scripted sequences."""

    def __init__(self, rule, sig0):
        self.rule = rule
        self.sig = min(max(float(sig0), rule.sigmin), rule.sigmax) if rule.clamp_sigma0 else float(sig0)
        self.prim_win = 0
        self.dual_win = 0
        self.frozen = False
        self.freeze_at = -1
        self.freeze_reason = ''
        self.step = rule.factor
        self.dirs = []
        self.sigmas = []
        self.log = []  # dicts with the GPU sigma_log columns

    def state(self):
        return dict(sig=self.sig, prim_win=self.prim_win, dual_win=self.dual_win, frozen=self.frozen, freeze_at=self.freeze_at,
                    freeze_reason=self.freeze_reason, step=self.step, dirs=list(self.dirs), sigmas=list(self.sigmas))

    def set_state(self, d):
        self.sig, self.prim_win, self.dual_win = float(d['sig']), int(d['prim_win']), int(d['dual_win'])
        self.frozen, self.freeze_at, self.freeze_reason = bool(d['frozen']), int(d['freeze_at']), str(d['freeze_reason'])
        self.step, self.dirs, self.sigmas = float(d['step']), [int(x) for x in d['dirs']], [float(x) for x in d['sigmas']]

    def _record(self, k, old, new, rule_name, errRp, errRd, relgap, feasratio, pw, dw, always=False):
        if always or new != old:
            self.log.append(dict(iter=k, old_sigma=old, new_sigma=new, rule=rule_name, errRp=errRp, errRd=errRd, relgap=relgap,
                                 feasratio=feasratio, prim_win=pw, dual_win=dw))

    def _change(self, k, d, errRp, errRd, relgap, feasratio, pw, dw):
        r = self.rule
        before = self.sig
        if r.step_schedule == 'halve_on_reversal' and self.dirs and d == -self.dirs[-1]:
            self.step = math.sqrt(self.step)
        moved = self.sig * self.step if d > 0 else self.sig / self.step
        self.sig = min(max(moved, r.sigmin), r.sigmax)
        self.dirs.append(d)
        self.sigmas.append(self.sig)
        self._record(k, before, self.sig, 'admm_schedule', errRp, errRd, relgap, feasratio, pw, dw)
        if r.step_schedule == 'halve_on_reversal' and math.log2(self.step) < 1.0 / 16.0:
            self._freeze(k, 'step', errRp, errRd, relgap, feasratio, pw, dw)

    def _freeze(self, k, reason, errRp, errRd, relgap, feasratio, pw, dw):
        if self.frozen:
            return
        r = self.rule
        before = self.sig
        W = r.pingpong_window
        if len(self.dirs) >= W:
            net = sum(self.dirs[-W:])
            if abs(net) <= r.pingpong_net and r.freeze_value == 'band_centre':
                log_sum = 0.0
                for v in self.sigmas[-W:]:
                    log_sum += math.log(v)
                self.sig = math.exp(log_sum / W)
        self.sig = min(max(r.freeze_bias * self.sig, r.sigmin), r.sigmax)
        self.frozen = True
        self.freeze_at = k
        self.freeze_reason = reason
        self._record(k, before, self.sig, 'freeze_' + reason, errRp, errRd, relgap, feasratio, pw, dw, always=True)

    def update(self, k, errRp, errRd, relgap, ratioconst=1.0):
        """sigma for iteration k + 1 from the residuals of iteration k (both adaptive phases: plain ADMM only here)."""
        r = self.rule
        if self.frozen:
            return self.sig
        feasratio = _div(ratioconst * errRp, errRd)
        if feasratio < 1:
            self.prim_win += 1
        else:
            self.dual_win += 1
        pw, dw = self.prim_win, self.dual_win
        if schedule_check(k):
            thr_p = max(1.0, r.ratio * self.dual_win) if r.guard >= 1.0 else r.ratio * self.dual_win
            thr_d = max(1.0, r.ratio * self.prim_win) if r.guard >= 1.0 else r.ratio * self.prim_win
            if self.prim_win > thr_p:
                self.prim_win = 0
                self._change(k, +1, errRp, errRd, relgap, feasratio, pw, dw)
            elif self.dual_win > thr_d:
                self.dual_win = 0
                self._change(k, -1, errRp, errRd, relgap, feasratio, pw, dw)
        if not self.frozen:
            if r.freeze_iteration >= 0 and k >= r.freeze_iteration:
                self._freeze(k, 'freeze_iteration', errRp, errRd, relgap, feasratio, pw, dw)
            elif k >= r.pingpong_min_iter and len(self.dirs) >= r.pingpong_window:
                net = sum(self.dirs[-r.pingpong_window:])
                if abs(net) <= r.pingpong_net:
                    self._freeze(k, 'ping_pong', errRp, errRd, relgap, feasratio, pw, dw)
        return self.sig


def schedule_check(k):
    # solver.cu:2052-2057
    return ((k <= 200 and k % 10 == 1) or (200 < k <= 1000 and k % 25 == 1) or (1000 < k <= 5000 and k % 50 == 1)
            or (5000 < k <= 10000 and k % 200 == 1) or (k > 10000 and k % 1000 == 1))


class SlidingDiffQueue:
    """include/cuadmm/monitors.h SlidingDiffQueue."""

    def __init__(self, cap):
        self.cap = cap
        self.q1 = collections.deque()
        self.dq = collections.deque()

    def push(self, v):
        if self.q1:
            self.dq.append(v - self.q1[-1])
        self.q1.append(v)
        if len(self.q1) > self.cap:
            self.q1.popleft()
            if self.dq:
                self.dq.popleft()

    def full(self):
        return len(self.q1) == self.cap

    def empty(self):
        return not self.q1

    def reset(self):
        self.q1.clear()
        self.dq.clear()


class Monitor1:
    """include/cuadmm/monitors.h Monitor1 (capacity 20, update_interval 25)."""

    def __init__(self, cap=20):
        self.cap = cap
        self.update_interval = 25
        self.pobj, self.dobj = SlidingDiffQueue(cap), SlidingDiffQueue(cap)
        self.obj_diff, self.ratio = SlidingDiffQueue(cap), SlidingDiffQueue(cap)
        self.last_reason = ''

    def reset(self):
        for q in (self.pobj, self.dobj, self.ratio, self.obj_diff):
            q.reset()

    def push(self, pobj, dobj, pinf, dinf, relgap):
        self.pobj.push(pobj)
        self.dobj.push(dobj)
        self.obj_diff.push(pobj - dobj)
        self.ratio.push(relgap / (1e-16 + max(pinf, dinf)))

    def full(self):
        return self.pobj.full()

    def chase_update_sig(self, sig):
        self.last_reason = ''
        if self.pobj.empty():
            return sig
        if not all(x > 1.2 for x in self.ratio.q1):
            return sig
        pa = all(x > 0.0 for x in self.pobj.dq)
        pd = all(x < 0.0 for x in self.pobj.dq)
        da = all(x > 0.0 for x in self.dobj.dq)
        dd = all(x < 0.0 for x in self.dobj.dq)
        op = all(x > 0.0 for x in self.obj_diff.q1)
        on = all(x < 0.0 for x in self.obj_diff.q1)
        help_dual = max(1e2, sig)
        help_primal = min(1e-2, sig)
        if pa and da:
            if op:
                self.reset()
                self.last_reason = 'ascend,pobj>dobj,help_dual'
                return help_dual
            elif on:
                self.reset()
                self.last_reason = 'ascend,pobj<dobj,help_primal'
                return help_primal
        if pd and dd:
            if op:
                self.reset()
                self.last_reason = 'descend,pobj>dobj,help_primal'
                return help_primal
            elif on:
                self.reset()
                self.last_reason = 'descend,pobj<dobj,help_dual'
                return help_dual
        return sig

    def state(self):
        return {k: [list(q.q1), list(q.dq)] for k, q in
                (('pobj', self.pobj), ('dobj', self.dobj), ('obj_diff', self.obj_diff), ('ratio', self.ratio))}

    def set_state(self, st):
        for k, q in (('pobj', self.pobj), ('dobj', self.dobj), ('obj_diff', self.obj_diff), ('ratio', self.ratio)):
            q.q1 = collections.deque(float(x) for x in st[k][0])
            q.dq = collections.deque(float(x) for x in st[k][1])


# --------------------------------------------------------------------------------------------------------------------
# The solver replica
# --------------------------------------------------------------------------------------------------------------------
class History:
    """Per-iteration history (iter, errRp, errRd, relgap, pobj, dobj, sig, prim_win, dual_win) in preallocated blocks."""
    NCOL = 9
    COLS = ('iter', 'errRp', 'errRd', 'relgap', 'pobj', 'dobj', 'sig', 'prim_win', 'dual_win')

    def __init__(self, block=1 << 16):
        self.block = block
        self.done = []  # full blocks
        self.cur = np.empty((block, self.NCOL))
        self.k = 0

    def append(self, row):
        self.cur[self.k] = row
        self.k += 1
        if self.k == self.block:
            self.done.append(self.cur)
            self.cur = np.empty((self.block, self.NCOL))
            self.k = 0

    def __len__(self):
        return len(self.done) * self.block + self.k

    def array(self):
        return np.concatenate(self.done + [self.cur[:self.k]]) if (self.done or self.k) else np.empty((0, self.NCOL))

    @staticmethod
    def from_array(a, block=1 << 16):
        h = History(block)
        a = np.asarray(a, dtype=np.float64).reshape(-1, History.NCOL)
        nfull = a.shape[0] // block
        h.done = [a[i * block:(i + 1) * block].copy() for i in range(nfull)]
        rest = a[nfull * block:]
        h.cur[:rest.shape[0]] = rest
        h.k = rest.shape[0]
        return h


SIGMA_LOG_COLUMNS = ['iter', 'old_sigma', 'new_sigma', 'rule', 'errRp', 'errRd', 'relgap', 'feasratio', 'prim_win', 'dual_win']
VALIDATION_COLUMNS = ['iter', 'time_s', 'sigma', 'trigger', 'checkpoint', 'internal_kkt', 'internal_errRp', 'internal_errRd',
                      'internal_relgap', 'primal_res', 'dual_res', 'relgap', 'kkt_full', 'pobj', 'dobj', 'X_cone_violation',
                      'X_cone_violation_normb', 'dual_cone_violation', 'min_eig_X', 'min_eig_Z', 'X_negative_blocks', 'finite',
                      'validated', 'failed', 'validation_s', 'callback_s', 'S_cone_violation', 'min_eig_S', 'dimacs1',
                      'dimacs2', 'dimacs3', 'dimacs4', 'dimacs5', 'dimacs6', 'max_abs_dimacs', 'err6_defined', 'err4_z', 'err6_z']
TAU_ADMM = 1.618  # solver.cu:1941


class Emulator:
    """One plain-ADMM solve (SDPSolver::init + solve with switch_admm = 0)."""

    def __init__(self, prob, rule='old', sig0=1.0, stop_tol=1e-4, max_iter=1000000000, validate_interval=50,
                 validate_tol=None, strict_dimacs_tol=0.0, stop_on_validation=True, history_stride=0, history_until=None,
                 X0=None, y0=None, S0=None, _skip_init=False):
        self.prob = prob if isinstance(prob, Problem) else Problem(prob)
        self.rule = RULES[rule]() if isinstance(rule, str) else rule
        self.sig0 = float(sig0)
        self.stop_tol = float(stop_tol)
        self.max_iter = int(max_iter)
        self.validate_interval = int(validate_interval)
        self.validate_tol = float(stop_tol if validate_tol is None else validate_tol)
        self.strict_dimacs_tol = float(strict_dimacs_tol)
        self.stop_on_validation = bool(stop_on_validation)
        self.history_stride = int(history_stride)
        self.history_until = int(history_until) if history_until is not None else -1  # -1: no limit
        self.warm = X0 is not None or y0 is not None or S0 is not None
        self._setup_operators()
        self.validator = ExternalValidator(self.prob) if self.validate_interval > 0 else None
        if not _skip_init:
            self._init_iterate(X0, y0, S0)

    # ---- init(): scaling and operators (solver.cu:533-746) ----
    def _setup_operators(self):
        p = self.prob
        m, n = p.m, p.n
        # get_normA (sparse_matrix_norm.cu): normA_i = max(1, ||column i of At||), sequential sum of squares, then
        # every entry of the column divided by it (in place, before the CSR copies and the CHOLMOD factorization)
        normA = np.empty(m)
        cp, vals = p.At_colptr, p.At_val
        for i in range(m):
            s = 0.0
            for v in vals[cp[i]:cp[i + 1]].tolist():
                s += v * v
            normA[i] = max(1.0, math.sqrt(s))
        self.normA = normA
        At_s = vals / normA[p.At_col]
        self.A = sp.csr_matrix((At_s, p.At_row, p.At_colptr), shape=(m, n))  # A_csr (= CSC of At)
        self.At = self.A.T.tocsr()  # At_csr
        # CHOLMOD on A A^T + 1e-15 I (solver.cu:547-551, cholesky_cpu.h get_A); dense Cholesky here
        AAt = (self.A @ self.A.T).toarray()
        AAt[np.diag_indices(m)] += 1e-15
        try:
            self.chol = sla.cho_factor(AAt, lower=True, check_finite=False)
            self.chol_kind = 'cholesky'
        except np.linalg.LinAlgError:
            self.chol = sla.lu_factor(AAt, check_finite=False)
            self.chol_kind = 'lu'
        # norms of the original b and C (solver.cu:689-690)
        self.norm_borg = 1 + nrm2(p.b_val)
        self.norm_Corg = 1 + nrm2(p.C_val)
        # b /= normA (sparse_vector_div_dense_vector), then b, C /= (1 + norm) as multiplications by 1/scale
        b_s = p.b_val / normA[p.b_idx]
        self.bscale = 1 + nrm2(b_s)
        self.Cscale = 1 + nrm2(p.C_val)
        self.objscale = self.bscale * self.Cscale
        self.b_s_val = b_s * (1 / self.bscale)
        self.C_s_val = p.C_val * (1 / self.Cscale)
        self.b_s = np.zeros(m)
        self.b_s[p.b_idx] = self.b_s_val
        self.C_s = np.zeros(n)
        self.C_s[p.C_idx] = self.C_s_val
        # cone structure: PSD blocks grouped by size (projection), 'l' and 'u' index sets
        groups = collections.OrderedDict()
        lidx, uidx = [], []
        for (t, bn), off in zip(p.blk, p.offsets):
            if t == 's':
                groups.setdefault(bn, []).append(off)
            elif t == 'l':
                lidx.extend(range(off, off + bn))
            else:
                uidx.extend(range(off, off + bn))
        self.psd_groups = []
        for bn, offs in sorted(groups.items()):
            ii, jj = _tri_index(bn)  # svec entry: column ii, row jj <= ii
            loc = ii * (ii + 1) // 2 + jj
            vidx = np.array([off + loc for off in offs])  # (nb, L)
            diag = ii == jj
            fac_in = np.where(diag, 1.0, SQRT2INV)  # vector_to_matrices: SQRT2INV + diag * (1 - SQRT2INV) (== 1 exactly)
            fac_out = np.where(diag, 1.0, SQRT2)  # matrices_to_vector: SQRT2 + diag * (1 - SQRT2) (== 1 exactly)
            cat = 'small' if bn <= SMALL_MAT_LIMIT else ('medium' if bn <= MEDIUM_MAT_LIMIT else 'large')
            self.psd_groups.append(dict(n=bn, nb=len(offs), vidx=vidx, ii=ii, jj=jj, fac_in=fac_in, fac_out=fac_out, cat=cat))
        self.lidx = np.array(lidx, dtype=np.int64)
        self.uidx = np.array(uidx, dtype=np.int64)

    def _init_iterate(self, X0, y0, S0):
        p = self.prob
        m, n = p.m, p.n
        X = np.zeros(n) if X0 is None else np.array(X0, dtype=np.float64)
        y = np.zeros(m) if y0 is None else np.array(y0, dtype=np.float64)
        S = np.zeros(n) if S0 is None else np.array(S0, dtype=np.float64)
        y = y * self.normA  # dense_vector_mul_dense_vector(y, normA)
        X = X * (1 / self.bscale)  # dense_vector_div_scalar: multiplication by 1/scale
        S = S * (1 / self.Cscale)
        y = y * (1 / self.Cscale)
        # initial residuals (solver.cu:710-746)
        Aty = self.At @ y
        Rp = self.b_s - (self.A @ X)
        SmC = S - self.C_s
        Rd = Aty + SmC
        self.X, self.y, self.S, self.Rp, self.SmC = X, y, S, Rp, SmC
        self.errRp = nrm2(self.normA * Rp * self.bscale) / self.norm_borg
        self.errRd = nrm2(Rd * self.Cscale) / self.norm_Corg
        self.maxfeas = max(self.errRp, self.errRd)
        self.pobj = float(np.dot(self.C_s_val, X[p.C_idx])) * self.objscale
        self.dobj = float(np.dot(self.b_s_val, y[p.b_idx])) * self.objscale
        self.relgap = abs(self.pobj - self.dobj) / (1 + abs(self.pobj) + abs(self.dobj))
        self.sig = self.sig0
        self.prim_win = 0
        self.dual_win = 0
        self.ratioconst = 1e0
        self.bf = BalanceFreezeState(self.rule, self.sig0) if self.rule.freeze else None  # v1.2
        if self.bf is not None:
            self.sig = self.bf.sig  # balance_freeze clamps sigma_0 into its bounds
        self.feasratio = float('nan')
        self.tau = TAU_ADMM
        self.monitor = Monitor1()
        # solve() bookkeeping
        self.iter = 1  # the next iteration to run (the C++ loop variable)
        self.stop_reason = ''
        self.finished = False
        self.nonfinite = False
        self.prev_internal_ok = False
        self.sigma_changes = 0
        self.sigma_log = []
        self.validations = []
        self.practical_validated = self.strict_validated = False
        self.first_practical_iteration = self.first_strict_iteration = -1
        self.first_internal_below_1e_3 = self.first_internal_below_tol = -1
        self.best_validation = None
        self.cpu_iter_s = 0.0  # CPU seconds in iterations (wall clock of the Python process)
        self.cpu_validation_s = 0.0
        self.iterations_done = 0
        self.history = History()  # every history_stride iterations, up to history_until
        self._init_v11_state()

    # v1.1 state (dead-band window, blocked attempts, output options); defaults also used for v1.0 checkpoints
    _V11_DEFAULTS = dict(db_sumlog=0.0, db_count=0, db_blocked_up=0, db_blocked_down=0, gzip_validation=False)

    def _init_v11_state(self):
        for k, v in self._V11_DEFAULTS.items():
            setattr(self, k, v)
        self.segments = []

    # ---- one external validation (validate_current_iterate, solver.cu:2121-2195) ----
    def _validate(self, done, trigger, t_pending=0.0):
        # t_pending: iteration seconds of the current run() segment not yet added to cpu_iter_s (v1.1 time_s fix)
        t0 = time.perf_counter()
        Xu = self.X * self.bscale
        Su = self.S * self.Cscale
        yu = self.y / self.normA * self.Cscale
        r = self.validator.evaluate(Xu, yu, Su, self.validate_tol)
        dt = time.perf_counter() - t0
        self.cpu_validation_s += dt
        rec = dict(iter=done, time_s=self.cpu_iter_s + t_pending + self.cpu_validation_s, sigma=self.sig, trigger=trigger, checkpoint=0,
                   internal_kkt=max(self.maxfeas, self.relgap), internal_errRp=self.errRp, internal_errRd=self.errRd,
                   internal_relgap=self.relgap, internal_pobj=self.pobj, internal_dobj=self.dobj, validation_s=dt, callback_s=0.0)
        for k in ('primal_res', 'dual_res', 'relgap', 'kkt_full', 'pobj', 'dobj', 'X_cone_violation', 'X_cone_violation_normb',
                  'dual_cone_violation', 'min_eig_X', 'min_eig_Z', 'X_negative_blocks', 'S_cone_violation', 'min_eig_S',
                  'max_abs_dimacs', 'err4_z', 'err6_z'):
            rec[k] = r.get(k, float('nan'))
        rec['finite'] = int(r['finite'])
        rec['validated'] = int(r['validated'])
        rec['failed'] = r['failed']
        rec['err6_defined'] = int(r.get('err6_defined', False))
        for i in range(6):
            rec[f'dimacs{i + 1}'] = r['dimacs'][i] if 'dimacs' in r else float('nan')
        rec['merit'] = merit(r)
        # best externally evaluated iterate (ValidationResult::better_than)
        b = self.best_validation
        if b is None or _better(rec, b):
            self.best_validation = rec
        self.validations.append(rec)
        return rec

    # ---- the solve loop (solver.cu:1280-2111) ----
    def run(self, pause_at=None, wall_limit=None, progress_every=0, checkpoint_path=None, checkpoint_every_s=None):
        """Run iterations until the solve ends (validated / max_iter / non-finite), until iteration `pause_at` is
        completed, or until `wall_limit` seconds have passed (both resumable). Returns the stop reason or 'paused'."""
        if self.finished:
            return self.stop_reason
        p = self.prob
        A, At, chol, chol_kind = self.A, self.At, self.chol, self.chol_kind
        b_s, C_s, normA = self.b_s, self.C_s, self.normA
        C_idx, b_idx, C_s_val, b_s_val = p.C_idx, p.b_idx, self.C_s_val, self.b_s_val
        bscale, Cscale, objscale = self.bscale, self.Cscale, self.objscale
        norm_borg, norm_Corg = self.norm_borg, self.norm_Corg
        groups, lidx, uidx = self.psd_groups, self.lidx, self.uidx
        has_l, has_u = lidx.size > 0, uidx.size > 0
        rule = self.rule
        validating = self.validate_interval > 0
        stop_tol = self.stop_tol
        X, y, S, Rp, SmC = self.X, self.y, self.S, self.Rp, self.SmC
        t_start = time.perf_counter()
        t_last_ckpt = t_start
        t_iter = 0.0
        status = None
        while True:
            it = self.iter
            done = it - 1
            # pause points come before Step 0, so that a resumed run executes Step 0 of iteration `it` exactly once
            if pause_at is not None and done >= pause_at:
                status = 'paused'
                break
            now = time.perf_counter()
            if wall_limit is not None and now - t_start > wall_limit:
                status = 'paused'
                break
            if checkpoint_path and checkpoint_every_s and now - t_last_ckpt > checkpoint_every_s:
                self.X, self.y, self.S, self.Rp, self.SmC = X, y, S, Rp, SmC
                self.cpu_iter_s += t_iter
                t_iter = 0.0
                self.save_state(checkpoint_path)
                t_last_ckpt = time.perf_counter()
            # ---------------- Step 0: termination checks and validation (solver.cu:1285-1400) ----------------
            internal_kkt = max(self.maxfeas, self.relgap)
            if done > 0 and internal_kkt < 1e-3 and self.first_internal_below_1e_3 < 0:
                self.first_internal_below_1e_3 = done
            if done > 0 and internal_kkt < stop_tol and self.first_internal_below_tol < 0:
                self.first_internal_below_tol = done
            breakyes = False
            if not validating:
                if internal_kkt < stop_tol:
                    breakyes, self.stop_reason = True, 'internal_kkt'
                if it > self.max_iter and not breakyes:
                    breakyes, self.stop_reason = True, 'max_iter'
            else:
                internal_ok = internal_kkt < stop_tol
                at_cap = it > self.max_iter
                trigger = ''
                if self.nonfinite:
                    if not self.validations:
                        raise RuntimeError(f'non-finite KKT residuals at iteration {done} before any external validation')
                    breakyes, self.stop_reason = True, 'non_finite'
                elif done > 0 or at_cap or internal_ok:
                    if at_cap:
                        trigger = 'max_iter'
                    elif internal_ok and not self.prev_internal_ok:
                        trigger = 'internal_crossing'
                    elif done > 0 and done % self.validate_interval == 0:
                        trigger = 'interval'
                self.prev_internal_ok = internal_ok
                if trigger:
                    self.X, self.y, self.S, self.Rp, self.SmC = X, y, S, Rp, SmC
                    rec = self._validate(done, trigger, t_iter)
                    if rec['validated']:
                        st = self.strict_dimacs_tol
                        if not self.practical_validated:
                            self.practical_validated = True
                            self.first_practical_iteration = done
                        strict_ok = st > 0 and math.isfinite(rec['max_abs_dimacs']) and rec['max_abs_dimacs'] <= st
                        if strict_ok and not self.strict_validated:
                            self.strict_validated = True
                            self.first_strict_iteration = done
                        if self.stop_on_validation and (st <= 0 or strict_ok):
                            breakyes = True
                            self.stop_reason = 'strict_validated' if strict_ok else 'externally_validated'
                if not breakyes and at_cap:
                    breakyes, self.stop_reason = True, 'max_iter'
            if breakyes:
                self.finished = True
                status = self.stop_reason
                break
            t0 = time.perf_counter()
            sig = self.sig
            # ---------------- Step 1: y = (AA^T + eps I)^{-1} (Rp / sig - A (S - C)) (solver.cu:1622-1637) ----------------
            rhsy = -(A @ SmC)
            rhsy += (1 / sig) * Rp
            if chol_kind == 'cholesky':
                y = sla.cho_solve(chol, rhsy, check_finite=False)
            else:
                y = sla.lu_solve(chol, rhsy, check_finite=False)
            # ---------------- Step 2: Xinput = -(A^T y - C + X / sig) (solver.cu:1648-1660) ----------------
            Rd1 = (At @ y) - C_s
            Xinput = -(Rd1 + X * (1.0 / sig))
            # ---------------- Step 3: S = Pi_K*(Xinput) (solver.cu:1665-1863) ----------------
            S = np.zeros_like(Xinput) if has_u else np.empty_like(Xinput)
            for g in groups:
                vidx, ii, jj, nn = g['vidx'], g['ii'], g['jj'], g['n']
                Mv = Xinput[vidx] * g['fac_in']
                M = np.zeros((g['nb'], nn, nn))
                M[:, jj, ii] = Mv
                M[:, ii, jj] = Mv
                w, Q = np.linalg.eigh(M)  # LAPACK dsyevd (GPU: syevjBatched for n <= 32, syevd otherwise)
                np.maximum(w, 0.0, out=w)  # max_dense_vector_zero
                P = np.matmul(Q * w[:, None, :], np.swapaxes(Q, 1, 2))  # (Q diag(w+)) Q^T
                S[vidx] = P[:, jj, ii] * g['fac_out']  # matrices_to_vector reads the upper-triangle entry (j, i)
            if has_l:
                xl = Xinput[lidx]
                S[lidx] = np.where(xl > 0.0, xl, 0.0)
            if has_u:
                S[uidx] = 0.0
            SmC = S - C_s
            # ---------------- Step 4: X += tau sig (A^T y - C + S) (solver.cu:1931-1949) ----------------
            Rd = Rd1 + S
            tau = TAU_ADMM
            if self.errRd < stop_tol:
                tau = max(1.618, tau / 1.1)
            self.tau = tau
            X = X + (tau * sig) * Rd
            # ---------------- Step 5: residuals (solver.cu:1954-1982) ----------------
            Rp = b_s - (A @ X)
            errRp = nrm2(normA * Rp * bscale) / norm_borg
            pobj = float(np.dot(C_s_val, X[C_idx])) * objscale
            errRd = nrm2(Rd * Cscale) / norm_Corg
            dobj = float(np.dot(b_s_val, y[b_idx])) * objscale
            maxfeas = max(errRp, errRd)
            relgap = abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj))
            self.errRp, self.errRd, self.pobj, self.dobj, self.maxfeas, self.relgap = errRp, errRd, pobj, dobj, maxfeas, relgap
            if not (math.isfinite(errRp) and math.isfinite(errRd) and math.isfinite(relgap)):
                if validating:
                    self.nonfinite = True
                else:
                    raise RuntimeError(f'non-finite KKT residuals at iteration {it}')
            # ---------------- primal/dual "wins" and the sigma update (solver.cu:2000-2090) ----------------
            sig_before = sig
            if self.bf is not None:
                # v1.2 balance_freeze: the state object votes, checks and freezes exactly as the C++ policy
                self.feasratio = _div(self.ratioconst * errRp, errRd)
                n_log = len(self.bf.log)
                self.sig = self.bf.update(it, errRp, errRd, relgap, self.ratioconst)
                self.prim_win, self.dual_win = self.bf.prim_win, self.bf.dual_win
                self.sigma_log.extend(self.bf.log[n_log:])
            else:
                feasratio = _div(self.ratioconst * errRp, errRd)
                self.feasratio = feasratio
                if feasratio < 1:
                    self.prim_win += 1
                else:
                    self.dual_win += 1
            pw0, dw0 = self.prim_win, self.dual_win
            if rule.adaptive and self.bf is None:
                k = it
                db = rule.deadband
                if db is not None:  # v1.1 dead-band window: sum of log(errRp / errRd) since the previous check
                    self.db_sumlog += math.log(max(errRp, DB_TINY)) - math.log(max(errRd, DB_TINY))
                    self.db_count += 1
                if schedule_check(k):
                    if rule.freeze_after is None or k <= rule.freeze_after:
                        sigscale = max(2.0 * math.exp(-k / 50000.0), 2.0)
                        if self.prim_win > max(rule.guard, rule.ratio * self.dual_win):
                            if db is None or self.db_sumlog / self.db_count < -math.log(db):
                                self.prim_win = 0
                                self.sig = min(rule.sigmax, self.sig * sigscale)
                            else:
                                self.db_blocked_up += 1
                        elif self.dual_win > max(rule.guard, rule.ratio * self.prim_win):
                            if db is None or self.db_sumlog / self.db_count > math.log(db):
                                self.dual_win = 0
                                self.sig = max(rule.sigmin, self.sig / sigscale)
                            else:
                                self.db_blocked_down += 1
                    if db is not None:
                        self.db_sumlog, self.db_count = 0.0, 0
                if self.sig != sig_before:
                    self.sigma_log.append(dict(iter=it, old_sigma=sig_before, new_sigma=self.sig, rule='admm_schedule', errRp=errRp,
                                               errRd=errRd, relgap=relgap, feasratio=feasratio, prim_win=pw0, dual_win=dw0))
                if rule.monitor1 and k > 5000 and k % self.monitor.update_interval == 0:
                    sig_mon = self.sig
                    self.monitor.push(pobj, dobj, errRp, errRd, relgap)
                    if self.monitor.full():
                        self.sig = self.monitor.chase_update_sig(self.sig)
                    if self.sig != sig_mon:
                        self.sigma_log.append(dict(iter=it, old_sigma=sig_mon, new_sigma=self.sig, rule='monitor1', errRp=errRp,
                                                   errRd=errRd, relgap=relgap, feasratio=feasratio, prim_win=pw0, dual_win=dw0))
            if self.sig != sig_before:
                self.sigma_changes += 1
            if not rule.adaptive and self.sig != self.sig0:
                raise RuntimeError('fixed sigma policy violated')
            if self.history_stride > 0 and it % self.history_stride == 0 and (self.history_until < 0 or it <= self.history_until):
                self.history.append((it, errRp, errRd, relgap, pobj, dobj, self.sig, self.prim_win, self.dual_win))
            self.iter = it + 1
            self.iterations_done += 1
            t_iter += time.perf_counter() - t0
            if progress_every and it % progress_every == 0:
                print(f'  it {it:9d} errRp {errRp:.3e} errRd {errRd:.3e} relgap {relgap:.3e} sig {self.sig:.3e} '
                      f'| {1e6 * (self.cpu_iter_s + t_iter) / max(1, self.iterations_done):.1f} us/it', flush=True)
        self.X, self.y, self.S, self.Rp, self.SmC = X, y, S, Rp, SmC
        self.cpu_iter_s += t_iter
        return status

    # ---- results ----
    def summary(self):
        it_done = self.iter - 1
        b = self.best_validation
        return dict(problem=self.prob.path, name=self.prob.name, emulator=__version__, rule=self.rule.to_dict(), sig0=self.sig0,
                    stop_tol=self.stop_tol, validate_interval=self.validate_interval, validate_tol=self.validate_tol,
                    strict_dimacs_tol=self.strict_dimacs_tol, stop_on_validation=self.stop_on_validation, max_iter=self.max_iter,
                    finished=self.finished, stop_reason=self.stop_reason, iterations=it_done, sigma_changes=self.sigma_changes,
                    sigma_log_entries=len(self.sigma_log), final_sig=self.sig, practical_validated=self.practical_validated,
                    first_practical_iteration=self.first_practical_iteration, strict_validated=self.strict_validated,
                    first_strict_iteration=self.first_strict_iteration, first_internal_below_1e_3=self.first_internal_below_1e_3,
                    first_internal_below_tol=self.first_internal_below_tol, validation_count=len(self.validations),
                    errRp=self.errRp, errRd=self.errRd, relgap=self.relgap, pobj=self.pobj, dobj=self.dobj,
                    bscale=self.bscale, Cscale=self.Cscale, norm_borg=self.norm_borg, norm_Corg=self.norm_Corg,
                    cpu_iter_s=self.cpu_iter_s, cpu_validation_s=self.cpu_validation_s,
                    us_per_iteration=1e6 * self.cpu_iter_s / max(1, self.iterations_done),
                    best_iter=b['iter'] if b else -1, best_merit=b['merit'] if b else None,
                    best_max_abs_dimacs=b['max_abs_dimacs'] if b else None, warm_start=self.warm, linear_solver=self.chol_kind,
                    psd_groups=[(g['n'], g['nb'], g['cat']) for g in self.psd_groups],
                    deadband_blocked_up=self.db_blocked_up, deadband_blocked_down=self.db_blocked_down,
                    freeze_iteration=self.bf.freeze_at if self.bf is not None else -1,
                    freeze_reason=self.bf.freeze_reason if self.bf is not None else '',
                    segments=self.segments)

    def write_outputs(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        _write_csv(os.path.join(out_dir, 'sigma_log.csv'), SIGMA_LOG_COLUMNS, self.sigma_log)
        if self.gzip_validation:
            _write_csv(os.path.join(out_dir, 'validation_history.csv.gz'), VALIDATION_COLUMNS, self.validations, gz=True)
        else:
            _write_csv(os.path.join(out_dir, 'validation_history.csv'), VALIDATION_COLUMNS, self.validations)
        if len(self.history):
            h = self.history.array()
            np.savez(os.path.join(out_dir, 'history.npz'), iter=h[:, 0].astype(np.int64), errRp=h[:, 1], errRd=h[:, 2],
                     relgap=h[:, 3], pobj=h[:, 4], dobj=h[:, 5], sig=h[:, 6], prim_win=h[:, 7].astype(np.int64),
                     dual_win=h[:, 8].astype(np.int64))
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(_jsonable(self.summary()), f, indent=1)

    # ---- checkpoint / resume ----
    _SCALARS = ('sig0', 'stop_tol', 'max_iter', 'validate_interval', 'validate_tol', 'strict_dimacs_tol', 'stop_on_validation',
                'history_stride', 'history_until', 'warm', 'errRp', 'errRd', 'maxfeas', 'pobj', 'dobj', 'relgap', 'sig', 'prim_win',
                'dual_win', 'ratioconst', 'feasratio', 'tau', 'iter', 'stop_reason', 'finished', 'nonfinite', 'prev_internal_ok',
                'sigma_changes', 'practical_validated', 'strict_validated', 'first_practical_iteration', 'first_strict_iteration',
                'first_internal_below_1e_3', 'first_internal_below_tol', 'cpu_iter_s', 'cpu_validation_s', 'iterations_done')
    _VAL_STR = ('trigger', 'failed')
    _VAL_NUM = tuple(c for c in VALIDATION_COLUMNS if c not in ('trigger', 'failed')) + ('internal_pobj', 'internal_dobj', 'merit')
    _VAL_INT = ('iter', 'checkpoint', 'X_negative_blocks', 'finite', 'validated', 'err6_defined')
    _SIG_NUM = ('iter', 'old_sigma', 'new_sigma', 'errRp', 'errRd', 'relgap', 'feasratio', 'prim_win', 'dual_win')
    _SIG_INT = ('iter', 'prim_win', 'dual_win')

    @classmethod
    def _recs_to_arrays(cls, recs, num, strs):
        a = np.array([[float(r.get(k, float('nan'))) for k in num] for r in recs], dtype=np.float64).reshape(-1, len(num))
        s = {k: np.array([str(r.get(k, '')) for r in recs], dtype=str) for k in strs}
        return a, s

    @classmethod
    def _arrays_to_recs(cls, a, s, num, strs, ints):
        out = []
        cols = [s[k].tolist() for k in strs]
        for i, row in enumerate(a.tolist()):
            r = {k: (int(v) if k in ints else v) for k, v in zip(num, row)}
            for k, c in zip(strs, cols):
                r[k] = c[i]
            out.append(r)
        return out

    def save_state(self, path):
        """Checkpoint: iterate, every scalar of the solve state, Monitor1 queues, logs and history (.npz)."""
        meta = {k: getattr(self, k) for k in self._SCALARS}
        meta.update({k: getattr(self, k) for k in self._V11_DEFAULTS})
        meta.update(problem=self.prob.path, rule=self.rule.to_dict(), monitor=self.monitor.state(),
                    balance_freeze=self.bf.state() if self.bf is not None else None,
                    best_validation=self.best_validation, version=__version__, segments=self.segments,
                    check=dict(bscale=self.bscale, Cscale=self.Cscale, normA_sum=float(np.sum(self.normA))))
        va, vs = self._recs_to_arrays(self.validations, self._VAL_NUM, self._VAL_STR)
        sa, ss = self._recs_to_arrays(self.sigma_log, self._SIG_NUM, ('rule',))
        tmp = path + '.tmp.npz'
        np.savez(tmp, X=self.X, y=self.y, S=self.S, Rp=self.Rp, SmC=self.SmC, history=self.history.array(),
                 val_num=va, val_trigger=vs['trigger'], val_failed=vs['failed'], sig_num=sa, sig_rule=ss['rule'],
                 meta=np.array(json.dumps(_jsonable(meta))))
        os.replace(tmp, path)

    @classmethod
    def load_state(cls, path, problem_dir=None):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z['meta']))
        prob = Problem(problem_dir or meta['problem'])
        e = cls(prob, rule=SigmaRule.from_dict(meta['rule']), _skip_init=True, sig0=meta['sig0'], stop_tol=meta['stop_tol'],
                validate_interval=meta['validate_interval'])
        for k in cls._SCALARS:
            setattr(e, k, _float(meta[k]))
        for k, v in cls._V11_DEFAULTS.items():  # absent from v1.0 checkpoints
            setattr(e, k, _float(meta.get(k, v)))
        e.segments = list(meta.get('segments', []))
        e.validator = ExternalValidator(prob) if e.validate_interval > 0 else None
        chk = meta['check']
        if chk['bscale'] != e.bscale or chk['Cscale'] != e.Cscale or chk['normA_sum'] != float(np.sum(e.normA)):
            raise RuntimeError('the scaling recomputed from the problem differs from the checkpoint')
        e.X, e.y, e.S, e.Rp, e.SmC = (np.array(z[k]) for k in ('X', 'y', 'S', 'Rp', 'SmC'))
        e.monitor = Monitor1()
        e.monitor.set_state(meta['monitor'])
        e.bf = BalanceFreezeState(e.rule, e.sig0) if e.rule.freeze else None
        if e.bf is not None and meta.get('balance_freeze'):
            e.bf.set_state(meta['balance_freeze'])
        e.validations = cls._arrays_to_recs(np.array(z['val_num']), {'trigger': np.array(z['val_trigger']),
                                            'failed': np.array(z['val_failed'])}, cls._VAL_NUM, cls._VAL_STR, cls._VAL_INT)
        e.sigma_log = cls._arrays_to_recs(np.array(z['sig_num']), {'rule': np.array(z['sig_rule'])}, cls._SIG_NUM, ('rule',),
                                          cls._SIG_INT)
        bv = meta['best_validation']
        e.best_validation = {k: _float(v) for k, v in bv.items()} if bv else None
        e.history = History.from_array(np.array(z['history']))
        return e


def _better(a, b):
    # ValidationResult::better_than
    if bool(a['validated']) != bool(b['validated']):
        return bool(a['validated'])
    if a['validated']:
        x = a['max_abs_dimacs'] if math.isfinite(a['max_abs_dimacs']) else float('inf')
        yv = b['max_abs_dimacs'] if math.isfinite(b['max_abs_dimacs']) else float('inf')
        return x < yv
    return a['merit'] < b['merit']


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return repr(o)  # 'nan', 'inf', '-inf' (restored by _float)
    return o


def _fmt(v):
    if isinstance(v, float):
        return repr(v)  # full precision (17 significant digits); the GPU CSVs have 12
    if isinstance(v, str) and (',' in v or v == ''):
        return '"' + v + '"'
    return str(v)


def _write_csv(path, cols, rows, gz=False):
    import gzip
    tmp = path + '.tmp'
    with (gzip.open(tmp, 'wt', compresslevel=6) if gz else open(tmp, 'w')) as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(_fmt(_float(r.get(c, ''))) for c in cols) + '\n')
    os.replace(tmp, path)


def _float(v):
    if isinstance(v, str) and v in ('nan', 'inf', '-inf'):
        return float(v)
    return v


def read_gpu_csv(path):
    """Reads a sigma_log / validation_history CSV (optionally .gz) of the GPU runs or of the emulator into a list of dicts
    (numbers as floats/ints)."""
    import gzip
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        rd = csv.DictReader(f)
        out = []
        for row in rd:
            d = {}
            for k, v in row.items():
                try:
                    d[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        d[k] = float(v)
                    except (ValueError, TypeError):
                        d[k] = v
            out.append(d)
    return out


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------
def _cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return ''


def parse_rule(spec):
    """'old' | 'new' | ... | 'balance_freeze' | 'balance_freeze:factor=1.2,step_schedule=halve_on_reversal,...'"""
    if ':' not in spec:
        if spec not in RULES:
            raise ValueError('unknown rule ' + spec)
        return RULES[spec]()
    name, params = spec.split(':', 1)
    if name != 'balance_freeze':
        raise ValueError('only balance_freeze takes parameters')
    kw = {}
    spec = 'balance_freeze:' + ','.join(sorted(item for item in params.split(',') if item))
    for item in params.split(','):
        if not item:
            continue
        key, value = item.split('=', 1)
        if key in ('sigma_min', 'sigma_max'):
            kw['sigmin' if key == 'sigma_min' else 'sigmax'] = float(value)
        elif key in ('factor', 'freeze_bias'):
            kw[key] = float(value)
        elif key in ('freeze_iteration', 'pingpong_window', 'pingpong_net', 'pingpong_min_iter'):
            kw[key] = int(value)
        elif key in ('step_schedule', 'freeze_value'):
            kw[key] = value
        else:
            raise ValueError('unknown balance_freeze parameter ' + key)
    return balance_freeze_rule(name=spec, **kw)


def _cli(argv=None):
    ap = argparse.ArgumentParser(description='CPU replica of the cuADMM plain-ADMM solve (main 2d8a755)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='start a solve')
    r.add_argument('problem_dir')
    r.add_argument('--out', required=True, help='output directory (sigma_log.csv, validation_history.csv, summary.json, state.npz)')
    r.add_argument('--rule', default='old', help='old, new, fixed, oldnom1, newfreeze, newdb, balance_freeze, or '
                   'balance_freeze:key=value,... with the option names of cuADMM-reorg penalty.* (factor, step_schedule, '
                   'freeze_iteration, pingpong_window, pingpong_net, pingpong_min_iter, freeze_value, freeze_bias, sigma_min, '
                   'sigma_max)')
    r.add_argument('--sig', type=float, default=1.0, help='initial sigma (the campaigns used 1; the CLI default of cuadmm_exe is 1e2)')
    r.add_argument('--tol', type=float, default=1e-4)
    r.add_argument('--max-iter', type=int, default=1000000000, help='safety cap of the solve (the GPU max_iter semantics)')
    r.add_argument('--validate-interval', type=int, default=50)
    r.add_argument('--validate-tol', type=float, default=None)
    r.add_argument('--strict-dimacs-tol', type=float, default=0.0)
    r.add_argument('--no-stop', action='store_true', help='keep iterating after a validated snapshot (records first practical/strict)')
    r.add_argument('--history-stride', type=int, default=0, help='record (iter, errRp, errRd, relgap, pobj, dobj, sig, wins) '
                   'every K iterations (history.npz)')
    r.add_argument('--history-until', type=int, default=None, help='... up to this iteration only')
    r.add_argument('--gzip-validation', action='store_true', help='write validation_history.csv.gz (v1.1)')
    for q in (r, sub.add_parser('resume', help='continue a paused solve from <dir>/state.npz')):
        if q is not r:
            q.add_argument('dir')
            q.add_argument('--max-iter', type=int, default=None)
        q.add_argument('--pause-at', type=int, default=None, help='stop (resumable) after this iteration')
        q.add_argument('--wall-limit', type=float, default=None, help='stop (resumable) after this many seconds')
        q.add_argument('--checkpoint-every', type=float, default=300.0, help='seconds between checkpoints')
        q.add_argument('--progress-every', type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        rule = parse_rule(a.rule)
        e = Emulator(Problem(a.problem_dir), rule=rule, sig0=a.sig, stop_tol=a.tol, max_iter=a.max_iter,
                     validate_interval=a.validate_interval, validate_tol=a.validate_tol, strict_dimacs_tol=a.strict_dimacs_tol,
                     stop_on_validation=not a.no_stop, history_stride=a.history_stride, history_until=a.history_until)
        e.gzip_validation = bool(a.gzip_validation)
        out = a.out
    else:
        out = a.dir
        e = Emulator.load_state(os.path.join(out, 'state.npz'))
        if e.finished:
            print(f'{out}: the solve already ended ({e.stop_reason}); nothing to resume')
            return 0
        if a.max_iter is not None:
            e.max_iter = a.max_iter
    os.makedirs(out, exist_ok=True)
    ck = os.path.join(out, 'state.npz')
    seg = dict(cmd=a.cmd, host=os.uname().nodename, cpu=_cpu_model(), slurm_job=os.environ.get('SLURM_JOB_ID', ''),
               start_iter=e.iter - 1, cpu_iter_s0=e.cpu_iter_s, cpu_validation_s0=e.cpu_validation_s,
               started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    t_seg = time.perf_counter()
    status = e.run(pause_at=a.pause_at, wall_limit=a.wall_limit, progress_every=a.progress_every, checkpoint_path=ck,
                   checkpoint_every_s=a.checkpoint_every)
    seg.update(end_iter=e.iter - 1, wall_s=time.perf_counter() - t_seg, iter_s=e.cpu_iter_s - seg.pop('cpu_iter_s0'),
               validation_s=e.cpu_validation_s - seg.pop('cpu_validation_s0'), status=status)
    e.segments.append(seg)
    e.save_state(ck)
    e.write_outputs(out)
    s = e.summary()
    print(json.dumps({k: s[k] for k in ('name', 'stop_reason', 'iterations', 'sigma_log_entries', 'final_sig', 'first_practical_iteration',
                                        'first_strict_iteration', 'us_per_iteration')}) + f' status={status}')
    return 0


if __name__ == '__main__':
    sys.exit(_cli())
