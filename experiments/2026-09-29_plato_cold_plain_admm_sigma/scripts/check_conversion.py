#!/usr/bin/env python3
"""Independent checks of a converted instance (does not import sdpa_to_cuadmm.py).

1. An independent SDPA reader (a different tokenizer, no svec) builds the symmetric data matrices F_0..F_m as
   dictionaries of upper-triangle entries.
2. A TXT reader reads At.txt, b.txt, C.txt, blk.txt and con_num.txt the way cuADMM does (0-based COO).
3. Forward check: for random block-diagonal symmetric Y (random nonnegative diagonals for the 'l' blocks),
   <F_k, Y> computed from the full symmetric definition (both triangles) must equal (A svec(Y))_k from At.txt,
   and <F_0, Y> must equal -C . svec(Y); b must equal the SDPA c vector.
4. Adjoint check: for random w in R^m, the upper triangle of M = sum_k w_k F_k - F_0 must equal svec^-1(At w + C).
5. Round trip: the TXT data written back to SDPA and re-read must give the original entries (to 1e-15 relative,
   since v * sqrt(2) / sqrt(2) is not always exact in floating point).
6. Structure: block sizes and kinds vs the SDPA header, vec_len, con_num, largest indices; numerical rank of A
   (dense A A^T eigenvalues) when m <= 6000.
Writes <dir>/conversion_check.json; exit status 0 if every check passes, 1 otherwise.

usage: check_conversion.py <file.dat-s | zip:path::member> <converted dir> [--trials 3] [--seed 12345]
"""
import argparse, gzip, io, json, math, os, re, sys, zipfile
import numpy as np
import scipy.sparse as sp

TOL = 1e-12


def read_text(src):
    if src.startswith('zip:'):
        zpath, member = src[4:].split('::', 1)
        return zipfile.ZipFile(zpath).read(member).decode()
    return gzip.open(src, 'rt').read() if src.endswith('.gz') else open(src).read()


def independent_sdpa(src):
    """Reads the whole file as a stream of numbers after the comment lines (a different approach from the converter)."""
    text = read_text(src)
    body = [l for l in text.splitlines() if l.strip() and l.lstrip()[0] not in '"*']
    # header: first number of line 1 = m, first number of line 2 = nBLOCK, line 3+ = the block sizes
    num = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
    m = int(float(num.findall(body[0])[0]))
    nb = int(float(num.findall(body[1])[0]))
    rest = ' '.join(body[2:])
    vals = num.findall(rest)
    blocks = [int(float(v)) for v in vals[:nb]]
    c = np.array([float(v) for v in vals[nb:nb + m]])
    data = np.array([float(v) for v in vals[nb + m:]])
    if data.size % 5:
        raise SystemExit('independent reader: the entry section is not a multiple of 5 numbers')
    rec = data.reshape(-1, 5)
    F = {}
    for matno, blk, i, j, v in rec:
        matno, blk, i, j = int(matno), int(blk) - 1, int(i) - 1, int(j) - 1
        if v == 0.0:
            continue
        a, b = min(i, j), max(i, j)
        F.setdefault(matno, {})
        F[matno][(blk, a, b)] = F[matno].get((blk, a, b), 0.0) + v
    return m, blocks, c, F


def read_txt(d):
    blk = []
    for line in open(os.path.join(d, 'blk.txt')):
        t = line.split()
        if len(t) == 2:
            blk.append((t[0], int(t[1])))
        elif len(t) == 1:
            blk.append(('s', int(t[0])))
    kinds = [tuple([l.split()[0], int(l.split()[1])]) for l in open(os.path.join(d, 'dimacs_blocks.txt'))]
    con_num = int(open(os.path.join(d, 'con_num.txt')).read().split()[0])
    vec_len = sum(n * (n + 1) // 2 if t == 's' else n for t, n in blk)
    At = np.loadtxt(os.path.join(d, 'At.txt'), ndmin=2)
    A = sp.csr_matrix((At[:, 2], (At[:, 1].astype(int), At[:, 0].astype(int))), shape=(con_num, vec_len))
    def vec(name, n):
        v = np.zeros(n)
        p = os.path.join(d, name)
        if os.path.getsize(p):
            x = np.loadtxt(p, ndmin=2)
            np.add.at(v, x[:, 0].astype(int), x[:, 2])
        return v
    return blk, kinds, con_num, vec_len, A, vec('b.txt', con_num), vec('C.txt', vec_len), At


def offsets(blk):
    off, out = 0, []
    for t, n in blk:
        out.append(off)
        off += n * (n + 1) // 2 if t == 's' else n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dir')
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--seed', type=int, default=12345)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    m, sblocks, c, F = independent_sdpa(a.src)
    blk, kinds, con_num, vec_len, A, b, C, At = read_txt(a.dir)
    offs = offsets(blk)
    res = dict(problem=os.path.basename(a.dir), checks={})
    ok_all = True

    def check(name, ok, detail):
        nonlocal ok_all
        res['checks'][name] = dict(ok=bool(ok), detail=detail)
        ok_all = ok_all and bool(ok)

    # structure
    exp = []
    for s in sblocks:
        exp.append(('s', s) if s > 1 else ('l', abs(s)))
    exp_kinds = [('s', s) if s > 0 else ('l', -s) for s in sblocks]
    check('blocks', blk == exp and kinds == exp_kinds, dict(sdpa=sblocks[:10], cuadmm=blk[:10], n=len(blk)))
    check('con_num', con_num == m, dict(m=m, con_num=con_num))
    check('largest_indices', int(At[:, 0].max()) <= vec_len - 1 and int(At[:, 1].max()) == m - 1,
          dict(max_svec=int(At[:, 0].max()), vec_len=vec_len, max_con=int(At[:, 1].max())))
    check('b_equals_c', np.array_equal(b, c), dict(max_abs_diff=float(np.max(np.abs(b - c)))))

    def svec_of(Y):
        """cuADMM svec of a block-diagonal symmetric Y given as a list of blocks (matrices or diagonal vectors)."""
        v = np.zeros(vec_len)
        for bi, ((t, n), o) in enumerate(zip(blk, offs)):
            if t == 's':
                iu, ju = np.triu_indices(n)
                w = Y[bi][iu, ju] * np.where(iu == ju, 1.0, math.sqrt(2.0))
                v[o + ju * (ju + 1) // 2 + iu] = w
            else:
                v[o:o + n] = Y[bi]
        return v

    def inner(Fk, Y):
        s = 0.0
        for (bi, i, j), val in Fk.items():
            t = blk[bi][0]
            if t == 's':
                s += val * (Y[bi][i, j] + Y[bi][j, i]) if i != j else val * Y[bi][i, i]
            else:
                s += val * Y[bi][i]
        return s

    worst_f = worst_0 = 0.0
    for trial in range(a.trials):
        Y = []
        for t, n in blk:
            if t == 's':
                R = rng.standard_normal((n, n))
                Y.append((R + R.T) / 2)
            else:
                Y.append(rng.random(n))
        y = svec_of(Y)
        conv = A @ y
        ymax = max(1.0, max(float(np.max(np.abs(x))) for x in Y))  # once per trial
        for k in range(1, m + 1):
            ref = inner(F.get(k, {}), Y)
            scale = sum(abs(v) for v in F.get(k, {}).values()) * ymax
            worst_f = max(worst_f, abs(conv[k - 1] - ref) / max(scale, 1e-300))
        ref0 = inner(F.get(0, {}), Y)
        scale0 = sum(abs(v) for v in F.get(0, {}).values()) + 1e-300
        worst_0 = max(worst_0, abs(-C @ y - ref0) / scale0)
    check('forward_inner_products', worst_f <= TOL, dict(worst_relative=worst_f, trials=a.trials))
    check('objective_matrix', worst_0 <= TOL, dict(worst_relative=worst_0, trials=a.trials))

    # adjoint: svec(sum w_k F_k - F_0) == At w + C
    worst_adj = 0.0
    for trial in range(a.trials):
        w = rng.standard_normal(m)
        M = {}
        for k in range(0, m + 1):
            coef = -1.0 if k == 0 else w[k - 1]
            for key, val in F.get(k, {}).items():
                M[key] = M.get(key, 0.0) + coef * val
        ref = np.zeros(vec_len)
        for (bi, i, j), val in M.items():
            t, n = blk[bi]
            if t == 's':
                ref[offs[bi] + j * (j + 1) // 2 + i] = val * (math.sqrt(2.0) if i != j else 1.0)
            else:
                ref[offs[bi] + i] = val
        conv = A.T @ w + C
        scale = np.maximum(np.abs(ref), 1.0)
        worst_adj = max(worst_adj, float(np.max(np.abs(conv - ref) / scale)))
    check('adjoint', worst_adj <= 1e-10, dict(worst_relative=worst_adj, trials=a.trials))

    # round trip TXT -> SDPA -> independent reader
    lines = [f'{m}', f'{len(sblocks)}', ' '.join(str(s) for s in sblocks), ' '.join(f'{v:.17g}' for v in b)]
    inv = {}
    for bi, ((t, n), o) in enumerate(zip(blk, offs)):
        if t == 's':
            for jj in range(n):
                for ii in range(jj + 1):
                    inv[o + jj * (jj + 1) // 2 + ii] = (bi, ii, jj, 1.0 if ii == jj else 1.0 / math.sqrt(2.0))
        else:
            for ii in range(n):
                inv[o + ii] = (bi, ii, ii, 1.0)
    for idx in np.nonzero(C)[0]:
        bi, ii, jj, f = inv[int(idx)]
        lines.append(f'0 {bi + 1} {ii + 1} {jj + 1} {-C[idx] * f:.17g}')
    for r, col, val in At:
        bi, ii, jj, f = inv[int(r)]
        lines.append(f'{int(col) + 1} {bi + 1} {ii + 1} {jj + 1} {val * f:.17g}')
    tmp = os.path.join(a.dir, 'roundtrip.dat-s')
    open(tmp, 'w').write('\n'.join(lines) + '\n')
    m2, sb2, c2, F2 = independent_sdpa(tmp)
    worst_rt, missing = 0.0, 0
    for k in set(F) | set(F2):
        for key in set(F.get(k, {})) | set(F2.get(k, {})):
            u, v = F.get(k, {}).get(key), F2.get(k, {}).get(key)
            if u is None or v is None:
                missing += 1
                continue
            worst_rt = max(worst_rt, abs(u - v) / max(abs(u), 1e-300))
    os.remove(tmp)
    check('round_trip', m2 == m and sb2 == sblocks and np.array_equal(c2, c) and missing == 0 and worst_rt <= 1e-15,
          dict(worst_relative=worst_rt, missing_entries=missing))

    # rank of A (A A^T eigenvalues) for moderate m; larger instances: the CHOLMOD pivot report of cuADMM's init
    if m <= 6000:
        G = (A @ A.T).toarray()
        ev = np.linalg.eigvalsh(G)
        thr = ev[-1] * m * np.finfo(float).eps
        rank = int(np.sum(ev > thr))
        check('full_row_rank', rank == m, dict(rank=rank, m=m, min_eig_AAt=float(ev[0]), max_eig_AAt=float(ev[-1]),
                                               condition=float(ev[-1] / ev[0]) if ev[0] > 0 else None))
    else:
        res['checks']['full_row_rank'] = dict(ok=None, detail=f'm = {m} > 6000: see the factorization report of cuADMM init')
    res['all_ok'] = ok_all
    json.dump(res, open(os.path.join(a.dir, 'conversion_check.json'), 'w'), indent=1, default=float)
    print(json.dumps(dict(problem=res['problem'], all_ok=ok_all, **{k: v['ok'] for k, v in res['checks'].items()})))
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
