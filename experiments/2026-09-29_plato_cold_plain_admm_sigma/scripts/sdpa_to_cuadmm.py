#!/usr/bin/env python3
"""Converter from the sparse SDPA format (.dat-s) to cuADMM's TXT input (blk.txt, At.txt, b.txt, C.txt, con_num.txt).

SDPA pair (SDPA manual):  (P) min c^T x  s.t.  X = sum_i F_i x_i - F_0 >= 0
                          (D) max <F_0, Y>  s.t.  <F_i, Y> = c_i,  Y >= 0
cuADMM pair:              min <C, X>  s.t.  <A_i, X> = b_i, X in K;   max b^T y  s.t.  sum_i y_i A_i + S = C, S in K*
Mapping: cuADMM's primal is SDPA's (D) written as a minimization:
    C = -F_0,  A_i = F_i,  b_i = c_i,  X_cuADMM = Y_SDPA,  y_cuADMM = -x_SDPA,  S_cuADMM = X_SDPA,
    optimal value (SDPA convention, as published) = -<C, X> = -b^T y.
Blocks: an SDPA block of positive size n is an n x n PSD block ('s n'); a block of negative size -n is a diagonal
(LP, nonnegative) block ('l n'); a 1 x 1 PSD block is written as 'l 1' (the same cone: a nonnegative scalar; cuADMM
keeps it out of its eigendecompositions). Block order is kept. svec convention of cuADMM: upper triangle, column by
column, index offset + j(j+1)/2 + i for 0-based i <= j, off-diagonal entries multiplied by sqrt(2), so that
svec(A) . svec(X) = <A, X>.
Checks (the instance is rejected, exit status 2, if any fails): header and counts consistent; every entry has a matrix
number in [0, m], a block in [1, nBLOCK], indices in range (1-based) and i == j in diagonal blocks; non-finite
values; duplicate entries (same matrix, block and unordered (i, j)), which are reported and rejected rather than
guessed; empty constraint matrices F_i (would make A rank deficient). Reported but accepted: explicit zeros (dropped),
lower-triangle entries (mirrored to the upper triangle), blocks that no F_i touches.

usage: sdpa_to_cuadmm.py <file.dat-s | file.dat-s.gz | zip:member> <output dir>   (writes also meta.json)
"""
import gzip, io, json, math, os, re, sys, zipfile
from collections import defaultdict

SQRT2 = math.sqrt(2.0)


class ConversionError(Exception):
    pass


def open_text(src):
    if src.startswith('zip:'):
        zpath, member = src[4:].split('::', 1)
        return io.TextIOWrapper(zipfile.ZipFile(zpath).open(member), errors='strict')
    if src.endswith('.gz'):
        return gzip.open(src, 'rt')
    return open(src)


def tokens(line):
    return [t for t in re.split(r'[\s,{}()]+', line.strip()) if t]


def read_sdpa(src):
    """Returns dict(m, blocks (signed sizes), c, entries {(matno, blk, i, j): value} with 0-based i <= j, stats)."""
    with open_text(src) as f:
        lines = f.read().splitlines()
    k = 0
    while k < len(lines) and (not lines[k].strip() or lines[k].lstrip()[0] in '"*'):
        k += 1
    stats = dict(comment_lines=k)

    def next_ints(count, what):
        nonlocal k
        vals = []
        while len(vals) < count:
            if k >= len(lines):
                raise ConversionError(f'unexpected end of file while reading {what}')
            toks = tokens(lines[k])
            k += 1
            for t in toks:
                if len(vals) == count:
                    break  # trailing text on a header line (e.g. '= mDIM') is ignored
                try:
                    vals.append(int(float(t)) if re.fullmatch(r'[-+]?\d+(\.0*)?', t) else None)
                except ValueError:
                    vals.append(None)
                if vals[-1] is None:
                    vals.pop()
                    break  # a non-numeric token ends the header line
        return vals

    m = next_ints(1, 'm')[0]
    nblock = next_ints(1, 'nBLOCK')[0]
    blocks = next_ints(nblock, 'bLOCKsTRUCT')
    if m <= 0 or nblock <= 0 or len(blocks) != nblock or any(b == 0 for b in blocks):
        raise ConversionError(f'bad header: m={m} nBLOCK={nblock} blocks={blocks}')
    c = []
    while len(c) < m:
        if k >= len(lines):
            raise ConversionError('unexpected end of file while reading c')
        for t in tokens(lines[k]):
            if len(c) < m:
                c.append(float(t))
        k += 1
    entries = {}
    raw = defaultdict(int)
    stats.update(entry_lines=0, zeros_dropped=0, lower_mirrored=0, duplicates=[], nonfinite=0)
    for ln in range(k, len(lines)):
        toks = tokens(lines[ln])
        if not toks:
            continue
        if len(toks) != 5:
            raise ConversionError(f'line {ln + 1}: expected 5 fields, got {len(toks)}: {lines[ln]!r}')
        matno, blk, i, j = (int(float(t)) for t in toks[:4])
        v = float(toks[4])
        stats['entry_lines'] += 1
        if not (0 <= matno <= m):
            raise ConversionError(f'line {ln + 1}: matrix number {matno} outside [0, {m}]')
        if not (1 <= blk <= nblock):
            raise ConversionError(f'line {ln + 1}: block {blk} outside [1, {nblock}]')
        n = abs(blocks[blk - 1])
        if not (1 <= i <= n and 1 <= j <= n):
            raise ConversionError(f'line {ln + 1}: index ({i}, {j}) outside a block of size {n}')
        if blocks[blk - 1] < 0 and i != j:
            raise ConversionError(f'line {ln + 1}: off-diagonal entry ({i}, {j}) in the diagonal block {blk}')
        if not math.isfinite(v):
            stats['nonfinite'] += 1
            raise ConversionError(f'line {ln + 1}: non-finite value {toks[4]}')
        if i > j:
            i, j = j, i
            stats['lower_mirrored'] += 1
        key = (matno, blk - 1, i - 1, j - 1)
        raw[key] += 1
        if raw[key] > 1:
            stats['duplicates'].append(dict(line=ln + 1, key=key))
        if v == 0.0:
            stats['zeros_dropped'] += 1
            continue
        entries[key] = entries.get(key, 0.0) + v
    if stats['duplicates']:
        raise ConversionError(f"{len(stats['duplicates'])} duplicate entries (first: {stats['duplicates'][0]})")
    return dict(m=m, blocks=blocks, c=c, entries=entries, stats=stats)


def cuadmm_blocks(blocks):
    """cuADMM blocks (type, size, offset) and the DIMACS kind of each ('s': PSD in SDPA, 'l': diagonal in SDPA)."""
    out, off = [], 0
    for b in blocks:
        if b > 1:
            out.append(dict(type='s', size=b, offset=off, dimacs='s'))
            off += b * (b + 1) // 2
        elif b == 1:
            out.append(dict(type='l', size=1, offset=off, dimacs='s'))  # 1 x 1 PSD block = nonnegative scalar
            off += 1
        else:
            out.append(dict(type='l', size=-b, offset=off, dimacs='l'))
            off += -b
    return out, off


def svec_index(blk, i, j):
    """0-based i <= j within the block."""
    if blk['type'] == 's':
        return blk['offset'] + j * (j + 1) // 2 + i
    assert i == j
    return blk['offset'] + i


def convert(src, outdir):
    d = read_sdpa(src)
    m, blocks, c, entries = d['m'], d['blocks'], d['c'], d['entries']
    blks, vec_len = cuadmm_blocks(blocks)
    At, C = [], {}
    nnz_per_mat = defaultdict(int)
    touched_blocks = set()
    for (matno, b, i, j), v in entries.items():
        blk = blks[b]
        idx = svec_index(blk, i, j)
        val = v * (SQRT2 if (blk['type'] == 's' and i != j) else 1.0)
        if matno == 0:
            C[idx] = C.get(idx, 0.0) - val  # C = -F_0
        else:
            At.append((idx, matno - 1, val))
            nnz_per_mat[matno] += 1
            touched_blocks.add(b)
    empty_constraints = [k for k in range(1, m + 1) if nnz_per_mat[k] == 0]
    if empty_constraints:
        raise ConversionError(f'{len(empty_constraints)} empty constraint matrices F_i (first: {empty_constraints[:5]})')
    untouched = [b + 1 for b in range(len(blocks)) if b not in touched_blocks]
    os.makedirs(outdir, exist_ok=True)
    At.sort()
    with open(os.path.join(outdir, 'At.txt'), 'w') as f:
        for idx, col, val in At:
            f.write(f'{idx} {col} {val:.17g}\n')
    with open(os.path.join(outdir, 'C.txt'), 'w') as f:
        for idx in sorted(C):
            if C[idx] != 0.0:
                f.write(f'{idx} 0 {C[idx]:.17g}\n')
    with open(os.path.join(outdir, 'b.txt'), 'w') as f:
        for k, v in enumerate(c):
            if v != 0.0:
                f.write(f'{k} 0 {v:.17g}\n')
    with open(os.path.join(outdir, 'blk.txt'), 'w') as f:
        for blk in blks:
            f.write(f"{blk['type']} {blk['size']}\n")  # the type is always explicit (a bare size means 's')
    with open(os.path.join(outdir, 'con_num.txt'), 'w') as f:
        f.write(f'{m}\n')
    with open(os.path.join(outdir, 'dimacs_blocks.txt'), 'w') as f:
        # for the validator: the SDPA kind of each cuADMM block ('s' PSD, 'l' diagonal), for the DIMACS norm
        for blk in blks:
            f.write(f"{blk['dimacs']} {blk['size']}\n")
    psd = [b for b in blocks if b > 1]
    meta = dict(source=src, m=m, nblock=len(blocks), sdpa_blocks=blocks, vec_len=vec_len,
                psd_block_sizes=sorted(set(psd)), n_psd_blocks=len(psd), max_psd_block=max(psd) if psd else 0,
                n_1x1_psd_blocks=sum(1 for b in blocks if b == 1), lp_block_sizes=[-b for b in blocks if b < 0],
                nnz_At=len(At), nnz_C=sum(1 for v in C.values() if v != 0.0), nnz_b=sum(1 for v in c if v != 0.0),
                nnz_entries_total=len(entries), untouched_blocks=untouched, **{k: v for k, v in d['stats'].items() if k != 'duplicates'},
                duplicates=len(d['stats']['duplicates']), cuadmm_blocks=[(b['type'], b['size']) for b in blks],
                est_host_bytes_At=len(At) * 16, est_gpu_bytes_dense_blocks=8 * sum(b * b for b in psd) * 6)
    json.dump(meta, open(os.path.join(outdir, 'meta.json'), 'w'), indent=1)
    return meta


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    try:
        meta = convert(sys.argv[1], sys.argv[2])
    except ConversionError as e:
        print(f'CONVERSION_FAILED: {e}')
        sys.exit(2)
    print(json.dumps({k: meta[k] for k in ('m', 'nblock', 'vec_len', 'n_psd_blocks', 'max_psd_block', 'lp_block_sizes', 'n_1x1_psd_blocks',
                                           'nnz_At', 'nnz_C', 'nnz_b', 'zeros_dropped', 'lower_mirrored', 'duplicates', 'untouched_blocks')}))
