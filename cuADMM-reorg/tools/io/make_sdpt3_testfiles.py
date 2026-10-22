#!/usr/bin/env python3
"""Writes the SDPT3 test files of test/data (plan Phase 2b: the repository has no SDPT3 data):

    sdpt3_small_v5.mat   with scipy.io.savemat (MATLAB v5)
    sdpt3_small_v73.mat  with h5py (MATLAB v7.3: an HDF5 file with a 512-byte userblock holding the MATLAB header,
                         version 0x0200 at offset 124 and 'IM' at 126; arrays stored transposed, cells as object
                         references under /#refs#, sparse matrices as groups with data/ir/jc)

The problem is the one of test/io_mat_test.hpp (write_sdpt3 / check_sdpt3):
    blk = {'s', [2 1]; 'l', 2; 'u', 1}
    At{1} = [1 0; sqrt2 0; 0 2; 5 0]  (4 x 2: the svec of the 2x2 block then the 1x1)
    At{2} = sparse([3 0; 0 -1]),  At{3} = [0 7]
    C{1} = [1 2 0; 2 3 0; 0 0 4],  C{2} = [1; 0],  C{3} = -1,  b = [1; 2]
usage: make_sdpt3_testfiles.py <output directory>
"""
import os
import sys

import h5py
import numpy as np
import scipy.io
import scipy.sparse as sp

SQRT2 = np.sqrt(2.0)


def problem():
    blk = np.empty((3, 2), dtype=object)
    blk[0, 0] = 's'
    blk[0, 1] = np.array([[2.0, 1.0]])
    blk[1, 0] = 'l'
    blk[1, 1] = np.array([[2.0]])
    blk[2, 0] = 'u'
    blk[2, 1] = np.array([[1.0]])
    At = np.empty((3, 1), dtype=object)
    At[0, 0] = np.array([[1.0, 0.0], [SQRT2, 0.0], [0.0, 2.0], [5.0, 0.0]])
    At[1, 0] = sp.csc_matrix(np.array([[3.0, 0.0], [0.0, -1.0]]))
    At[2, 0] = np.array([[0.0, 7.0]])
    C = np.empty((3, 1), dtype=object)
    C[0, 0] = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    C[1, 0] = np.array([[1.0], [0.0]])
    C[2, 0] = np.array([[-1.0]])
    b = np.array([[1.0], [2.0]])
    return blk, At, C, b


def write_v5(path):
    blk, At, C, b = problem()
    scipy.io.savemat(path, {'blk': blk, 'At': At, 'C': C, 'b': b}, format='5', oned_as='column')


class V73Writer:
    def __init__(self, f):
        self.f = f
        self.refs = f.create_group('#refs#')
        self.count = 0

    def dense(self, loc, name, a):
        a = np.asarray(a, dtype=np.float64)
        d = loc.create_dataset(name, data=np.ascontiguousarray(a.T))  # MATLAB m x n -> HDF5 (n, m)
        d.attrs['MATLAB_class'] = np.bytes_(b'double')
        return d

    def char(self, loc, name, s):
        codes = np.array([[ord(c) for c in s]], dtype=np.uint16)  # 1 x len
        d = loc.create_dataset(name, data=np.ascontiguousarray(codes.T))
        d.attrs['MATLAB_class'] = np.bytes_(b'char')
        d.attrs['MATLAB_int_decode'] = np.int32(2)
        return d

    def sparse(self, loc, name, m):
        m = sp.csc_matrix(m)
        g = loc.create_group(name)
        g.create_dataset('data', data=np.asarray(m.data, dtype=np.float64))
        g.create_dataset('ir', data=np.asarray(m.indices, dtype=np.uint64))
        g.create_dataset('jc', data=np.asarray(m.indptr, dtype=np.uint64))
        g.attrs['MATLAB_class'] = np.bytes_(b'double')
        g.attrs['MATLAB_sparse'] = np.uint64(m.shape[0])
        return g

    def element(self, v):
        """stores one cell element under /#refs# and returns its reference"""
        name = 'e%d' % self.count
        self.count += 1
        if isinstance(v, str):
            d = self.char(self.refs, name, v)
        elif sp.issparse(v):
            d = self.sparse(self.refs, name, v)
        else:
            d = self.dense(self.refs, name, v)
        return d.ref

    def cell(self, loc, name, cells):
        rows, cols = cells.shape
        refs = np.empty((cols, rows), dtype=h5py.ref_dtype)  # transposed
        for i in range(rows):
            for j in range(cols):
                refs[j, i] = self.element(cells[i, j])
        d = loc.create_dataset(name, data=refs, dtype=h5py.ref_dtype)
        d.attrs['MATLAB_class'] = np.bytes_(b'cell')
        return d


def write_v73(path):
    blk, At, C, b = problem()
    with h5py.File(path, 'w', userblock_size=512) as f:
        w = V73Writer(f)
        w.cell(f, 'blk', blk)
        w.cell(f, 'At', At)
        w.cell(f, 'C', C)
        w.dense(f, 'b', b)
    header = b'MATLAB 7.3 MAT-file, Platform: cuADMM make_sdpt3_testfiles.py, Created on: 2026 HDF5 schema 1.00 .'
    header = header.ljust(116, b' ') + b'\x00' * 8 + b'\x00\x02' + b'IM'
    header = header.ljust(512, b'\x00')
    with open(path, 'r+b') as f:
        f.seek(0)
        f.write(header)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    write_v5(os.path.join(out, 'sdpt3_small_v5.mat'))
    write_v73(os.path.join(out, 'sdpt3_small_v73.mat'))
    print('wrote', os.path.join(out, 'sdpt3_small_v5.mat'), 'and sdpt3_small_v73.mat')
