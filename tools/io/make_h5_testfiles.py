#!/usr/bin/env python3
"""Writes native-format (.h5, schema 1) test files of test/data with h5py, the way Python and MATLAB users would
write them (plan Section 12): string attributes as variable-length UTF-8 (h5py's default for str), fixed-length
ASCII (numpy bytes, MATLAB-like), variable-length ASCII and fixed-length UTF-8; integer datasets as int32
(MATLAB-like); and malformed files for the robustness tests (an svec index out of range, a negative index, a C
index out of range, a missing sense attribute).

The problem: blocks s 2 and l 2 (vec_len 5), m = 2, At = [(0,0) 1, (3,0) 2, (2,1) 3, (4,1) 4], b = [1, 2],
C = [(0) 1, (3) 1], sense min, offset 0.5, with a report line and DIMACS kinds in /meta.
usage: make_h5_testfiles.py <output directory>
"""
import os
import sys

import h5py
import numpy as np


def sattr(obj, name, value, mode):
    if mode == 'vlen_utf8':
        obj.attrs[name] = value
    elif mode == 'fixed_ascii':
        obj.attrs[name] = np.bytes_(value)
    elif mode == 'vlen_ascii':
        obj.attrs.create(name, data=value.encode(), dtype=h5py.string_dtype('ascii'))
    elif mode == 'fixed_utf8':
        obj.attrs.create(name, data=value.encode(), dtype=h5py.string_dtype('utf-8', len(value)))
    else:
        raise ValueError(mode)


def write(path, mode='vlen_utf8', int_dtype=np.int64, rowind_bad=None, c_bad=None, sense=True, report_mode='vlen'):
    with h5py.File(path, 'w') as f:
        sattr(f, 'format', 'cuadmm', mode)
        f.attrs['version'] = np.int64(1)
        if sense:
            sattr(f, 'sense', 'min', mode)
        f.attrs['objective_offset'] = 0.5
        sattr(f, 'svec', 'upper, column-major, off-diagonal x sqrt(2)', mode)
        sattr(f, 'source', 'make_h5_testfiles.py', mode)
        g = f.create_group('blocks')
        g.create_dataset('kind', data=np.array([ord('s'), ord('l')], dtype=np.uint8))
        g.create_dataset('size', data=np.array([2, 2], dtype=int_dtype))
        a = f.create_group('At')
        rowind = [0, 3, 2, 4]
        if rowind_bad is not None:
            rowind[2] = rowind_bad
        a.create_dataset('colptr', data=np.array([0, 2, 4], dtype=int_dtype))
        a.create_dataset('rowind', data=np.array(rowind, dtype=np.int64 if rowind_bad is not None else int_dtype))
        a.create_dataset('val', data=np.array([1.0, 2.0, 3.0, 4.0]))
        b = f.create_group('b')
        b.create_dataset('index', data=np.array([0, 1], dtype=int_dtype))
        b.create_dataset('val', data=np.array([1.0, 2.0]))
        c = f.create_group('C')
        c.create_dataset('index', data=np.array([0, 3 if c_bad is None else c_bad], dtype=np.int64 if c_bad is not None else int_dtype))
        c.create_dataset('val', data=np.array([1.0, 1.0]))
        m = f.create_group('meta')
        m.create_dataset('dimacs_kinds', data=np.array([ord('s'), ord('l')], dtype=np.uint8))
        if report_mode == 'vlen':
            m.create_dataset('report', data=np.array(['written by h5py', 'second line'], dtype=h5py.string_dtype('utf-8')))
        else:
            m.create_dataset('report', data=np.array([b'written by h5py', b'second line'], dtype='S16'))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    write(os.path.join(out, 'h5_vlen_utf8.h5'), 'vlen_utf8')
    write(os.path.join(out, 'h5_fixed_ascii.h5'), 'fixed_ascii', report_mode='fixed')
    write(os.path.join(out, 'h5_vlen_ascii.h5'), 'vlen_ascii')
    write(os.path.join(out, 'h5_fixed_utf8.h5'), 'fixed_utf8')
    write(os.path.join(out, 'h5_int32.h5'), 'fixed_ascii', int_dtype=np.int32, report_mode='fixed')
    write(os.path.join(out, 'h5_bad_oob.h5'), rowind_bad=2 ** 40)
    write(os.path.join(out, 'h5_bad_oob_small.h5'), rowind_bad=5)
    write(os.path.join(out, 'h5_bad_neg.h5'), rowind_bad=-1)
    write(os.path.join(out, 'h5_bad_C_oob.h5'), c_bad=2 ** 40)
    write(os.path.join(out, 'h5_no_sense.h5'), sense=False)
    print('wrote the h5 test files to', out)
