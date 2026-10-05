#!/usr/bin/env python3
"""Dataset inventory of the 26 Kocvara sparse structural-optimization SDPs.

Parses the published objectives from the saved collection page (data/sources/kocvara_problems.html), records the
download provenance (source URLs, times, archive and file SHA-256), the SDPA structure and the converted cuADMM sizes
(meta.json of each converted instance), the conversion-check results, the byte-identity with the PLATO mirror, and
memory estimates. Writes data/published_objectives.json and dataset_inventory.csv."""
import csv, hashlib, html, json, os, re, zipfile

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.expanduser('~/cuadmm-data/plato_kocvara')
LOG = os.path.join(D, 'download_log.tsv')

# published objectives: rows "problem n m objective" of the collection page
t = open(os.path.join(D, 'docs', 'problems.html'), errors='replace').read()
txt = html.unescape(re.sub(r'<[^>]+>', ' ', t))
txt = re.sub(r'\s+', ' ', txt)
ref = {}
for m in re.finditer(r'\b((?:mater-|trto|buck|vibra|shmup)\d)\s+(\d+)\s+(\d+(?:\+\d+)?)\s+([-+]?\d[\d.]*(?:e[-+]?\d+)?)(\s*\(exact value\))?', txt):
    name, n, msize, val, exact = m.groups()
    mant = val.split('e')[0].lstrip('+-').replace('.', '').lstrip('0')
    ref[name] = dict(objective=float(val), text=val, exact=bool(exact), digits=None if exact else len(mant), n=int(n), m_published=msize,
                     # half a unit in the last published digit, relative to 1 + |ref| (0 for exact values)
                     resolution=0.0 if exact else (0.5 * 10 ** (int(f'{float(val):e}'.split('e')[1]) - (len(mant) - 1))) / (1 + abs(float(val))))
assert len(ref) == 26, sorted(ref)
json.dump(ref, open(os.path.join(EXP, 'data', 'published_objectives.json'), 'w'), indent=1)

dl = {}
for line in open(LOG):
    code, url, when, sha, size = line.rstrip('\n').split('\t')
    dl[os.path.basename(url)] = dict(url=url, time=when, sha256=sha, bytes=int(size), http=code[-3:])
rows = []
for fam in ('buck', 'mater', 'shmup', 'trto', 'vibra'):
    z = zipfile.ZipFile(os.path.join(D, 'raw', fam + '.zip'))
    for info in z.infolist():
        name = info.filename[:-len('.dat-s')]
        data = z.read(info.filename)
        meta = json.load(open(os.path.join(D, 'txt', name, 'meta.json')))
        chk = json.load(open(os.path.join(D, 'txt', name, 'conversion_check.json')))
        mirror = dl.get(name + '.dat-s.gz')
        psd = [b for b in meta['sdpa_blocks'] if b > 1]
        dense_psd = sum(b * b for b in psd)
        rows.append({
            'dataset': name, 'family': fam, 'source_archive_url': dl[fam + '.zip']['url'], 'download_time': dl[fam + '.zip']['time'],
            'archive_sha256': dl[fam + '.zip']['sha256'], 'original_filename': info.filename, 'file_bytes': info.file_size,
            'file_sha256': hashlib.sha256(data).hexdigest(), 'file_date_in_archive': '%04d-%02d-%02d' % info.date_time[:3],
            'license': 'none stated on the source page (academic test problems; cite Kocvara and the references on the page)',
            'plato_mirror': 'identical bytes' if mirror else 'not mirrored on PLATO',
            'on_plato_benchmark_page': name in ('buck5', 'mater-6', 'shmup4', 'shmup5', 'trto4', 'trto5', 'vibra4', 'vibra5'),
            'constraints_m': meta['m'], 'sdpa_nblock': meta['nblock'], 'sdpa_block_structure': ' '.join(str(b) for b in meta['sdpa_blocks'])
            if len(meta['sdpa_blocks']) <= 12 else f"{len(meta['sdpa_blocks'])} blocks: " + ', '.join(f'{meta["sdpa_blocks"].count(v)} x {v}' for v in sorted(set(meta['sdpa_blocks']), reverse=True)),
            'psd_block_sizes': ' '.join(str(v) for v in sorted(set(psd), reverse=True)), 'n_psd_blocks': len(psd), 'max_psd_block': max(psd) if psd else 0,
            'lp_blocks': ' '.join(str(v) for v in meta['lp_block_sizes']) or 'none', 'one_by_one_psd_blocks_as_l': meta['n_1x1_psd_blocks'],
            'free_variables': 'none', 'cuadmm_vec_len': meta['vec_len'], 'nnz_A': meta['nnz_At'], 'nnz_C': meta['nnz_C'], 'nnz_b': meta['nnz_b'],
            'explicit_zeros_dropped': meta['zeros_dropped'], 'lower_triangle_entries_mirrored': meta['lower_mirrored'],
            'duplicate_entries': meta['duplicates'], 'blocks_untouched_by_A': len(meta['untouched_blocks']),
            'est_gpu_mib_dense_psd': round(8 * dense_psd * 8 / 2 ** 20, 1),  # ~8 dense copies of every PSD block (projection work)
            'est_host_mib_problem': round((meta['nnz_At'] * 16 + meta['vec_len'] * 8 * 6) / 2 ** 20, 1),
            'full_row_rank': chk['checks']['full_row_rank']['ok'], 'rank_note': '' if chk['checks']['full_row_rank']['ok'] is not None
            else 'm > 6000: from the CHOLMOD factorization report of cuADMM init',
            'conversion_checks_passed': chk['all_ok'],
            'published_objective': ref[name]['objective'], 'published_objective_text': ref[name]['text'] + (' (exact value)' if ref[name]['exact'] else ''),
            'published_digits': ref[name]['digits'] if ref[name]['digits'] else 'exact', 'reference_resolution_rel': ref[name]['resolution'],
            'reference_source': 'PENSDP (Kocvara page); large problems may differ from SDPT3/MOSEK in the 5th-6th digit',
        })
with open(os.path.join(EXP, 'dataset_inventory.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['dataset']:8s} m={r['constraints_m']:6d} vec_len={r['cuadmm_vec_len']:7d} psd={r['psd_block_sizes'][:20]:20s} lp={r['lp_blocks'][:10]:10s} "
          f"nnzA={r['nnz_A']:8d} rank={r['full_row_rank']} ok={r['conversion_checks_passed']} ref={r['published_objective_text']}")
