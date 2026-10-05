#!/usr/bin/env python3
"""Convergence plots from the validation histories (external validator values, not internal residuals):
eta and max_abs_DIMACS against the solve clock, one panel per dataset, every configuration of a stage (rep 1).
Also sigma against the iteration for the adaptive runs (from the per-iteration history).
usage: plots.py <stage>   -> results/plots/<stage>_eta.png, <stage>_dimacs.png, <stage>_sigma.png"""
import csv, glob, gzip, json, math, os, sys
import matplotlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIG = os.path.expanduser('~/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma')
INV = {r['dataset']: r for r in csv.DictReader(open(os.path.join(EXP, 'dataset_inventory.csv')))}
COLORS = {'adaptive': 'tab:red', 'fixed': 'tab:blue', 'tuned': 'tab:green'}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def runs(stage):
    out = {}
    for d in sorted(glob.glob(os.path.join(EXP, 'runs', stage, '*'))):
        if not os.path.exists(os.path.join(d, 'run.json')):
            continue
        r = json.load(open(os.path.join(d, 'run.json')))
        if r['rep'] != 1:
            continue
        vh = A.read_csv(os.path.join(d, 'validation_history.csv'))
        out.setdefault(r['dataset'], []).append((r, vh))
    return out


def grid(n):
    c = min(6, n)
    return (n + c - 1) // c, c


def panel_plot(stage, data, key, ylabel, fname, thresholds):
    ds = sorted(data, key=lambda d: int(INV[d]['cuadmm_vec_len']))
    nr, nc = grid(len(ds))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.6 * nr), squeeze=False)
    for ax, d in zip(axes.flat, ds):
        for r, vh in sorted(data[d], key=lambda x: x[0]['config']):
            t = [f(v['time_s']) + f(v['validation_s']) for v in vh]
            y = [max(f(v['primal_res']), f(v['dual_res']), f(v['relgap'])) if key == 'eta' else f(v['max_abs_dimacs']) for v in vh]
            base = r['config'] if r['config'] in COLORS else None
            lab = r['config'] if base else f"{r['config']}"
            ax.semilogy(t, y, '-', lw=1, color=COLORS.get(base), label=lab)
        for th in thresholds:
            ax.axhline(th, color='gray', lw=0.6, ls=':')
        ax.set_title(d, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(ds):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=6)
    fig.supxlabel('solve clock (s), at validator confirmation', fontsize=9)
    fig.supylabel(ylabel, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.join(EXP, 'results', 'plots'), exist_ok=True)
    fig.savefig(os.path.join(EXP, 'results', 'plots', fname), dpi=130)
    plt.close(fig)


def sigma_plot(stage, data):
    ds = sorted(data, key=lambda d: int(INV[d]['cuadmm_vec_len']))
    nr, nc = grid(len(ds))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.4 * nr), squeeze=False)
    for ax, d in zip(axes.flat, ds):
        for r, _ in data[d]:
            if r['policy'] != 'legacy_adaptive':
                continue
            p = os.path.join(BIG, 'runs', stage, r['run_id'], 'history.csv.gz')
            if not os.path.exists(p):
                continue
            it, sg = [], []
            with gzip.open(p, 'rt') as fh:
                for k, row in enumerate(csv.DictReader(fh)):
                    if k % 10 == 0:
                        it.append(int(row['iter']))
                        sg.append(f(row['sig']))
            ax.semilogy(it, sg, color='tab:red', lw=1)
        ax.set_title(d, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(ds):]:
        ax.axis('off')
    fig.supxlabel('iteration', fontsize=9)
    fig.supylabel('sigma (adaptive)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, 'results', 'plots', f'{stage}_sigma.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    stage = sys.argv[1]
    data = runs(stage)
    panel_plot(stage, data, 'eta', 'eta (external)', f'{stage}_eta.png', (1e-4,))
    panel_plot(stage, data, 'dimacs', 'max |DIMACS error| (external)', f'{stage}_dimacs.png', (1e-4, 1e-6))
    sigma_plot(stage, data)
    print('plots written to results/plots/')
