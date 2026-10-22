#!/usr/bin/env python3
"""make_specs.py: the run specifications of the Phase-4b CPU screen (plan Section 3.5 Table 2, Phase 4b).

usage: make_specs.py stage1 <out.txt>                              factor x step_schedule x K_f, sigma_0 in {0.5, 1, 2}
       make_specs.py comparators <out.txt>                         old and new (residual_balancing) at every sigma_0, fixed sigma = 1
       make_specs.py stage2 <out.txt> <tag> [<tag> ...]            freeze_value x freeze_bias for the given Stage-1 tags, sigma_0 = 1
       make_specs.py sigma0 <out.txt> <sigma0,...> <tag> [<tag> ...]  the given settings at other sigma_0 values
spec line: <name> <dataset> <rule> <sigma0> <max_iter> <validate_interval>   (the format of the sigma-study pool)
run name: <dataset>_<tag>_s<sigma0>; tag: bf-f<factor>-<const|halve>-K<K_f>[-<centre|last>-b<bias>] | old | new | fixed
"""
import sys

PROBLEMS = ['trto1', 'trto2', 'buck1', 'buck2', 'vibra1', 'vibra2', 'mater-1', 'mater-2', 'shmup1']  # the nine small PLATO problems
CAP = 200000            # iteration cap of every run (the GPU gates use time caps; see docs/DECISIONS.md)
VINT = 50               # validation every 50 iterations (the sigma study's protocol)
SIGMA0 = ['0.5', '1', '2']
FACTORS = [('2', 'f2'), ('1.2', 'f1.2')]
SCHEDULES = [('constant', 'const'), ('halve_on_reversal', 'halve')]
KF = ['10000', '20000', '50000']
FREEZE_VALUES = [('band_centre', 'centre'), ('last', 'last')]
BIASES = ['1', '2', '4']


def stage1_tags():
    return ['bf-%s-%s-K%s' % (ft, st, k) for _, ft in FACTORS for _, st in SCHEDULES for k in KF]


def rule_of_tag(tag):
    """The replica's rule spec of a run tag."""
    if tag in ('old', 'new', 'fixed'):
        return tag
    parts = tag.split('-')
    assert parts[0] == 'bf' and len(parts) in (4, 6), tag
    factor = dict((t, v) for v, t in FACTORS)[parts[1]]
    sched = dict((t, v) for v, t in SCHEDULES)[parts[2]]
    assert parts[3][0] == 'K'
    kv = ['factor=' + factor, 'step_schedule=' + sched, 'freeze_iteration=' + parts[3][1:]]
    if len(parts) == 6:
        kv.append('freeze_value=' + dict((t, v) for v, t in FREEZE_VALUES)[parts[4]])
        assert parts[5][0] == 'b'
        kv.append('freeze_bias=' + parts[5][1:])
    return 'balance_freeze:' + ','.join(kv)


def line(ds, tag, sigma0):
    return '%s_%s_s%s %s %s %s %d %d' % (ds, tag, sigma0, ds, rule_of_tag(tag), sigma0, CAP, VINT)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    what, out = sys.argv[1], sys.argv[2]
    lines = []
    if what == 'stage1':
        lines.append('# Phase-4b CPU screen, Stage 1: factor x step_schedule x K_f at sigma_0 in {0.5, 1, 2}; cap %d, validation every %d' % (CAP, VINT))
        for tag in stage1_tags():
            for s in SIGMA0:
                for ds in PROBLEMS:
                    lines.append(line(ds, tag, s))
    elif what == 'comparators':
        lines.append('# Phase-4b CPU screen comparators: old (legacy_adaptive) and new (residual_balancing) at every sigma_0, fixed sigma = 1')
        for tag in ('old', 'new'):
            for s in SIGMA0:
                for ds in PROBLEMS:
                    lines.append(line(ds, tag, s))
        for ds in PROBLEMS:
            lines.append(line(ds, 'fixed', '1'))
    elif what == 'stage2':
        tags = sys.argv[3:]
        lines.append('# Phase-4b CPU screen, Stage 2: freeze_value x freeze_bias for %s at sigma_0 = 1' % ', '.join(tags))
        for tag in tags:
            assert len(tag.split('-')) == 4, tag
            for _, fv in FREEZE_VALUES:
                for b in BIASES:
                    if fv == 'centre' and b == '1':
                        continue  # the Stage-1 setting itself
                    for ds in PROBLEMS:
                        lines.append(line(ds, '%s-%s-b%s' % (tag, fv, b), '1'))
    elif what == 'sigma0':
        sigmas, tags = sys.argv[3].split(','), sys.argv[4:]
        lines.append('# Phase-4b CPU screen: %s at sigma_0 in {%s}' % (', '.join(tags), ', '.join(sigmas)))
        for tag in tags:
            for s in sigmas:
                for ds in PROBLEMS:
                    lines.append(line(ds, tag, s))
    else:
        print(__doc__)
        return 1
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('%s: %d runs' % (out, len(lines) - 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
