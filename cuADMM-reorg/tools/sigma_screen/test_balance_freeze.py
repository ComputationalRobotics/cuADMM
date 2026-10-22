#!/usr/bin/env python3
"""Tests of the v1.2 additions of cuadmm_emu.py: the balance_freeze rule (plan Section 3.5) as BalanceFreezeState, its
use by the Emulator, the rule-spec parser, the sigma_0 clamp and the checkpoint round trip.

run: python -B test_balance_freeze.py   (about a minute on one core; run it through Slurm, never on a login node)
The unchanged rules (old, new, fixed, oldnom1, newfreeze, newdb) are covered by the v1.1 suite of the sigma study,
run against this directory:  python -B <sigma study>/emu-run/test_emu_rules.py <this directory>.
The C++ policy is checked against BalanceFreezeState by test/penalty_policy_test.hpp (BalanceFreezeAgreesWithTheReplica*).
"""
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cuadmm_emu as ce  # noqa: E402

DATA = '/n/home00/yukuanwei/cuadmm-data/plato_kocvara/txt'
UP, DOWN = (1e-3, 1e-1), (1e-1, 1e-3)  # (errRp, errRd): the primal wins (sigma up) / the dual wins (sigma down)


def _same_run(a, b):
    assert all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ('X', 'y', 'S', 'Rp', 'SmC'))
    assert a.sig == b.sig and a.prim_win == b.prim_win and a.dual_win == b.dual_win and a.iter == b.iter
    assert len(a.sigma_log) == len(b.sigma_log)
    for r, s in zip(a.sigma_log, b.sigma_log):
        assert r == s, (r, s)
    assert len(a.validations) == len(b.validations)
    for r, s in zip(a.validations, b.validations):
        for k in r:
            if k in ('time_s', 'validation_s', 'callback_s'):
                continue
            x, y = r[k], s[k]
            assert x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y)), (k, x, y)
    assert np.array_equal(a.history.array(), b.history.array())
    assert a.bf.state() == b.bf.state()


def test_parse_rule():
    r = ce.parse_rule('balance_freeze')
    assert r.freeze and r.adaptive and not r.monitor1 and r.name == 'balance_freeze'
    assert (r.factor, r.step_schedule, r.freeze_iteration, r.pingpong_window, r.pingpong_net, r.pingpong_min_iter,
            r.freeze_value, r.freeze_bias, r.sigmin, r.sigmax, r.guard, r.ratio) == \
        (2.0, 'constant', 10000, 8, 2, 1000, 'band_centre', 1.0, 1e-3, 1e3, 1.0, 1.35)
    assert r.freeze_after is None and r.deadband is None and r.clamp_sigma0
    r = ce.parse_rule('balance_freeze:freeze_bias=2,factor=1.2,step_schedule=halve_on_reversal,freeze_iteration=-1,sigma_min=0.01')
    assert r.name == 'balance_freeze:factor=1.2,freeze_bias=2,freeze_iteration=-1,sigma_min=0.01,step_schedule=halve_on_reversal'
    assert (r.factor, r.step_schedule, r.freeze_bias, r.sigmin, r.sigmax) == (1.2, 'halve_on_reversal', 2.0, 0.01, 1e3)
    assert r.freeze_iteration == 10000  # a negative option means the policy's own value, as in the C++
    assert r.to_dict()['freeze'] is True and r.to_dict()['freeze_iteration'] == 10000
    for bad in ('balance_freeze:foo=1', 'new:factor=2', 'nope', 'balance_freeze:step_schedule=linear', 'balance_freeze:freeze_value=mean'):
        try:
            ce.parse_rule(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert ce.RULES['balance_freeze']().to_dict() == ce.parse_rule('balance_freeze').to_dict()
    for name in ('old', 'new', 'fixed', 'oldnom1', 'newfreeze', 'newdb'):
        assert ce.parse_rule(name).to_dict() == ce.RULES[name]().to_dict() and not ce.RULES[name]().freeze


def test_freeze_at_freeze_iteration():
    st = ce.BalanceFreezeState(ce.parse_rule('balance_freeze:freeze_iteration=100,pingpong_min_iter=1000000'), 1.0)
    sig = [st.update(k, *UP, 1e-3) for k in range(1, 201)]  # the primal wins every iteration
    # checks at 1, 11, 21, ...: no change at k = 1 (prim_win = 1 is not > max(1, 0)); x2 at 11, 21, ..., 91 (nine changes)
    assert sig[0] == 1.0 and sig[9] == 1.0 and sig[10] == 2.0 and sig[20] == 4.0 and sig[90] == 512.0
    assert st.frozen and st.freeze_at == 100 and st.freeze_reason == 'freeze_iteration'
    assert all(s == 512.0 for s in sig[99:])  # dirs = [+] * 9: net 9 > 2, no band centre; bias 1
    assert [e['rule'] for e in st.log] == ['admm_schedule'] * 9 + ['freeze_freeze_iteration']
    assert st.log[-1]['old_sigma'] == 512.0 and st.log[-1]['new_sigma'] == 512.0 and st.log[-1]['iter'] == 100
    assert st.log[0] == dict(iter=11, old_sigma=1.0, new_sigma=2.0, rule='admm_schedule', errRp=1e-3, errRd=1e-1, relgap=1e-3,
                             feasratio=1e-3 / 1e-1, prim_win=11, dual_win=0)
    # frozen: no vote, no change; the counters stay as they were at the freeze (prim_win: the votes of 92..100)
    assert st.update(201, *DOWN, 1e-3) == 512.0 and (st.prim_win, st.dual_win) == (9, 0) and len(st.log) == 10


def _drive_pattern(spec, sig0, n, primal_of_k):
    st = ce.BalanceFreezeState(ce.parse_rule(spec), sig0)
    out = []
    for k in range(1, n + 1):
        out.append(st.update(k, *(UP if primal_of_k(k) else DOWN), 1e-3))
    return st, out


def _plus_plus_minus_minus(k):  # primal votes 1..21, dual 22..41, primal 42..61, dual 62..81, ...
    return k == 1 or ((k - 2) // 20) % 2 == 0


def test_ping_pong_band_centre():
    st, sig = _drive_pattern('balance_freeze:pingpong_min_iter=1,freeze_iteration=1000000', 1.0, 200, _plus_plus_minus_minus)
    # changes at 11, 21 (+, +), 31, 41 (-, -), 51, 61 (+, +), 71, 81 (-, -): net movement 0 over the last 8 -> ping-pong at 81
    assert st.dirs == [1, 1, -1, -1, 1, 1, -1, -1] and st.sigmas == [2.0, 4.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0]
    assert st.frozen and st.freeze_at == 81 and st.freeze_reason == 'ping_pong'
    centre = math.exp(sum(math.log(v) for v in st.sigmas) / 8)  # 2 exactly up to rounding
    assert sig[80] == centre and abs(centre - 2.0) < 1e-12 and all(s == centre for s in sig[80:])
    assert [e['rule'] for e in st.log][-2:] == ['admm_schedule', 'freeze_ping_pong'] and st.log[-1]['old_sigma'] == 1.0
    assert st.log[-1]['new_sigma'] == centre and st.log[-1]['iter'] == 81
    # the freeze value options
    st, sig = _drive_pattern('balance_freeze:pingpong_min_iter=1,freeze_iteration=1000000,freeze_value=last,freeze_bias=4', 1.0, 100,
                             _plus_plus_minus_minus)
    assert st.frozen and st.freeze_reason == 'ping_pong' and sig[80] == 4.0 and st.sig == 4.0
    st, sig = _drive_pattern('balance_freeze:pingpong_min_iter=1,freeze_iteration=1000000,freeze_bias=0.5', 1.0, 100, _plus_plus_minus_minus)
    assert st.frozen and st.sig == 0.5 * centre
    # before k_min no ping-pong freeze: with k_min = 1000 the pattern keeps bouncing until 1000; the ping-pong test runs
    # at every update (not only at the checks), so the freeze comes at the first k >= 1000 with |net| <= 2
    st, sig = _drive_pattern('balance_freeze:freeze_iteration=1000000', 1.0, 1100, _plus_plus_minus_minus)
    assert st.frozen and st.freeze_reason == 'ping_pong' and 1000 <= st.freeze_at <= 1100 and len(st.dirs) > 8
    assert abs(sum(st.dirs[-8:])) <= 2 and all(a == b for a, b in zip(sig[:999], sig[:999]))
    changes_before = [e for e in st.log if e['iter'] < 1000 and e['rule'] == 'admm_schedule']
    assert len(changes_before) == len(st.dirs) - sum(1 for e in st.log if 1000 <= e['iter'] and e['rule'] == 'admm_schedule')
    assert not any(e['rule'].startswith('freeze') and e['iter'] < 1000 for e in st.log)
    # steady drift never triggers it (net movement W)
    st, sig = _drive_pattern('balance_freeze:pingpong_min_iter=1,freeze_iteration=1000000', 1.0, 200, lambda k: True)
    assert not st.frozen and st.sig == 1e3  # clamped at sigma_max after 10 doublings
    # a stricter window: W = 4, net 0 freezes after +, +, -, -
    st, sig = _drive_pattern('balance_freeze:pingpong_min_iter=1,freeze_iteration=1000000,pingpong_window=4,pingpong_net=0', 1.0, 100,
                             _plus_plus_minus_minus)
    assert st.frozen and st.freeze_at == 41 and st.dirs == [1, 1, -1, -1] and abs(st.sig - 2.0) < 1e-12


def test_halve_on_reversal():
    spec = 'balance_freeze:step_schedule=halve_on_reversal,pingpong_min_iter=1000000,freeze_iteration=1000000'
    alternate = lambda k: k == 1 or ((k - 2) // 10) % 2 == 0  # +, -, +, -, ... (one change per check from k = 11)

    def expected(factor):
        # the same arithmetic as the rule: the step is square-rooted at every reversal, the freeze "step" comes with the
        # first change whose log2(step) < 1/16 (2 -> sqrt 2 -> 2^(1/4) -> 2^(1/8) -> 2^(1/16): the boundary case 1/16 is
        # decided by the rounding of the square roots, so the expectation mirrors the arithmetic instead of fixing a count)
        step, s, d, sig, dirs = factor, 1.0, 1, [], []
        while True:
            if dirs:
                step = math.sqrt(step)
            s = s * step if d > 0 else s / step
            sig.append(s)
            dirs.append(d)
            if math.log2(step) < 1.0 / 16.0:
                return sig, dirs, step
            d = -d

    for factor, n_changes in ((2.0, None), (1.5, 5), (1.2, 4)):
        # 1.5: steps 1.5, 1.2247, 1.1067, 1.0520 (log2 0.073), 1.0257 (log2 0.037 < 1/16): five changes, unambiguous
        # 1.2: steps 1.2, 1.0954, 1.0466 (log2 0.0658), 1.0231 (log2 0.0329): four changes
        st, sig = _drive_pattern(spec + ',factor=%g' % factor, 1.0, 200, alternate)
        exp_sig, exp_dirs, exp_step = expected(factor)
        if n_changes is not None:
            assert len(exp_sig) == n_changes, (factor, len(exp_sig))
        assert st.sigmas == exp_sig and st.dirs == exp_dirs and st.step == exp_step, factor
        n = len(exp_sig)
        assert st.frozen and st.freeze_reason == 'step' and st.freeze_at == 11 + 10 * (n - 1), factor
        assert st.sig == exp_sig[-1] and sig[st.freeze_at - 1] == exp_sig[-1] and all(v == exp_sig[-1] for v in sig[st.freeze_at - 1:])
        assert [e['rule'] for e in st.log] == ['admm_schedule'] * n + ['freeze_step'], factor
        assert st.log[-1]['old_sigma'] == exp_sig[-1] and st.log[-1]['new_sigma'] == exp_sig[-1]  # n < W: no band centre
    # with the constant schedule the same votes never shrink the step
    st2, _ = _drive_pattern('balance_freeze:pingpong_min_iter=1000000,freeze_iteration=1000000', 1.0, 200, alternate)
    # checks at 1, 11, ..., 191: no change at k = 1, then one change per check: 19 changes
    assert st2.step == 2.0 and not st2.frozen and len(st2.sigmas) == 19 and st2.sigmas == ([2.0, 1.0] * 10)[:19]
    # the step freeze with a band: W = 4 and eight alternating changes before the step is small enough (factor 2^8)
    st3, _ = _drive_pattern('balance_freeze:step_schedule=halve_on_reversal,pingpong_min_iter=1000000,freeze_iteration=1000000,'
                            'pingpong_window=4,pingpong_net=4,factor=256', 1.0, 200, alternate)
    assert st3.frozen and st3.freeze_reason == 'step' and len(st3.dirs) >= 4
    centre = math.exp(sum(math.log(v) for v in st3.sigmas[-4:]) / 4)
    assert st3.sig == centre and st3.log[-1]['new_sigma'] == centre


def test_sigma0_clamp_and_bounds():
    assert ce.BalanceFreezeState(ce.parse_rule('balance_freeze'), 5000.0).sig == 1e3
    assert ce.BalanceFreezeState(ce.parse_rule('balance_freeze'), 1e-5).sig == 1e-3
    assert ce.BalanceFreezeState(ce.parse_rule('balance_freeze:sigma_min=0.1,sigma_max=10'), 0.01).sig == 0.1
    assert ce.BalanceFreezeState(ce.parse_rule('balance_freeze'), 7.5).sig == 7.5
    e = ce.Emulator(ce.Problem(os.path.join(DATA, 'trto1')), rule=ce.parse_rule('balance_freeze'), sig0=5000.0, validate_interval=0)
    assert e.sig == 1e3 and e.sig0 == 5000.0 and e.bf is not None and e.bf.sig == 1e3
    e = ce.Emulator(ce.Problem(os.path.join(DATA, 'trto1')), rule='new', sig0=5000.0, validate_interval=0)
    assert e.sig == 5000.0 and e.bf is None  # residual_balancing does not clamp sigma_0 (2d8a755)


def test_emulator_uses_the_state_object():
    spec = 'balance_freeze:freeze_iteration=2000,pingpong_min_iter=1000000'  # K_f before any ping-pong freeze
    e = ce.Emulator(ce.Problem(os.path.join(DATA, 'trto1')), rule=ce.parse_rule(spec), sig0=1.0, validate_interval=0, history_stride=1)
    e.run(pause_at=3000)
    h = e.history.array()  # (iter, errRp, errRd, relgap, pobj, dobj, sig, prim_win, dual_win)
    st = ce.BalanceFreezeState(ce.parse_rule(spec), 1.0)
    for row in h:
        assert st.update(int(row[0]), row[1], row[2], row[3]) == row[6], int(row[0])
        assert (st.prim_win, st.dual_win) == (int(row[7]), int(row[8]))
    assert e.bf.state() == st.state() and e.sigma_log == st.log and e.sigma_changes == sum(1 for r in e.sigma_log if r['new_sigma'] != r['old_sigma'])
    assert e.bf.frozen and e.bf.freeze_at == 2000 and e.summary()['freeze_iteration'] == 2000 and e.summary()['freeze_reason']
    assert np.all(h[h[:, 0] >= 2000, 6] == e.sig) and e.summary()['rule']['freeze'] is True
    print('    trto1 balance_freeze K_f 2000: %d changes, frozen at %d (%s), sigma %g' % (len(e.sigma_log) - 1, e.bf.freeze_at,
                                                                                        e.bf.freeze_reason, e.sig))


def test_resume_exact():
    d = os.path.join(DATA, 'vibra1')
    kw = dict(rule=ce.parse_rule('balance_freeze:freeze_iteration=2500,pingpong_min_iter=1000000'), sig0=1.0, validate_interval=50,
              strict_dimacs_tol=1e-6, history_stride=1)
    a = ce.Emulator(ce.Problem(d), **kw)
    a.run(pause_at=3000)
    b = ce.Emulator(ce.Problem(d), **kw)
    b.run(pause_at=1234)
    assert not b.bf.frozen and len(b.bf.dirs) > 0
    with tempfile.TemporaryDirectory() as t:
        b.save_state(os.path.join(t, 's.npz'))
        c = ce.Emulator.load_state(os.path.join(t, 's.npz'))
    assert c.bf.state() == b.bf.state() and c.rule.to_dict() == b.rule.to_dict() and c.sig == b.sig
    c.run(pause_at=3000)
    _same_run(a, c)
    assert a.bf.frozen and a.bf.freeze_at == 2500
    # a checkpoint after the freeze keeps the frozen state
    with tempfile.TemporaryDirectory() as t:
        a.save_state(os.path.join(t, 's.npz'))
        f = ce.Emulator.load_state(os.path.join(t, 's.npz'))
    assert f.bf.frozen and f.bf.freeze_at == 2500 and f.bf.sig == a.sig and f.summary()['freeze_iteration'] == 2500


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print('ok', t.__name__, flush=True)
    print('all %d tests passed' % len(tests))
