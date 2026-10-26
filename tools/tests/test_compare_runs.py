#!/usr/bin/env python3
"""Negative tests of tools/compare_runs.py: what the comparison must reject, and the two comparison modes.

usage: python -B tools/tests/test_compare_runs.py      (standard library only; no GPU, no solver build)

Each test builds two small run directories (history.csv, solution/{X,y,S}.txt, summary.json, exit_code.txt) and checks
the verdict of compare_dirs() and the exit code of the command line (0 identical, 1 different, 2 invalid).
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.normpath(os.path.join(HERE, "..", "compare_runs.py"))
_spec = importlib.util.spec_from_file_location("compare_runs", TOOL)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

COLUMNS = "iter,time_s,pobj,dobj,eta_p,eta_d,eta_g,sigma,tau,phase"


def history_rows(n=3):
    return [[k, 0.1 * k, -1.0 - 0.01 * k, -1.1 - 0.01 * k, 1e-3 / k, 2e-3 / k, 3e-3 / k, 100.0, 1.618, "admm"]
            for k in range(1, n + 1)]


def write_run(d, rows, X=(1.0, 2.0, 3.0), y=(0.5,), S=(0.25, 0.0), summary=None, exit_code="0", columns=COLUMNS,
              raw_history=None):
    os.makedirs(os.path.join(d, "solution"), exist_ok=True)
    with open(os.path.join(d, "history.csv"), "w") as f:
        if raw_history is not None:
            f.write(raw_history)
        else:
            f.write(columns + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")
    for name, v in (("X.txt", X), ("y.txt", y), ("S.txt", S)):
        if v is None:
            continue
        with open(os.path.join(d, "solution", name), "w") as f:
            f.write("\n".join(str(x) for x in v) + "\n")
    if summary != "absent":
        with open(os.path.join(d, "summary.json"), "w") as f:
            f.write(summary if isinstance(summary, str) else json.dumps({"iterations": len(rows), "stop_reason": "max_iter"}))
    if exit_code is not None:
        with open(os.path.join(d, "exit_code.txt"), "w") as f:
            f.write(exit_code + "\n")


def cli_exit(a, b, rtol=None):
    cmd = [sys.executable, "-B", TOOL, "compare", a, b]
    if rtol is not None:
        cmd += ["--rtol", repr(rtol)]
    return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class CompareRunsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="compare_runs_test_")
        self.a = os.path.join(self.tmp, "a")
        self.b = os.path.join(self.tmp, "b")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    # -- the positive case, for reference
    def test_identical_runs_are_identical(self):
        write_run(self.a, history_rows())
        write_run(self.b, history_rows())
        r = cr.compare_dirs(self.a, self.b)
        self.assertTrue(r["identical"])
        self.assertEqual(r["invalid"], [])
        self.assertEqual(cli_exit(self.a, self.b), 0)

    # -- nonfinite values are rejected, even when both runs carry the same one
    def test_nan_in_history_is_invalid(self):
        rows = history_rows()
        rows[1][6] = "nan"
        write_run(self.a, history_rows())
        write_run(self.b, rows)
        r = cr.compare_dirs(self.a, self.b)
        self.assertFalse(r["identical"])
        self.assertTrue(any("nonfinite" in m for m in r["invalid"]))
        self.assertEqual(cli_exit(self.a, self.b), 2)

    def test_same_nan_on_both_sides_is_still_invalid(self):
        rows = history_rows()
        rows[0][2] = "nan"
        write_run(self.a, rows)
        write_run(self.b, rows)
        self.assertEqual(cli_exit(self.a, self.b), 2)

    def test_inf_in_solution_is_invalid(self):
        write_run(self.a, history_rows())
        write_run(self.b, history_rows(), X=(1.0, "inf", 3.0))
        r = cr.compare_dirs(self.a, self.b)
        self.assertTrue(any("nonfinite" in m for m in r["invalid"]))
        self.assertEqual(cli_exit(self.a, self.b), 2)

    # -- mandatory fields
    def test_missing_mandatory_history_column_is_invalid(self):
        cols = "iter,time_s,pobj,dobj,eta_p,eta_d,eta_g,tau,phase"  # no sigma
        rows = [r[:7] + r[8:] for r in history_rows()]
        write_run(self.a, history_rows())
        write_run(self.b, rows, columns=cols)
        r = cr.compare_dirs(self.a, self.b)
        self.assertTrue(any("mandatory" in m and "sigma" in m for m in r["invalid"]))
        self.assertEqual(cli_exit(self.a, self.b), 2)

    def test_missing_iteration_column_is_invalid(self):
        cols = "time_s,pobj,dobj,eta_p,eta_d,eta_g,sigma,tau,phase"
        rows = [r[1:] for r in history_rows()]
        write_run(self.a, rows, columns=cols)
        write_run(self.b, rows, columns=cols)
        self.assertEqual(cli_exit(self.a, self.b), 2)

    def test_missing_solution_file_is_invalid(self):
        write_run(self.a, history_rows())
        write_run(self.b, history_rows(), S=None)
        self.assertEqual(cli_exit(self.a, self.b), 2)

    def test_missing_or_unparsable_summary_is_invalid(self):
        write_run(self.a, history_rows())
        write_run(self.b, history_rows(), summary="absent")
        self.assertEqual(cli_exit(self.a, self.b), 2)
        write_run(self.b, history_rows(), summary="{not json")
        self.assertEqual(cli_exit(self.a, self.b), 2)

    def test_failed_run_is_invalid(self):
        write_run(self.a, history_rows())
        write_run(self.b, history_rows(), exit_code="1")
        r = cr.compare_dirs(self.a, self.b)
        self.assertTrue(any("exited with code 1" in m for m in r["invalid"]))
        self.assertEqual(cli_exit(self.a, self.b), 2)

    def test_malformed_history_line_is_invalid(self):
        write_run(self.a, history_rows())
        write_run(self.b, [], raw_history=COLUMNS + "\n1,0.1,-1.0\n")
        self.assertEqual(cli_exit(self.a, self.b), 2)

    # -- iteration ids
    def test_shifted_iteration_ids_are_a_difference(self):
        rows = history_rows()
        for r in rows:
            r[0] += 1  # the same values under the ids 2, 3, 4
        write_run(self.a, history_rows())
        write_run(self.b, rows)
        r = cr.compare_dirs(self.a, self.b)
        self.assertFalse(r["identical"])
        self.assertEqual(r["invalid"], [])
        self.assertEqual(r["history"]["iteration_id_mismatch"]["row"], 1)
        self.assertEqual(r["history"]["first_difference"]["field"], "iter")
        self.assertEqual(cli_exit(self.a, self.b), 1)

    def test_first_difference_reports_the_iteration_id(self):
        rows = history_rows(5)
        for r in rows:
            r[0] *= 10  # ids 10, 20, 30, 40, 50 on both sides
        rows_b = [list(r) for r in rows]
        rows_b[2][2] = rows_b[2][2] * (1 + 1e-3)
        write_run(self.a, rows)
        write_run(self.b, rows_b)
        r = cr.compare_dirs(self.a, self.b)
        self.assertEqual(r["history"]["first_difference"]["iteration"], 30)
        self.assertEqual(r["history"]["first_difference"]["field"], "pobj")

    # -- bitwise versus tolerance
    def test_negative_zero_differs_bitwise_but_not_within_tolerance(self):
        write_run(self.a, history_rows(), X=(1.0, 0.0, 3.0))
        write_run(self.b, history_rows(), X=(1.0, -0.0, 3.0))
        self.assertFalse(cr.compare_dirs(self.a, self.b)["identical"])
        self.assertEqual(cli_exit(self.a, self.b), 1)
        self.assertTrue(cr.compare_dirs(self.a, self.b, rtol=1e-6)["identical"])
        self.assertEqual(cli_exit(self.a, self.b, rtol=1e-6), 0)

    def test_last_bit_differs_bitwise_but_not_within_tolerance(self):
        rows_b = history_rows()
        rows_b[0][2] = repr(float(rows_b[0][2]) * (1 + 2.3e-16))  # one ulp
        write_run(self.a, history_rows())
        write_run(self.b, rows_b)
        self.assertFalse(cr.compare_dirs(self.a, self.b)["identical"])
        self.assertTrue(cr.compare_dirs(self.a, self.b, rtol=1e-6)["identical"])

    def test_tolerance_mode_still_rejects_a_real_difference(self):
        rows_b = history_rows()
        rows_b[1][4] = repr(float(rows_b[1][4]) * (1 + 1e-5))
        write_run(self.a, history_rows())
        write_run(self.b, rows_b)
        r = cr.compare_dirs(self.a, self.b, rtol=1e-6)
        self.assertFalse(r["identical"])
        self.assertEqual(r["history"]["first_difference"]["field"], "eta_p")
        self.assertEqual(cli_exit(self.a, self.b, rtol=1e-6), 1)

    def test_different_lengths_are_a_difference(self):
        write_run(self.a, history_rows(3))
        write_run(self.b, history_rows(4))
        r = cr.compare_dirs(self.a, self.b)
        self.assertFalse(r["identical"])
        self.assertEqual(r["history"]["note"], "different number of iterations")
        self.assertEqual(cli_exit(self.a, self.b), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
