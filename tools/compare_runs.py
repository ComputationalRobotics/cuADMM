#!/usr/bin/env python3
"""
compare_runs.py: bitwise comparison of cuADMM runs (the "Comparison script" of the reorganization's developer
prompt, reorg-plan/ in the history of main).

Runs a reference build and the new build (or any two builds) with equivalent options on one TXT problem, or compares
two existing run directories, and reports the first iteration and field that differ.

"Bitwise" (developer prompt): identical per-iteration history (pobj, dobj, eta_p, eta_d, eta_g, sigma) and identical
returned X, y, S. The histories must be written with 17 significant digits: the reference CLIs print 10, so the
reference worktrees carry the printing-only patch tools/reference/print17.patch (main.cu: setprecision(10) -> 17 in
the history, the JSON summary and json_number). --save-solution already writes 17 digits.

Builds are named KIND:EXE where KIND is
    legacy@9bb4ba5    reference for `fixed` and `legacy_adaptive` (the historical rule, Monitor1 included)
    legacy@2d8a755    reference for `fixed` and `residual_balancing` (its `--sigma-policy legacy_adaptive` is the
                      October 13 rule: guard max(1, 1.35 other), factor 2, sigma in [1e-3, 1e3], no Monitor1)
    legacy@2d8a755-sgs  the same source built without CUADMM_PLAIN_ADMM_ONLY: reference for scheme sgs_admm
                      (tools/reference/sgs_no_plain_only.patch)
    legacy@c23e67e    the main commit the reorganization starts from (Phase 0 copy): only `fixed` is comparable,
                      its adaptive rule (factor 1.2) is neither reference policy
    legacy@pre0       the copied CLI of the early reorganization after Pre-0 (policies fixed, legacy_adaptive,
                      residual_balancing under their plan names; --penalty-factor); history only
    reorg             the command-line interface of cuADMM 2.0 (cuadmm_exe; Phase 2 and later)
The legacy CLIs take the old flag names; `reorg` takes the new ones (today's flags are accepted as aliases).

Run directory layout (both CLIs write it): history.csv, summary.json, solution/{X,y,S}.txt, sigma_log.csv,
command.txt, stdout.txt.

usage:
  compare_runs.py run --build-a KIND:EXE --build-b KIND:EXE --problem DIR --out DIR [case options]
  compare_runs.py selfcheck --build KIND:EXE --problem DIR --out DIR [case options]     # the build against itself
  compare_runs.py compare DIR_A DIR_B [--rtol X] [--report FILE]
  compare_runs.py batch --spec CASES.json --build-a KIND:EXE --build-b KIND:EXE --out DIR [--only NAME ...]
case options: --policy {fixed,legacy_adaptive,residual_balancing,balance_freeze} --sigma0 X --max-iter N --tol X
              --scheme {admm,sgs_admm} [--tau X] [--streams N] [--lobpcg 0|1] [--switch-proj-iter N]
              [--switch-proj-tol X] [--time-limit S] [--factor X] [--extra-a "..."] [--extra-b "..."]
Exit code: 0 identical (within --rtol), 1 differences found, 2 a run failed or the directories are unusable.
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time

HISTORY_FIELDS = ["pobj", "dobj", "eta_p", "eta_d", "eta_g", "sigma"]  # compared bitwise
HISTORY_INFO_FIELDS = ["tau", "phase"]  # compared, reported, but informational
COLUMN_ALIASES = {
    "errRp": "eta_p", "errRd": "eta_d", "relgap": "eta_g", "sig": "sigma",
    "eta_p": "eta_p", "eta_d": "eta_d", "eta_g": "eta_g", "sigma": "sigma",
    "pobj": "pobj", "dobj": "dobj", "iter": "iter", "time_s": "time_s", "tau": "tau", "phase": "phase",
}
SOLUTION_FILES = ["X.txt", "y.txt", "S.txt"]
SUMMARY_KEYS = ["iterations", "stop_reason", "converged", "pobj", "dobj", "errRp", "errRd", "relgap", "sigma_changes",
                "sigma_final", "returned_best_iterate", "eta_p", "eta_d", "eta_g", "sigma"]

# policy name of a case -> --sigma-policy value of the legacy CLI at that commit (missing: not comparable there)
LEGACY_POLICY_FLAGS = {
    "9bb4ba5": {"fixed": "fixed", "legacy_adaptive": "legacy_adaptive"},
    "2d8a755": {"fixed": "fixed", "residual_balancing": "legacy_adaptive"},
    "2d8a755-sgs": {"fixed": "fixed", "residual_balancing": "legacy_adaptive"},
    "c23e67e": {"fixed": "fixed"},
    "pre0": {"fixed": "fixed", "legacy_adaptive": "legacy_adaptive", "residual_balancing": "residual_balancing"},
}
NEW_POLICIES = ["fixed", "legacy_adaptive", "residual_balancing", "balance_freeze"]


def fail(msg, code=2):
    print("compare_runs: " + msg, file=sys.stderr)
    sys.exit(code)


# ----------------------------------------------------------------------------------------------- command lines
class Build:
    def __init__(self, spec):
        if ":" not in spec:
            fail("build must be KIND:EXE, got " + spec)
        self.kind, self.exe = spec.split(":", 1)
        if self.kind.startswith("legacy@"):
            self.commit = self.kind.split("@", 1)[1]
            if self.commit not in LEGACY_POLICY_FLAGS:
                fail("unknown legacy commit " + self.commit + " (known: " + ", ".join(LEGACY_POLICY_FLAGS) + ")")
        elif self.kind == "reorg":
            self.commit = None
        else:
            fail("unknown build kind " + self.kind)
        if not os.path.isfile(self.exe) or not os.access(self.exe, os.X_OK):
            fail("not an executable: " + self.exe)

    def command(self, case, out):
        """Command line of this build for the case, writing the run directory `out`."""
        if self.kind == "reorg":
            return self._reorg_command(case, out)
        return self._legacy_command(case, out)

    def _legacy_command(self, case, out):
        table = LEGACY_POLICY_FLAGS[self.commit]
        if case["policy"] not in table:
            raise ValueError("policy %s has no equivalent in the legacy build at %s" % (case["policy"], self.commit))
        factor_args = []
        if case.get("factor") is not None and case["policy"] != "fixed" and float(case["factor"]) != 2.0:
            if self.commit != "pre0":
                raise ValueError("the reference builds have a fixed factor of 2; cannot set factor %g" % case["factor"])
            factor_args = ["--penalty-factor", repr(float(case["factor"]))]
        implied_tau = 1.95 if case["scheme"] == "sgs_admm" else 1.618
        if case.get("tau") is not None and float(case["tau"]) != implied_tau:
            raise ValueError("the legacy CLI cannot set tau (its %s uses %g)" % (case["scheme"], implied_tau))
        cmd = [self.exe, case["problem"],
               "--sigma-policy", table[case["policy"]], "--sig", repr(float(case["sigma0"])),
               "--max-iter", str(int(case["max_iter"])), "--tol", repr(float(case["tol"]))]
        if case.get("switch_iteration"):
            cmd += ["--switch-admm", str(int(case["switch_iteration"]))]  # the hybrid: sGS before, ADMM from there
        else:
            cmd += ["--algorithm", "sgs" if case["scheme"] == "sgs_admm" else "admm"]
        cmd += ["--streams", str(int(case.get("streams", 15))), "--lobpcg", "1" if case.get("lobpcg", True) else "0",
                "--switch-proj-iter", str(int(case.get("switch_proj_iter", 0))),
                "--switch-proj-tol", repr(float(case.get("switch_proj_tol", 1e-2)))]
        cmd += factor_args
        if case.get("time_limit") is not None:
            cmd += ["--time-limit", repr(float(case["time_limit"]))]
        cmd += ["--history", os.path.join(out, "history.csv"), "--summary", os.path.join(out, "summary.json"),
                "--save-solution", os.path.join(out, "solution"), "--sigma-log", os.path.join(out, "sigma_log.csv")]
        cmd += shlex.split(case.get("extra", ""))
        return cmd

    def _reorg_command(self, case, out):
        if case["policy"] not in NEW_POLICIES:
            raise ValueError("unknown policy " + case["policy"])
        cmd = [self.exe, case["problem"],
               "--policy", case["policy"], "--sigma0", repr(float(case["sigma0"])),
               "--max-iter", str(int(case["max_iter"])), "--tol", repr(float(case["tol"]))]
        hybrid = bool(case.get("switch_iteration"))
        if hybrid:
            cmd += ["--scheme-switch-iteration", str(int(case["switch_iteration"]))]
        else:
            cmd += ["--scheme", case["scheme"]]
        cmd += ["--psd-medium-streams", str(int(case.get("streams", 15))),
                "--psd-lowrank-enabled", "1" if case.get("lobpcg", True) else "0",
                "--psd-warmup-switch-iter", str(int(case.get("switch_proj_iter", 0))),
                "--psd-warmup-switch-tol", repr(float(case.get("switch_proj_tol", 1e-2)))]
        if case.get("sgs_tau_reduction"):
            cmd += ["--sgs-tau-reduction", "1"]  # today's hidden sGS tau rule, for the legacy histories only
        if case.get("tau") is not None:
            cmd += ["--tau", repr(float(case["tau"]))]
        elif case["scheme"] == "sgs_admm" and not hybrid:
            cmd += ["--tau", "1.95"]  # the legacy sGS step length (plan Phase 4 criterion)
        if hybrid:
            cmd += ["--sgs-tau", "1.95"]  # the hybrid of main: 1.95 before the switch, 1.618 after
        if case.get("factor") is not None and case["policy"] != "fixed":
            cmd += ["--factor", repr(float(case["factor"]))]
        if case.get("time_limit") is not None:
            cmd += ["--time-limit", repr(float(case["time_limit"]))]
        cmd += ["--history", os.path.join(out, "history.csv"), "--summary", os.path.join(out, "summary.json"),
                "--save-dir", os.path.join(out, "solution"), "--sigma-log", os.path.join(out, "sigma_log.csv")]
        cmd += shlex.split(case.get("extra", ""))
        return cmd


def run_build(build, case, out, dry_run=False):
    """Runs the build on the case into `out`; returns (ok, seconds)."""
    os.makedirs(out, exist_ok=True)
    cmd = build.command(case, out)
    with open(os.path.join(out, "command.txt"), "w") as f:
        f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
    if dry_run:
        print("  " + " ".join(shlex.quote(c) for c in cmd))
        return True, 0.0
    for name in ["history.csv", "summary.json", "sigma_log.csv"]:  # the legacy CLI appends to the summary
        p = os.path.join(out, name)
        if os.path.exists(p):
            os.remove(p)
    t0 = time.time()
    with open(os.path.join(out, "stdout.txt"), "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    with open(os.path.join(out, "exit_code.txt"), "w") as f:
        f.write("%d\n" % rc)
    return rc == 0, dt


# ----------------------------------------------------------------------------------------------- reading runs
def read_history(path):
    """Returns (rows, columns): rows as dicts keyed by canonical column names; numbers as floats, phase as str."""
    if not os.path.isfile(path):
        raise IOError("missing history " + path)
    with open(path) as f:
        header = f.readline().strip().split(",")
        cols = []
        for c in header:
            if c not in COLUMN_ALIASES:
                raise IOError("unknown history column %r in %s" % (c, path))
            cols.append(COLUMN_ALIASES[c])
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != len(cols):
                raise IOError("malformed history line in %s: %r" % (path, line))
            row = {}
            for c, v in zip(cols, parts):
                row[c] = v if c == "phase" else float(v)
            rows.append(row)
    return rows, cols


def read_vector(path):
    if not os.path.isfile(path):
        raise IOError("missing " + path)
    with open(path) as f:
        return [float(x) for x in f.read().split()]


def read_summary(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except ValueError as e:
        # an unreadable summary is reported, never skipped silently (the summary comparison is informational, but a
        # summary that does not parse is a defect of the writer)
        print("WARNING: summary %s is not valid JSON: %s" % (path, e))
        return {"__invalid_json__": str(e)}


def rel_diff(a, b):
    if a == b:
        return 0.0
    if isinstance(a, str) or isinstance(b, str):
        return float("inf")
    if math.isnan(a) and math.isnan(b):
        return 0.0
    den = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / den


# ----------------------------------------------------------------------------------------------- comparison
def compare_dirs(dir_a, dir_b, rtol=0.0):
    """Compares two run directories. Returns a report dict; report["identical"] is the verdict."""
    report = {"a": dir_a, "b": dir_b, "rtol": rtol, "history": {}, "solution": {}, "summary": {}, "errors": []}
    identical = True

    # history
    try:
        rows_a, cols_a = read_history(os.path.join(dir_a, "history.csv"))
        rows_b, cols_b = read_history(os.path.join(dir_b, "history.csv"))
        h = report["history"]
        h["iterations_a"], h["iterations_b"] = len(rows_a), len(rows_b)
        if len(rows_a) != len(rows_b):
            identical = False
            h["note"] = "different number of iterations"
        n = min(len(rows_a), len(rows_b))
        first = None
        per_field = {}
        for field in HISTORY_FIELDS + HISTORY_INFO_FIELDS:
            if field not in cols_a or field not in cols_b:
                continue
            count, worst, worst_at = 0, 0.0, None
            for k in range(n):
                a, b = rows_a[k][field], rows_b[k][field]
                d = rel_diff(a, b)
                if d > rtol:
                    count += 1
                    if first is None or k + 1 < first["iteration"]:
                        first = {"iteration": k + 1, "field": field, "a": a, "b": b, "rel_diff": d}
                    elif k + 1 == first["iteration"] and HISTORY_FIELDS.index(field) < HISTORY_FIELDS.index(first["field"]) \
                            if field in HISTORY_FIELDS and first["field"] in HISTORY_FIELDS else False:
                        first = {"iteration": k + 1, "field": field, "a": a, "b": b, "rel_diff": d}
                if d > worst:
                    worst, worst_at = d, k + 1
            per_field[field] = {"differing_iterations": count, "max_rel_diff": worst, "max_at": worst_at}
            if count > 0 and field in HISTORY_FIELDS:
                identical = False
        h["per_field"] = per_field
        h["first_difference"] = first
    except IOError as e:
        report["errors"].append(str(e))
        identical = False

    # solutions
    for name in SOLUTION_FILES:
        try:
            va = read_vector(os.path.join(dir_a, "solution", name))
            vb = read_vector(os.path.join(dir_b, "solution", name))
        except IOError as e:
            report["errors"].append(str(e))
            identical = False
            continue
        s = {"length_a": len(va), "length_b": len(vb)}
        if len(va) != len(vb):
            identical = False
            s["note"] = "different lengths"
        count, worst, worst_at, first, max_abs = 0, 0.0, None, None, 0.0
        for i in range(min(len(va), len(vb))):
            d = rel_diff(va[i], vb[i])
            if d > rtol:
                count += 1
                if first is None:
                    first = {"index": i, "a": va[i], "b": vb[i], "rel_diff": d}
            if d > worst:
                worst, worst_at = d, i
            max_abs = max(max_abs, abs(va[i] - vb[i]))
        s.update({"differing_entries": count, "max_rel_diff": worst, "max_at": worst_at, "max_abs_diff": max_abs,
                  "first_difference": first})
        if count > 0:
            identical = False
        report["solution"][name] = s

    # summary (informational)
    sa, sb = read_summary(os.path.join(dir_a, "summary.json")), read_summary(os.path.join(dir_b, "summary.json"))
    if sa and sb:
        for key in SUMMARY_KEYS:
            if key in sa and key in sb and sa[key] != sb[key]:
                report["summary"][key] = {"a": sa[key], "b": sb[key]}
    report["identical"] = identical
    return report


def print_report(report):
    print("compare: A = %s" % report["a"])
    print("         B = %s" % report["b"])
    for e in report["errors"]:
        print("  ERROR: " + e)
    h = report["history"]
    if h:
        print("  history: %d vs %d iterations" % (h.get("iterations_a", 0), h.get("iterations_b", 0)))
        fd = h.get("first_difference")
        if fd:
            print("  FIRST DIFFERENCE: iteration %d, field %s: %r vs %r (relative %.3e)"
                  % (fd["iteration"], fd["field"], fd["a"], fd["b"], fd["rel_diff"]))
        for field, s in h.get("per_field", {}).items():
            tag = "" if field in HISTORY_FIELDS else " (informational)"
            print("    %-6s differing iterations %6d, max relative difference %.3e%s%s"
                  % (field, s["differing_iterations"], s["max_rel_diff"],
                     (" at %d" % s["max_at"]) if s["max_at"] else "", tag))
    for name, s in report["solution"].items():
        fd = s.get("first_difference")
        print("  %-5s %d vs %d entries, differing %d, max relative %.3e, max absolute %.3e%s"
              % (name, s["length_a"], s["length_b"], s.get("differing_entries", 0), s.get("max_rel_diff", 0.0),
                 s.get("max_abs_diff", 0.0), (", first at index %d: %r vs %r" % (fd["index"], fd["a"], fd["b"])) if fd else ""))
    for key, d in report["summary"].items():
        print("  summary %s: %r vs %r" % (key, d["a"], d["b"]))
    print("  RESULT: %s" % ("IDENTICAL" if report["identical"] else "DIFFERENT"))


# ----------------------------------------------------------------------------------------------- cases
def case_from_args(args):
    case = {"problem": os.path.abspath(args.problem), "policy": args.policy, "sigma0": args.sigma0, "max_iter": args.max_iter,
            "tol": args.tol, "scheme": args.scheme, "streams": args.streams, "lobpcg": bool(args.lobpcg),
            "switch_proj_iter": args.switch_proj_iter, "switch_proj_tol": args.switch_proj_tol}
    for k in ["tau", "time_limit", "factor"]:
        if getattr(args, k) is not None:
            case[k] = getattr(args, k)
    return case


def add_case_options(p):
    p.add_argument("--policy", default="fixed", choices=NEW_POLICIES)
    p.add_argument("--sigma0", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-300, help="stopping tolerance (1e-300: never reached, the reference CLI rejects 0)")
    p.add_argument("--scheme", default="admm", choices=["admm", "sgs_admm"])
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--factor", type=float, default=None)
    p.add_argument("--streams", type=int, default=15)
    p.add_argument("--lobpcg", type=int, default=1)
    p.add_argument("--switch-proj-iter", type=int, default=0)
    p.add_argument("--switch-proj-tol", type=float, default=1e-2)
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--extra-a", default="", help="extra flags for build A")
    p.add_argument("--extra-b", default="", help="extra flags for build B")
    p.add_argument("--rtol", type=float, default=0.0, help="relative tolerance (0: bitwise)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", default=None, help="write the JSON report here")


def run_pair(build_a, build_b, case, out, rtol, dry_run, report_path=None, label=None):
    """Runs both builds on the case into out/a and out/b and compares. Returns the report (None on a failed run)."""
    ca, cb = dict(case), dict(case)
    ca["extra"], cb["extra"] = case.get("extra_a", ""), case.get("extra_b", "")
    results = {}
    for tag, build, c in [("a", build_a, ca), ("b", build_b, cb)]:
        try:
            ok, dt = run_build(build, c, os.path.join(out, tag), dry_run)
        except ValueError as e:
            print("  %s: %s" % (tag, e))
            return None
        results[tag] = (ok, dt)
        if not ok:
            print("  %s FAILED (%s), see %s" % (tag, build.kind, os.path.join(out, tag, "stdout.txt")))
    if dry_run:
        return None
    if not all(ok for ok, _ in results.values()):
        return None
    report = compare_dirs(os.path.join(out, "a"), os.path.join(out, "b"), rtol)
    report["case"] = case
    report["label"] = label
    report["builds"] = {"a": build_a.kind + ":" + build_a.exe, "b": build_b.kind + ":" + build_b.exe}
    report["seconds"] = {"a": results["a"][1], "b": results["b"][1]}
    print_report(report)
    print("  times: A %.1f s, B %.1f s" % (results["a"][1], results["b"][1]))
    path = report_path or os.path.join(out, "report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run two builds on one problem and compare")
    p.add_argument("--build-a", required=True)
    p.add_argument("--build-b", required=True)
    p.add_argument("--problem", required=True)
    p.add_argument("--out", required=True)
    add_case_options(p)

    p = sub.add_parser("selfcheck", help="run one build twice and compare (run-to-run determinism)")
    p.add_argument("--build", required=True)
    p.add_argument("--problem", required=True)
    p.add_argument("--out", required=True)
    add_case_options(p)

    p = sub.add_parser("compare", help="compare two existing run directories")
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--report", default=None)

    p = sub.add_parser("batch", help="run the cases of a JSON spec on two builds")
    p.add_argument("--spec", required=True, help="JSON list of cases: name, problem, policy, sigma0, max_iter, tol, scheme, ...")
    p.add_argument("--build-a", required=True)
    p.add_argument("--build-b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--extra-a", default="", help="extra flags for build A, appended to every case's")
    p.add_argument("--extra-b", default="", help="extra flags for build B, appended to every case's")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--problem-root", default="", help="prefix for relative problem paths of the spec")

    args = ap.parse_args()

    if args.cmd == "compare":
        try:
            report = compare_dirs(os.path.abspath(args.dir_a), os.path.abspath(args.dir_b), args.rtol)
        except Exception as e:  # unusable directories
            fail(str(e))
        print_report(report)
        if args.report:
            with open(args.report, "w") as f:
                json.dump(report, f, indent=1)
        sys.exit(0 if report["identical"] else 1)

    if args.cmd in ("run", "selfcheck"):
        build_a = Build(args.build_a if args.cmd == "run" else args.build)
        build_b = Build(args.build_b if args.cmd == "run" else args.build)
        case = case_from_args(args)
        case["extra_a"], case["extra_b"] = args.extra_a, args.extra_b
        report = run_pair(build_a, build_b, case, os.path.abspath(args.out), args.rtol, args.dry_run, args.report)
        if args.dry_run:
            sys.exit(0)
        if report is None:
            sys.exit(2)
        sys.exit(0 if report["identical"] else 1)

    if args.cmd == "batch":
        build_a, build_b = Build(args.build_a), Build(args.build_b)
        with open(args.spec) as f:
            cases = json.load(f)
        if args.only:
            cases = [c for c in cases if c["name"] in args.only]
        if not cases:
            fail("no cases selected")
        out = os.path.abspath(args.out)
        os.makedirs(out, exist_ok=True)
        rows = []
        worst = 0
        for c in cases:
            case = dict(c)
            if args.problem_root and not os.path.isabs(case["problem"]):
                case["problem"] = os.path.join(args.problem_root, case["problem"])
            case["problem"] = os.path.abspath(case["problem"])
            case.setdefault("scheme", "admm")
            case.setdefault("tol", 1e-300)
            # extra flags of the batch apply to every case, after the case's own
            case["extra_a"] = (case.get("extra_a", "") + " " + args.extra_a).strip()
            case["extra_b"] = (case.get("extra_b", "") + " " + args.extra_b).strip()
            print("=== case %s" % c["name"])
            report = run_pair(build_a, build_b, case, os.path.join(out, c["name"]), args.rtol, args.dry_run, label=c["name"])
            if args.dry_run:
                continue
            if report is None:
                rows.append({"name": c["name"], "status": "FAILED"})
                worst = 2
                continue
            fd = report["history"].get("first_difference")
            rows.append({"name": c["name"], "status": "identical" if report["identical"] else "DIFFERENT",
                         "iterations": report["history"].get("iterations_a"),
                         "first_difference": fd, "seconds_a": report["seconds"]["a"], "seconds_b": report["seconds"]["b"]})
            if not report["identical"]:
                worst = max(worst, 1)
        if args.dry_run:
            sys.exit(0)
        with open(os.path.join(out, "batch_report.json"), "w") as f:
            json.dump({"builds": {"a": build_a.kind + ":" + build_a.exe, "b": build_b.kind + ":" + build_b.exe},
                       "rtol": args.rtol, "cases": rows}, f, indent=1)
        print("\n=== batch summary (A = %s, B = %s)" % (build_a.kind, build_b.kind))
        print("%-32s %-10s %9s  %s" % ("case", "status", "iters", "first difference"))
        for r in rows:
            fd = r.get("first_difference")
            print("%-32s %-10s %9s  %s" % (r["name"], r["status"], r.get("iterations", ""),
                                            ("iteration %d %s" % (fd["iteration"], fd["field"])) if fd else ""))
        sys.exit(worst)


if __name__ == "__main__":
    main()
