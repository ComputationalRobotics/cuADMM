# Progress log of the cuADMM reorganization (Milestone 1)

Per phase: status, what was done, the acceptance commands with their output, and the checks still pending.
Conventions: "pending" means not yet run; a check that cannot be run here (no MATLAB, no H200, no second
configuration) is marked "not run: <reason>", never "passed".

Documents followed (sha256 of the versions read, both from commit c23e67ef58146e553e8c688fdbfbb675533677a3 of
ComputationalRobotics/cuADMM, which was origin/main on 2026-10-21):

- `reorg-plan/cuadmm_reorg_plan.tex` (revision 2, October 20, 2026): `56e1366cf3c3c48a9f87c1f64bf04c452cdbebf38466f54c25b83b82ddf0688c`
- `reorg-plan/developer_prompt.md`: `e0589bd54cc400d2bf1fb7839130493af2908e41304393b54346f653ab06ba86`

Repository state at the start: `origin/main` = c23e67e (no newer commit); submodule `psd_projection` at b566590.
Work happens on branch `reorg`, started from c23e67e, inside `cuADMM-reorg/` only.

Machine: FASRC Cannon. Builds on CPU nodes (`sapphire`/`shared`, 16 cores); GPU runs on one A100-SXM4-80GB of
partition `gpu` in allocations under one hour (no H200 used; approval would be needed). Full pinned configuration:
`tools/reference/pinned_config.md`.

## Phase 0: skeleton and safety net

Status: skeleton built; GPU acceptance pending (see below).

Done (2026-10-21):

- Reference worktrees `$HOME/cuadmm-worktrees/{ref-9bb4ba5,ref-2d8a755,ref-2d8a755-sgs}` (submodule b566590), with
  the printing-only patch `tools/reference/print17.patch` and, for the sGS build, `tools/reference/sgs_no_plain_only.patch`;
  built with `$HOME/cuadmm-tools/build_ref.sh` into `$HOME/cuadmm-builds/<name>/` (sm_80, RelWithDebInfo):

      === ref-9bb4ba5 Wed Oct 21 15:42:23 EDT 2026   build exit code: 0
      === ref-2d8a755 Wed Oct 21 15:43:27 EDT 2026   build exit code: 0
      === ref-2d8a755-sgs Wed Oct 21 15:44:24 EDT 2026   build exit code: 0
      (cuobjdump --list-elf libcuadmm_lib.so: sm_80 cubins; CMakeCache: -gencode arch=compute_80,code=sm_80, RelWithDebInfo)

- `tools/compare_runs.py` (written first, as the developer prompt asks): runs two builds with equivalent options on
  one TXT problem (`run`), one build twice (`selfcheck`), a JSON list of cases (`batch`), or compares two run
  directories (`compare`); reports the first differing iteration and field of the history (pobj, dobj, eta_p, eta_d,
  eta_g, sigma), the first differing entry of X, y, S, and per-field counts. Checked on hand-made directories
  (identical -> exit 0; a one-ulp difference in dobj at iteration 2 and in S[2] -> reported, exit 1).
- `cuADMM-reorg/` CMake project (`CMakeLists.txt`): library `cuadmm` from the verbatim copy of `src/`, `include/`
  (c23e67e) in `src/legacy/`, the submodule subset in `third_party/psd_projection/` (shared library
  `cuadmm_psd_kernels`), `cuadmm_exe` (copied CLI), `cuadmm_gen_problems` (writes the generator problems as TXT),
  `cuadmm_tests` (copied `test/tests.cpp` and the 14 test headers it includes, `regression_problems.hpp`,
  `solver_setup.hpp`, `test/data/`), optional MATLAB MEX target (`matlab/`, not built here: no MATLAB).
  Build on the CPU node (`tools/build.sh $HOME/cuadmm-builds/reorg-phase0`):

      [ 23%] Built target cuadmm_gen_problems ... [ 39%] Built target cuadmm_psd_kernels
      [ 90%] Built target cuadmm  [ 97%] Built target cuadmm_exe  [100%] Built target cuadmm_tests
      build exit code: 0

- Baseline case lists `tools/baselines/cases_{fixed,legacy_adaptive,residual_balancing,sgs}.json` (21 + 10 + 10 + 8
  cases: the 13 generator problems, `trto1`, `mater-1`) and the GPU job scripts `tools/slurm/phase0_baselines.sh`
  (configuration record, problem generation, self-checks, baselines) and `tools/slurm/run_tests.sh`.

Acceptance (developer prompt): the comparison script reports zero differences when it compares each reference
build with itself, and the copied tests pass.

GPU configuration used: `a100-40gb-mig3g20gb_cuda12.9_gcc12.2_ss7.10.1_t8` (one MIG 3g.20gb slice of an
A100-SXM4-40GB on partition `gpu_test`, 8 threads; the full-A100 allocation on `gpu` stayed queued behind ~1000
jobs). Record: `$HOME/cuadmm-baselines/a100-40gb-mig3g20gb_cuda12.9_gcc12.2_ss7.10.1_t8/config.txt`.

- Self-checks (each reference against itself; `tools/slurm/phase0_baselines.sh`, 2026-10-21 16:21-16:33, Phase 0
  build of commit 7411625): 6 of 6 identical (9bb4ba5 fixed and legacy_adaptive on PSD_Free, 2d8a755 fixed and
  residual_balancing on PSD_Free and fixed on trto1, 2d8a755-sgs sGS on PSD_Free).
- Baselines (`$CUADMM_BASELINES/<config>/{fixed,legacy_adaptive,residual_balancing,sgs}/<case>/a`): 55 of 55
  comparisons identical, 0 different, 0 failed runs:

      fixed (21 cases, 2d8a755 against the Phase 0 copy of c23e67e): all identical, 800-20000 iterations
      legacy_adaptive (10 cases, 9bb4ba5 run twice): all identical (6000-iteration runs reach the Monitor1 region)
      residual_balancing (10 cases, 2d8a755 run twice): all identical
      sgs (8 cases, 2d8a755 without CUADMM_PLAIN_ADMM_ONLY run twice): all identical

  A first attempt with `--tol 0` failed because the reference CLI rejects a zero tolerance; the fixed-iteration cases
  use `--tol 1e-300` (never reached; the hidden sGS tau reduction never fires).
- Copied tests (`tools/slurm/run_tests.sh`, build of 7411625): 87 tests, 86 passed, 1 failed:
  `FixedSigma.LegacyAdaptiveAndPlainAdmmUnchanged`, the golden test that fails by construction on `main` (plan B14:
  the guard of the October 13 rule blocks the first change of the 9bb4ba5 golden). Pre-0 restores the rule and the
  test passes again (see Phase 1 below). Runtime 87 s.

Phase 0 is accepted on this configuration. The same baselines are to be regenerated on the full A100 when the
`gpu` allocation is granted (one more configuration, same scripts).

## Pre-0 and Phase 1 (inside `cuADMM-reorg/`)

Status: implemented (commits 75eb6ed Pre-0, f8de01f Phase 1); GPU evidence below.

- Pre-0: `legacy_adaptive` = 9bb4ba5, `residual_balancing` = 2d8a755 with `penalty.factor`, policy parameters and
  build commit in the summary, hybrid-only policy removed (`docs/CHANGES.md`, `docs/DECISIONS.md` 11-12).
- Phase 1: B2, B5, B6, B12, B16, B17, B21, B22, Q3/B1, P4 (`docs/CHANGES.md`).
- Comparison of the build of f8de01f (first run from `$HOME/cuadmm-builds/reorg-dev`, 16:26-16:34) with the
  references on every baseline case: 49 of 49 identical (fixed 21 against 2d8a755, legacy_adaptive 10 against
  9bb4ba5, residual_balancing 10 against 2d8a755, sGS 8 against 2d8a755-sgs), 0 different.
- Tests of that run: 101 tests, 98 passed; the golden test passes (B14 fixed). The 3 failures were CLI tests that
  ran against the Phase 2 binary, because the build directory had been rebuilt during the run (DECISIONS 13); the
  clean rerun of f8de01f from its own worktree and build directory (`tools/slurm/build_and_compare.sh f8de01f`) is
  recorded below when it completes.

## Phase 2: options and front-ends

Status: implemented; GPU evidence pending.

- `include/cuadmm/{options,problem,result,solver,io,cuadmm}.h`, `src/options/options.cc` (option table, setter,
  getter, validation, documentation), `src/io/{problem_data,txt,io}.cc`, `src/report/writers.{h,cc}`,
  `src/solver/solver_facade.cu` (the facade over the ported engine), `apps/cli/cuadmm.cu` (the CLI on the table,
  with today's flags as aliases), `matlab/cuadmm_mex.cu` and `matlab/+cuadmm/solve.m` (not built: no MATLAB),
  `docs/OPTIONS.md` (generated by `cuadmm_exe --print-options markdown`).
- CPU tests (`cuadmm_cpu_tests`, run on the CPU node): `Options.*` 4 tests pass (table consistency and round trip of
  every option, parsing, validation, documentation coverage).
- Acceptance (plan): bitwise for equivalent options; the option table generated into the documentation. The GPU run
  of the Phase 2 commit against the references (`tools/slurm/build_and_compare.sh <commit> phase2 <config> reorg`)
  is recorded below when it completes.

## Phase 2b: input and output formats

Status: not started.

## Phase 3: thin core

Status: not started.

## Phase 4: algorithm layer

Status: not started.

## Phase 4b: CPU screen of `balance_freeze`

Status: not started. The validated sigma-study replica is available locally
(`$HOME/cuadmm-experiments/2026-10-19_sigma_rule_check/emu-run/cuadmm_emu.py`, v1.1, with its results and
cross-check reports); it will be copied into `tools/sigma_screen/` for the screen.
