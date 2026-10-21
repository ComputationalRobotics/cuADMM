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

- Self-checks and baselines: pending (GPU allocation requested: job `reorg-gpu`, `-p gpu`, 1 A100, 59 min).
- Copied tests: pending (same allocation).

## Pre-0 and Phase 1 (inside `cuADMM-reorg/`)

Status: not started.

## Phase 2: options and front-ends

Status: not started.

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
