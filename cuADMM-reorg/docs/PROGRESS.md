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

## Milestone 1 status (2026-10-22)

Complete through the Phase-4b CPU screen; stopped before Phase 5 (the plan re-plans Phases 5-10 once Milestone 1
lands). Branches `reorg` (HEAD: this commit) and `reorg-io` are pushed; the pull-request descriptions are in
`docs/PULL_REQUESTS.md` (no GitHub CLI on the machines: to be opened on the web). Evidence per phase below; in
short: every verified commit is bit for bit the references on the 53 baseline cases on both pinned configurations
(last: f10e149, 101 of 101 GPU tests, 51 CPU tests); the readers reach the validated objectives from every
format with bitwise cross-format histories; the screen's recommendation for `balance_freeze` is factor 1.2,
halve_on_reversal, freeze bias 4 (DECISIONS 36), the default policy unchanged. The GPU gates ran on 2026-10-22 (5.73
H200-hours; section "Phase 4b GPU gates"): G0 and G1 pass, G2 and G3 fail; the default policy stays
`residual_balancing`. Awaiting the user: the go-ahead for Milestone 2. Not verifiable here:
MATLAB (MEX, SPOT recovery), MOSEK dual signs against a MOSEK run, buck3's published objective, performance and
peak memory beyond the summaries' fields.

## Milestone 2: Phase 6a, PSD strategies (in progress)

Status: steps 1 and 2 done (the kernels moved in, 14178fa; the projector classes and the selection policy, the
commit recorded in the GPU-evidence list below); the benchmark and the warm-up default (step 3) wait for the
A100 benchmark job of branch `reorg-bench6a`.

- Step 1 (B8, B10, B11, B15): the FP16 composite filter, the Lanczos norm estimate, LOBPCG and the precision
  helpers of the vendored `third_party/psd_projection` subset moved to `src/cone/psd/kernels/` (namespace
  `cuadmm::cone::psd`, the CUADMM checks, one copy of each helper: the raw-pointer ReLU is a second signature of
  the legacy kernel, the print helpers and the `curand` fill dropped); the composite coefficients are a constexpr
  table (`kCompositeFp16`) with the damping factor documented; the Lanczos start vector is deterministic
  (`fill_deterministic`, a splitmix64 counter-based generator seeded by options.seed mixed with the block index;
  the LOBPCG cold start too). The vendored directory and the `cuadmm_psd_kernels` target are gone.
  `psd.lowrank.tol` and `psd.lowrank.max_iter` are honoured (engine fields `lobpcg_tol`, `lobpcg_maxit`); the
  tolerance stays absolute (DECISIONS 45).
- Acceptance of step 1: GPU suite 118 of 118 on the MIG (dev build, `$HOME/cuadmm-builds/reorg-dev/logs/
  tests_phase6a_step1.log`, 2026-10-22 22:05); the exact and LOBPCG paths are bit for bit (no baseline uses the
  warm-up): the per-commit comparison is in the GPU-evidence list.
- Step 2, the projectors (plan Sections 10.2-10.7): `src/cone/psd/batched/` (`JacobiBatchedEvd`: the small
  tier; `BatchedEvd` and `StreamedEvd` per medium size group, `MediumTier` around them), `src/cone/psd/large/`
  (`LargeBlockState` as a view on the engine's per-block arrays until Phase 6b, `LargeBlockJob`, `RankAnalyzer`,
  `LargeBlockPolicy`, `projectors/FullEvd`, `projectors/Lobpcg`, `projectors/CompositeFilter`), `eig_info`
  (the shared cuSOLVER info check). `PsdCone` decides the strategy of every large block with the policy at
  each cone step (`begin_iteration(k, eta)` from `AdmmSolver` through `AdmmState` and `ConeProduct`: the
  one-way warm-up switch k_sw), runs the projectors on the engine's buffers in the call order of Milestone 1,
  and keeps the strided-batched GEMM of a size group when every block of the iteration takes the exact path
  (`psd.large.final_product = gemm_compat`, the only value until Phase 6b). The engine lost
  `large_full_eig_project`, `large_lobpcg_project`, `large_composite_project` and the `LOBPCG_*` /
  `COMPOSITE_*` macros; it keeps the buffers, the per-block state, the counters and `cone_measures`
  (DECISIONS 39). The per-block policy of the plan: warm(b) = `psd.large.warmup` and n_b >= `warmup.min_size`
  (honoured now), start(b) = k_sw for warm-up blocks and `psd.lowrank.start_iter` otherwise, the analysis every
  `psd.lowrank.reevaluate` iterations from start(b), LOBPCG in a low-rank mode only inside the block's
  low-rank phase (a stale mode of an earlier solve is not used before it, as the engine's gate did). The
  legacy `SDPSolver::solve()` keeps the C++ API's schedule (`legacy_global_start`: every block from the switch,
  the warm-up on every large block). Options honoured: `psd.large.warmup.min_size`, `psd.lowrank.{reevaluate,
  ratio, relaxation, max_iter, tol, rank_rtol, rank_stop_factor}`, `psd.composite.{margin, lanczos_steps,
  norm_slack}`, `psd.large.final_product` (gemm_compat; syrk rejected until 6b); the result and the summary
  report `warmup_switch_iteration`.
- Tests of step 2: `test/large_block_policy_test.hpp` (CPU: the per-block threshold, the switch by residual
  and by iteration, one-way and never at iteration 1, the cadences per block, the low-rank path off, the
  legacy global start of the C++ API and today's CLI, the argument checks); `test/psd_strategies_test.hpp`
  (GPU, on the engine of `Large_LowRankS` and of the new generator case `Large_Two` with blocks 1100 and
  1050, inputs built on the host from Householder reflections with prescribed spectra): FullEvd against the
  host projection (1e-10), the rank analysis positive and negative then LOBPCG from the warm start (modes,
  k = ceil(1.2 r), the warm-start eigenvalues, 1e-8), a dense spectrum and eigenvalues below the rank
  tolerance, a rank grown past k rejected by LOBPCG's certificate with the exact fallback and the refreshed
  analysis, the composite filter accurate (relative Frobenius error < 2e-2) and bit for bit reproducible (B8),
  its rejection by the norm test with the input restored (B22), the per-block threshold on two blocks (a
  mixed iteration with per-block products, the switch, the engine's bookkeeping); end to end through the
  facade: the composite warm-up on `Large_LowRankS` and `Large_Two` converges to the generator's optimum and
  reports the switch iteration, repeats bit for bit, and without a warm-up reports none.
- Acceptance of step 2: CPU suite 57 of 57 (sapphire job 54779332), GPU suite 128 of 130 on the first full run (two test-side expectation errors, since fixed: the 12 new tests then 12 of 12) on the MIG (dev build,
  `$HOME/cuadmm-builds/reorg-dev/logs/tests_phase6a_step2*.log`); the per-commit comparison (exact and LOBPCG
  paths bit for bit on the 53 cases) is in the GPU-evidence list.

## Milestone 2: Phase 5, cone layer

Status: done (commit 1cca25e, verified bit for bit); authorized on 2026-10-22 with Phases 6a, 6b, 7 and 10
(docs/PLAN_M2.md).

- `src/cone/`: `Cone` (plan Section 8.1: kind, dimension, self-duality, `project_dual`, `project_primal` with the
  Moreau default, `measure`, scaling compatibility, host reference measure), `ConeRegistry` (input codes to
  kinds and factories; 'q', 'r', exponential and power cones rejected with their phase), `ConeProduct` (one object
  per kind present, the cone step `project_dual(z, s, k)`, the per-block measures in input order), `FreeCone`
  (S_u = 0, identity primal projection, measures max |v|, -sum |v|), `NonnegativeCone` (max(z, 0), min entry,
  sum of the negative entries), `SliceTable` (the blocks of a kind as device arrays: one launch per cone, binary
  search over the blocks, until Phase 8 makes the slices contiguous), `psd/PsdCone` (the three tiers of main moved
  from `AdmmState::project_psd_tiers` without a change, on the engine's buffers; `psd/svec_pack` unpacks and packs
  only the PSD entries), `termination/Certificate` (the coefficients of C - A^T y through `Cone::measure`, the
  lower bound with explicit per-block bounds; a negative entry of `certificate.block_bounds` means no bound and
  the bound is reported unavailable with the blocks listed: B24).
- `AdmmState` takes the product; `AdmmSolver::finish` takes the cone violations and the certificate from it; the
  facade builds the product after the engine's setup and evaluates the certificate variants (block bounds, trace
  bound, trace bound scale) through `Certificate`; the legacy `SDPSolver::solve()` wrapper builds a product too.
  The engine's `cone_measures` stays (the copied certificate test of main drives it; Phase 6 removes it with the
  tiers).
- Tests (`test/cone_test.hpp`): the registry; the projection properties of plan Section 8.3 (membership,
  idempotence, Moreau decomposition, orthogonality, nonexpansiveness; the free entries exactly 0, the nonnegative
  ones exactly max(z, 0)) on the small, batched-small, medium, LP-only and free-only problems; the product's
  measures bit for bit the engine's `cone_measures` (small, medium, large tiers); the host reference measures
  against the device measures; the free-variable tests of Section 9.3: an LP with free variables (new generator
  case `Nonneg_Free`: 24 nonnegative, 10 free, 22 constraints; plain ADMM needs about 19,000 iterations at fixed
  sigma = 1 and the adaptive rules do not reach 1e-6 within 20,000, DECISIONS 38), free only with non-unique X_u
  (`Free`, 30 free, 15 constraints, with the validator), a warm start with S_u != 0, sGS-ADMM with free blocks, a
  large block with free variables on the LOBPCG path (new case `Large_Free` through the regression harness:
  converged in 149 iterations with the positive LOBPCG mode), the certificate with per-block bounds (a valid lower
  bound) and with a missing bound (unavailable, the block named, the wrong count rejected). "Free blocks with
  dependent rows (presolve)" comes with Phase 7; "PushBox_N=30_SOS against MOSEK" needs MOSEK reference solutions
  that are not on the machines: not run.
- Acceptance: GPU suite 118 of 118 on the MIG (dev build), CPU 51 of 51; the clean per-commit comparison of
  1cca25e (53 of 53 identical, 118 tests) is recorded in the GPU-evidence list below.

## Phase 4b GPU gates (approved 2026-10-22, in progress)

The user approved the H200 gates G0-G3 on 2026-10-22 (at most 12.58 H200-hours in total, expected 6-9.5; G0 first,
G1-G3 only after the fixed-sigma reproduction holds; the comparators with every parameter explicit; nothing beyond
the plan's runs). Preparation on the CPU: the final commit f10e149 built for sm_90 (`CUADMM_CUDA_ARCH=90-real
tools/build.sh ~/cuadmm-builds/reorg-f10e149-sm90`, RelWithDebInfo, cubin sm_90; `cuadmm_exe` sha256
`927c0308f6783da655a68dc7790350c47846aba9b301eae9eefb6d59cf7a5d37`); the exact command lines parsed and read
their problems in a dry run on the CPU node (they stop at the CUDA initialization there). Job files regenerated
by `tools/sigma_screen/gpu_gates.sh` into `$HOME/cuadmm-experiments/2026-10-21_phase4b_screen/gpu_gates/`
(the first version kept in `prepared_v1/`): every policy parameter explicit (`residual_balancing`: factor 2,
win ratio 1.35, guard on, residual weight 1, bounds [1e-3, 1e3]; the candidate: factor 1.2, halve_on_reversal,
bias 4, band centre, K_f 10000, W 8, net 2, k_min 1000, the default schedule, the same bounds; `fixed`: sigma 1),
the campaign protocol (plain ADMM, tau 1.618, tol 1e-4, validation every 50 at 1e-4 with 8 threads, strict 1e-6,
caps on the iteration time, 15 streams, LOBPCG on). The fixed-sigma and old-rule comparators of G1 and G3 are
the campaign's H200 runs (9bb4ba5, same protocol, 1800 s caps; `gpu_gates/references.json`), re-run only where
G0 reproduces them (trto1, buck1), to avoid duplicate submissions; the pendulum comparators are the pendulum
experiments' runs (fixed sigma = 1: 230,315 iterations, 622 s solve / 625 s end to end; old rule: 45,393
iterations, 124.0 s). Caps: G0 0.58 h, G1 4.67 h, G2 1.0 h, G3 4.0 h (10.25 h; job limits 11.25 h).
Submissions and results are appended here as they come (`gpu_gates/submissions.txt`, `logs/`, `runs/`,
`gates_report.md` from `tools/sigma_screen/gates_analyze.py`).

- G0 submitted 2026-10-22 13:38 as job 54683287 (gpu_h200, 0:45 limit): trto1, buck1 (600 s caps) and pendulum
  N=80 (900 s) at fixed sigma = 1. Ran 13:38:55-13:55:47 on holygpu8a12201 (NVIDIA H200), elapsed 17:03 =
  0.28 H200-hours. **Pass**: trto1 strict stage at 60,150 iterations (campaign 60,150), practical at 38,474 (same),
  objective -1104.50028 (relative difference to the validated reference 4.5e-10), 24.5 s of iterations (campaign
  23.3 s); buck1 strict at 119,850 (same), practical 69,440 (same), objective -146.4190581 (3.6e-11), 70.1 s
  (68.9 s); pendulum N=80 practical 1e-4 at 230,315 iterations (same as the three repetitions of the pendulum
  experiment), 686 s of iterations to that snapshot (reference 622 s solve, 625 s end to end), then the strict
  stage not reached within the 900 s cap (302,334 iterations; the reference runs stopped at the practical stage).
  The iteration counts and objectives are reproduced exactly on the H200 with the sm_90 build; the times are 5-10%
  above the campaign's Release build.
- G1, G2, G3 submitted 2026-10-22 13:57 after the G0 verdict (jobs 54688317, 54688318, 54688319). G2 (pendulum)
  started 13:59 on holygpu8a12201, G3 (mater-3..6) 13:59 on holygpu8a12204. G1 (job 54688317) also landed on
  holygpu8a12204, on GPU-458ee7a8: all twelve runs ended within seconds with "could not initialize CUDA device 0"
  (the CUDA context could not be created on that GPU, while G3 ran on another GPU of the same node; the PLATO
  campaign's job scripts excluded this node); 41 s of H200 time. The outputs are kept in
  `runs/G1_attempt1_failed/`; G1 resubmitted 14:08 with `--exclude=holygpu8a12204` (job id in
  `submissions.txt`); `gpu_gates.sh` now excludes that node by default.
- G1 (job 54689882, holygpu8a12603, 14:08-15:37, 1.49 h), G2 (54688318, holygpu8a12201, 13:59-14:59, 1.01 h), G3
  (54688319, holygpu8a12204, 13:59-16:56, 2.94 h) completed; the analysis (`tools/sigma_screen/gates_analyze.py`,
  CPU job 54707952) is in `docs/phase4b_gates_report.md` (copy of `gpu_gates/gates_report.md`; raw runs in
  `gpu_gates/runs/`, logs in `gpu_gates/logs/`, references in `tools/sigma_screen/gates_references.json`).
  **Total 5.73 H200-hours** (approved at most 12.58; expected 6-9.5), every allocation released (all jobs COMPLETED).
- Results (candidate = balance_freeze, factor 1.2, halve_on_reversal, bias 4; residual_balancing with its explicit
  parameters; fixed sigma = 1 and the old rule from the campaign's H200 runs, trto1/buck1 fixed from G0):
  - G1 (sigma_0 = 1, strict-stage times): trto1 candidate 38.5 s, residual_balancing none within 600 s, fixed 24.5 s,
    old none; buck1 67.8 / none / 70.1 / none; vibra1 47.5 / 200 / 556 / 39.5; mater-1 23.6 / 5.4 / 124 / 32.3;
    mater-2 61.3 / 13.5 / 433 / 16.6; shmup1 none for every policy (1800 s). The candidate's iteration counts
    are exactly the CPU replica's (97,750; 121,900; 80,800; 73,300; 163,100), as are residual_balancing's (vibra1
    342,300; mater-1 16,300; mater-2 36,650). G1a no problem lost: **pass**; G1b changes stop (every run frozen
    by the step rule at 151-1801, total variation <= 11.4): **pass**; G1c no change followed by a >= 10x rise of the
    validated error: **pass** (largest 7.4x); G1d shifted geometric mean of the strict time: candidate 89 s,
    residual_balancing 183 s, fixed 225 s, old rule 173 s: **pass**.
  - G2 (pendulum N=80, 900 s caps, validated 1e-4 stage): sigma_0 = 1: candidate not reached (frozen at sigma 7.7
    at iteration 131; 299,319 iterations), residual_balancing 84,390 iterations / 252 s; sigma_0 = 100: candidate
    187,198 / 548 s, residual_balancing 96,395 / 285 s (5 spikes >= 10x). References: old rule 45,393 / 124 s,
    fixed sigma = 1 230,315 / 622 s. No run reached the strict stage within 900 s. **Fail**: the candidate is the
    slowest policy on this instance and misses the cap at sigma_0 = 1; sensitivity to sigma_0 is large.
  - G3 (1800 s caps): mater-3 candidate strict at 1,167,250 iterations / 643 s, residual_balancing 188,000 / 102 s,
    old rule 231,000 / 88 s, fixed none; mater-4 candidate none (practical at 330,496, 398 s), residual_balancing
    652,000 / 774 s (6 spikes >= 10x), old 769,500 / 498 s; mater-5 candidate none (practical 582,619, 1290 s),
    residual_balancing none (practical 61,548, 143 s; 5 spikes), old 890,000 / 1039 s; mater-6 nobody (candidate
    not even practical; residual_balancing practical at 88,414, 403 s; old practical at 86,415). **Fail** against
    the old rule on mater-4 and mater-5 and 7x slower on mater-3: the step freeze fixes sigma at 0.04-0.10 within
    the first 1,800 iterations, while the adaptive rules keep moving (hundreds of changes) and reach the strict
    stage. The candidate had no spike >= 10x anywhere (residual_balancing: 21 over its runs).
- Implication for the planned default: the candidate passes the small-problem gate G1 outright (and it is the only
  policy with bounded variation and no spikes), but it fails G2 and G3 against both comparators because its freeze
  comes too early at a value the larger problems outgrow. On this evidence the promotion of `balance_freeze` to
  the default is not justified; `penalty.policy` stays `residual_balancing`, and the freeze design (freeze time
  scaled with the problem, or a freeze only after the practical stage) goes to the Milestone 2 planning as O2/O8
  (DECISIONS 37). The Phase 5 milestone boundary stays in place.

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

Status: implemented; GPU evidence recorded below (b44596b: 49 of 49 identical on the MIG configuration).

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

Status: implemented on branch `reorg-io` (commits listed below), merged into `reorg` (b64cc18); CPU acceptance run;
GPU acceptance run on the MIG configuration (below); an independent review found three correctness bugs and one
build break, being fixed on `reorg-io` for a second merge (listed at the end of this section).

Done (2026-10-21), plan Section 12, developer prompt "References for the readers (Phase 2b)":

- Readers: native HDF5 schema 1 (`src/io/h5.cc`, also the writer, `/solution` and `/start`), SDPA sparse
  `.dat-s` / `.dat-s.gz` with the checks of `sdpa_to_cuadmm.py` (`src/io/sdpa.cc`, `text_source.cc`), SeDuMi,
  SDPT3 and MOSEK `prob` `.mat` through matio, v5 and v7.3 (`sedumi.cc`, `sdpt3.cc`, `mosek.cc`, `mat_common.cc`);
  the legacy TXT reader untouched. Every reader fills the CSC of At directly through `CscBuilder` (in-place bucket
  sort, duplicates merged or rejected per format, zeros dropped; `csc_builder.cc`), reports what it did
  (`source.report`), keeps the DIMACS kinds and the mapping to the file's blocks (`original_block_*`), and applies
  the exact simplifications of `io.simplify` (`simplify.cc`).
- Detection (`io.cc`): extension, file signature, and the variables of a `.mat` file; `io.format` overrides.
- Writers: `write_solution` in the input's convention or `io.save_format` (`solution_writers.cc`), the TXT problem
  writer (`txt_writer.cc`), `write_problem` (HDF5, chunked, optional deflate, `source_sha256` by `sha256.cc`).
- Tools: `cuadmm-convert` (`apps/cli/cuadmm_convert.cc`), `cuadmm_io_bench` (`bench/io_read_bench.cc`),
  `tools/io/make_sdpt3_testfiles.py` (the SDPT3 test files of `test/data`, v5 by scipy and v7.3 by h5py).
- CMake: `CUADMM_WITH_HDF5`, `CUADMM_WITH_MATIO`, `CUADMM_WITH_ZLIB` (AUTO/ON/OFF; `cmake/FindHDF5Lib.cmake`,
  `cmake/FindMatio.cmake`); `--formats` of the CLI lists what is compiled in; absent readers fail with a message.
- Tests (CPU, `test/io_*_test.hpp` included from `test/cpu_tests.cpp`): builder/svec/sha256, SDPA edge cases and the
  PLATO instances against the reference TXT conversions, simplifications and the mapping back, TXT and SDPA
  writers, HDF5 round trips (problem, solution, warm start, malformed files, detection), hand-made SeDuMi / SDPT3 /
  MOSEK files in v5 and v7.3 (A vs At and N = m, asymmetric PSD storage, K.q rejected, an `s` entry with several
  sizes, `l`/`u` blocks, MOSEK bounds and ranges, missing sense, maximization, cones rejected), the Python-written
  SDPT3 files, buck3 SeDuMi vs SDPA, the three SPOT MOSEK files vs TXT, and the MOSEK/SeDuMi/SDPT3 solution writers
  read back through matio. Tests of a format that is not compiled in skip (GTEST_SKIP).

Data prepared (read-only inputs untouched): `$HOME/cuadmm-data/plato_sdpa/` (`buck1`, `buck3`, `trto1`, `trto2`,
`mater-1`, `mater-2` `.dat-s` extracted from the Kocvara zips, plus `trto1.dat-s.gz`), `$HOME/cuadmm-data/plato_sedumi/buck3.mat`
(from `https://plato.asu.edu/ftp/sdp/buck3.mat.gz`; `trto1.mat.gz` and `mater-1.mat.gz` do not exist there: 404),
`$HOME/cuadmm-data/plato_h5/` (conversions for the GPU cases).

Builds (CPU node holy8a26208, job 54419762):

    BUILD_JOBS=8 tools/build.sh $HOME/cuadmm-builds/reorg-io -DCUADMM_WITH_HDF5=ON -DCUADMM_WITH_MATIO=ON -DCUADMM_WITH_ZLIB=ON \
        -DHDF5_ROOT=$HOME/cuadmm-io-env -DMATIO_ROOT=$HOME/cuadmm-io-env -DZLIB_ROOT=$HOME/cuadmm-io-env
    -- cuADMM-reorg: HDF5 TRUE (1.14.6), matio TRUE (1.5.30), zlib TRUE (1.3.2)   ...   build exit code: 0
    BUILD_JOBS=6 tools/build.sh $HOME/cuadmm-builds/reorg-io-nodeps -DCUADMM_WITH_HDF5=OFF -DCUADMM_WITH_MATIO=OFF -DCUADMM_WITH_ZLIB=OFF
    -- cuADMM-reorg: HDF5 FALSE (), matio FALSE (), zlib FALSE ()   ...   build exit code: 0

Acceptance (CPU node holy8a26208, 2026-10-21 21:19, build directories above):

- `./cuadmm_cpu_tests` with HDF5 + matio + zlib: `33 tests from 8 test suites ran. [ PASSED ] 33 tests.`
  Without the libraries (`reorg-io-nodeps`): `33 tests ... [ PASSED ] 22 tests. [ SKIPPED ] 11 tests` (the gz, HDF5
  and .mat tests skip; the "reader not compiled in" messages are checked by hand below).
- The TXT reader gives the same ProblemData as today: `src/io/txt.cc` is unchanged (the reader body is the Phase 2
  one over the verbatim `Problem::from_txt`); `IoSimplify.TxtWriterRoundTrip` and the PLATO comparisons read it.
- Cross-format checks printed by the tests (`max|diff| / max|ref|` over the entries of At, b and C; 0 = bitwise):

      trto1 / mater-1 / buck3 .dat-s vs the reference TXT conversion (sdpa_to_cuadmm.py):  0 / 0 / 0  (bitwise, same blocks and DIMACS kinds)
      trto1 .dat-s.gz vs .dat-s:                                                            0 (bitwise)
      buck3 SeDuMi .mat vs .dat-s (plan: <= 2e-16; blocks s 320, s 321, l 544):             0 (bitwise; the 544 block is reported as simplified)
      PushBox_N=30_MOMENT MOSEK vs TXT (plan: <= 6e-17):   At 5.551e-17, b 0, C 5.551e-17  (374258 entries; most entries bitwise, the rest limited by the 16-decimal TXT)
      PushBox_N=30_SOS (maximization) MOSEK vs TXT:        At 3.925e-17, b 4.163e-17, C 0
      PushT_N=10_MOMENT MOSEK vs TXT:                      At 5.551e-17, b 0, C 4.930e-17
      HDF5 round trips (plain, deflate 4, after /solution was added, trto1 .dat-s -> .h5): 0 (bitwise, metadata equal)
      SDPT3 v5 (matio) vs v7.3 (matio), and the scipy-written v5 / h5py-written v7.3 files of test/data: same arrays

- Formats listed by the CLI: `txt sdpa sdpa.gz h5 sedumi sdpt3 mosek` (with the libraries) and `txt sdpa`
  (without); a `.mat`, `.h5` or `.dat-s.gz` input of the second build fails with
  "reading .mat files (SeDuMi, SDPT3, MOSEK) needs matio, which is not compiled into this build (CUADMM_WITH_MATIO)",
  "the native HDF5 format is not compiled into this build (CUADMM_WITH_HDF5)", "... is gzip-compressed, and this
  build has no zlib (CUADMM_WITH_ZLIB): decompress the file first".
- HDF5 read rate (`cuadmm_io_bench`, node-local /tmp of holy8a26208, 1e8 nonzeros: 100 PSD blocks of side 1000,
  m = 100000, 1000 entries per column, 1539 MB uncompressed; page cache warm, which is the best a disk can do):

      generate (write)                    0.85 s   1801 MB/s
      read .h5                            0.71 s   2158 MB/s   140 Mnnz/s   (three runs: 0.711-0.775 s)
      read .h5, deflate level 1 (846 MB)  6.9-7.3 s  115-123 MB/s  14 Mnnz/s  (decompression-bound)
      read the TXT conversion (2470 MB)   33.8 s   73 MB/s     2.96 Mnnz/s  (the legacy reader, as the plan's 3.5e6/s)
      dd of the .h5 file from the page cache: 5.2 GB/s

  Reading is 23 x faster per nonzero than the TXT parser and reaches 2.2 GB/s, above the bandwidth of the node's
  local disk and of the network file systems here; the remaining cost is the copy out of the page cache plus the
  structure checks (sorted columns, ranges, finiteness), i.e. no parsing. Before the early exit of the
  simplification scan (commit below) the same read took 1.5 s (1 GB/s).
- `docs/OPTIONS.md` regenerated (`cuadmm_exe --print-options markdown`); the io options are "now".

GPU acceptance (run by the parent on the MIG configuration, 22:05-22:20, working tree at 4640e8a):
`tools/baselines/cases_formats.json` as written (fixed sigma = 1, 20,000 iterations, the Kocvara-page values) cannot
pass: 20,000 iterations converge none of the three instances, and the page's value for trto1 (1.1045) is 1000x the
optimum of the file as solved (the strict-validated runs of the sigma study and the PLATO campaign give
-1104.50028 in the TXT minimization; mater-1 and buck3 agree with the page: -143.4654, 607.6055). The acceptance
was therefore run with `tools/baselines/cases_formats_gpu.json` and `tools/io/check_formats.py <cuadmm_exe>
$HOME/cuadmm-data <out dir>`: (a) objective: trto1 (fixed sigma = 1, tolerance 1e-4: converged at 38,474 iterations)
and mater-1 (residual_balancing: converged at 8,850) from TXT, `.dat-s` and `.h5` give sign * pobj and sign * dobj
within 1.4e-4 of the validated references (1104.50028, -143.4653137) in the file's own sense; (b) cross-format: the
17-digit histories of the `.dat-s` and `.h5` runs are bit for bit those of the TXT runs (trto1 38,474 iterations,
mater-1 8,850, buck3 3,000), and buck3 from the SeDuMi `.mat` (544 block read as `l`) and from its `.h5` conversion
are bit for bit the TXT run too. buck3's published objective needs an H200-length run (the campaign ran 364,000
iterations in 1,800 s without reaching the strict stage): not checked. 11 of 11 cases pass; outputs in
`$HOME/cuadmm-baselines/a100-40gb-mig3g20gb_cuda12.9_gcc12.2_ss7.10.1_t8/dev_36e8552/formats3/` (`report.json`,
`reevaluation.log`). The same session ran the full GPU test suite of the merged tree: 101 ran, 101 passed.

Not done / not verified:

- The solution output read back by SPOT's `recover_mosek_sol_blk.m`: no MATLAB here; the writer's `barx` / `bars`
  layout (lower triangles column by column, in `bardim` order, `xx` / `slx` for the scalar block) is checked by
  `IoMat.MosekSolutionWriterReadBack` and on PushT_N=10_MOMENT through matio instead.
- MOSEK dual signs of a maximization (DECISIONS 25) are the documented MOSEK convention, not compared with a
  MOSEK run.
- Second-order cones (`K.q`, `blk 'q'`, `prob.cones`/`accs`), CBF, complex data: rejected with a message (Phase 9).
- The MEX gateway is not built (no MATLAB); it still takes the in-memory arrays of Phase 2.

### Phase 2b review fixes (2026-10-21, commits on `reorg-io` after eb3ee06)

Independent review of the merged tree (findings against plan Section 12): fixed 1-5, 7-9, 11, 13; 6, 10, 12 are
documented (DECISIONS 32-35, CHANGES "Phase 2b review fixes"). Each fix has a test:

- A. clean build without `HDF5_ROOT` found the system HDF5 1.10.5, which lacks `H5Treclaim`: `h5.cc` now uses
  `H5Treclaim` from 1.12 and `H5Dvlen_reclaim` before (`H5_VERSION_GE`).
- 1. string attributes / `/meta/report` read with the stored character set (h5py's variable-length UTF-8, fixed
  ASCII, variable ASCII, fixed UTF-8; int32 datasets): `tools/io/make_h5_testfiles.py` -> `test/data/h5_*.h5`,
  `IoH5.FilesWrittenByH5pyAndMatlabStyle`.
- 2. MOSEK `slx`/`sux` of ranged and fixed variables (`SourceInfo::mosek_var_slack/var_cost/fixed_*`, also in the
  `.h5` metadata): `IoMat.MosekWrittenDualsSatisfyTheDualConstraint` (c - a'y = slx - sux for every variable
  kind, slc - suc = y, sign conventions, min and max; the reviewer's minimal case gives y = 0, slx = 0, sux = 1).
- 3. `/At/rowind`, `/b/index`, `/C/index` range-checked before the simplification scan:
  `IoH5.MalformedIndicesAreRejectedBeforeTheSimplification` (2^40, 5, -1, C index 2^40; simplify on and off).
- 4. `/solution` and `/start` in the file's block layout with `/solution/blocks` (`file_block_layout`,
  `to_file_layout`, `from_file_layout` in `simplify.cc`); `solution_blocks.txt` with the TXT convention:
  `IoH5.SolutionAndWarmStartInTheFileBlockLayout`, `IoSimplify.TxtSolutionWriterAndFormatDispatch`.
- 5. missing `sense` attribute rejected unless `io.sense`: `IoH5.MissingSenseAttributeNeedsIoSense`.
- 7. `.cbf` / `.cbf.gz` -> "CBF comes with Phase 9": `IoSimplify.TxtSolutionWriterAndFormatDispatch`.
- 8. MOSEK quadratic terms rejected: `IoMat.MosekBoundsRangesSenseAndRejections`.
- 9. SDPA text source read errors are errors: `IoSdpa.ExplicitZerosDoNotTouchABlockAndReadErrorsAreReported`
  (a directory opened as a file), which also covers 13 (a block touched by an explicit zero only is untouched).
- 11. HDF5 files closed explicitly with the status checked; the SDPA JSON sidecar, `certificate.txt` and
  `solution_blocks.txt` streams checked.

Builds and tests after the fixes (CPU node holy8a26208):

    tools/build.sh ~/cuadmm-builds/reorg-io -DCUADMM_WITH_HDF5=ON -DCUADMM_WITH_MATIO=ON -DCUADMM_WITH_ZLIB=ON -DHDF5_ROOT=... -DMATIO_ROOT=... -DZLIB_ROOT=...
      -- cuADMM-reorg: HDF5 TRUE (1.14.6), matio TRUE (1.5.30), zlib TRUE (1.3.2);  build exit code: 0
      ./cuadmm_cpu_tests: 39 tests from 8 test suites ran. [ PASSED ] 39 tests.   --formats: txt sdpa sdpa.gz h5 sedumi sdpt3 mosek
    tools/build.sh ~/cuadmm-builds/reorg-io-nodeps -DCUADMM_WITH_HDF5=OFF -DCUADMM_WITH_MATIO=OFF -DCUADMM_WITH_ZLIB=OFF
      -- cuADMM-reorg: HDF5 FALSE (), matio FALSE (), zlib FALSE ();  build exit code: 0
      ./cuadmm_cpu_tests: 39 tests ran. [ PASSED ] 23 tests. [ SKIPPED ] 16 tests.   --formats: txt sdpa
    tools/build.sh ~/cuadmm-builds/reorg-io-auto        (clean configure, no HDF5_ROOT, the environment of ~/cuadmm-tools/env.sh)
      -- cuADMM-reorg: HDF5 TRUE (1.10.5), matio FALSE (), zlib TRUE (1.2.11)   (/usr/lib64/libhdf5.so, /usr/lib64/libz.so);  build exit code: 0
      ./cuadmm_cpu_tests: 39 tests ran. [ PASSED ] 32 tests. [ SKIPPED ] 7 tests.   --formats: txt sdpa sdpa.gz h5

The reviewer's probe files (`~/cuadmm-builds/reorg-io/io_test_tmp/review_probe/`) with the io-env build: the four
string variants and the int32 file convert; `huge_idx.h5`, `huge_C_idx.h5`, `vlen_utf8_oob.h5`, `vlen_utf8_neg.h5`
are rejected with "/At/rowind[2] = 1099511627776 outside [0, 5)" and the like (exit 1, no crash).

## Phase 3: thin core

Status: implemented (with Phase 4 in one commit); GPU evidence pending.

- `src/core/check.h` (one set of macros, `cuadmm::Error`; the legacy and vendored headers map onto it, B15),
  `src/core/device_buffer.h`, `src/core/gpu_context.{h,cu}`. The ported engine keeps its streams and handles until
  Phase 6b (stream pool, arena).
- Acceptance: bitwise (checked with Phase 4 below).

## Phase 4: algorithm layer

Status: implemented; GPU evidence on the MIG configuration recorded below (59e943a: 53 of 53 identical); the
second configuration (full A100) is in progress.

- `src/admm/{admm_state,iteration_scheme,penalty_policy,admm_solver}`, `src/termination/{stopping_rule,
  best_iterate_tracker}`, `src/report/observers`; the engine's `solve()` is a wrapper over `AdmmSolver::run`
  (DECISIONS 16); the facade runs the loop with observers (`Solver::add_observer`).
- CPU tests (`cuadmm_cpu_tests`, 13 tests, pass on the CPU node): `PenaltyPolicy.*` replay the reference histories of
  `test/data/golden/` through the policies and reproduce every sigma exactly: `PSD_Free` legacy_adaptive 6000
  iterations (136 changes), `mater-1` legacy_adaptive to tolerance 1e-4 (8117 iterations, 141 changes, 2 Monitor1
  corrections), `PSD_Free` residual_balancing 6000 iterations (135 changes); `balance_freeze` on scripted sequences
  (freeze at the iteration, ping-pong by net movement with the +,+,-,- pattern, band centre, bias, halve-on-reversal,
  sigma_0 clamp, state round trip, equality with residual_balancing before a freeze).
- Acceptance (plan): bitwise for ADMM with each existing policy; sGS-ADMM with FixedPenalty and tau = 1.95 against
  the 2d8a755 build without `CUADMM_PLAIN_ADMM_ONLY` at fixed sigma and tolerance 1e-300. The case lists gained the
  sGS-phase adaptive rule, two hybrid cases and one sGS case to tolerance with the reduction rule on. The GPU run is
  recorded below when it completes.
- B23 (done after the merge of Phase 2b): the externally evaluated snapshots are ranked by the maximum DIMACS error,
  validated ones first, the practical merit only between snapshots without a finite DIMACS error
  (`ValidationResult::better_than`; CHANGES.md "Phase 4, B23"); the GPU test
  `ValidationStopping.NoStopWhenConeFailsAndBestIterateReturned` checks the returned iterate against that ranking.

## Phase 4b: CPU screen of `balance_freeze`

Status: the CPU screen is running (Stage 1 and the comparators submitted on 2026-10-21 at 21:16); results are
recorded below as they complete. The GPU gates are prepared as commands only (they need approval: H200, or jobs
over one hour).

- The validated sigma-study replica (plan: "ask the maintainers for it") is in this account:
  `$HOME/cuadmm-experiments/2026-10-19_sigma_rule_check/emu-run/cuadmm_emu.py`, v1.1 (2026-10-19), sha256
  `f213e9c6875900491af0eddce14e1c818a20416ab32ec534585a6e63cb40db18`, with its cross-check against the GPU
  (`emu-run/results/crosscheck.md`), its rule comparison (`emu-run/results/rule_comparison.md`) and the report
  (`report/`). `tools/sigma_screen/cuadmm_emu.py` is that file plus the `balance_freeze` rule (v1.2, sha256
  `b0a5814e684d1a4b01366742838ce0cb6439bb2b91c236353635f44258f3af6b`): `BalanceFreezeState`, a transcription of
  `ResidualBalancingPenalty` with the freeze enabled (`src/admm/penalty_policy.cc`), selected as
  `--rule balance_freeze[:key=value,...]` with the option names of `penalty.*`; checkpoints carry the freeze state;
  `summary.json` gains `freeze_iteration` and `freeze_reason` (DECISIONS 19).
- The replica is unchanged for the other rules: the sigma study's v1.1 suite (`emu-run/test_emu_rules.py`, run
  against `tools/sigma_screen/` on the CPU node) passes, so v1.2 = v1.1 (= v1.0) on its cases.
  `tools/sigma_screen/test_balance_freeze.py` (7 tests: the rule parser, the freeze at K_f, ping-pong with the
  +,+,-,- pattern and the band centre, the last value and the bias, halve-on-reversal for three factors, the sigma_0
  clamp, the emulator against the state object on trto1, the checkpoint round trip on vibra1) passes on the CPU node.
- Cross-check of the C++ policy against the replica: `tools/sigma_screen/replay_policy.py` drives
  `BalanceFreezeState` with the residual sequences of the golden histories (PSD_Free residual_balancing, 6000
  iterations; mater-1 legacy_adaptive, 8117 iterations) for 15 parameter cases and writes
  `test/data/golden/balance_freeze_cases_{PSD_Free,mater-1}.csv` (sigma after every iteration, shortest round-trip
  decimals, and the names of the log entries); `PenaltyPolicy.BalanceFreezeAgreesWithTheReplicaOn{PsdFree,Mater1}`
  replay them through `make_penalty_policy` and compare every sigma and every log entry exactly: pass
  (`cuadmm_cpu_tests`: 15 tests pass on the CPU node). Cases on PSD_Free at sigma_0 = 100: the defaults (ping-pong
  freeze at 1000); `freeze_iteration=3000,pingpong_min_iter=100000` (K_f); `factor=1.2,step_schedule=halve_on_reversal`
  (step freeze at 121); `step_schedule=halve_on_reversal,pingpong_min_iter=100000,freeze_iteration=100000` (step
  freeze at 141); `freeze_value=last,freeze_bias=2`; `freeze_bias=4,pingpong_window=12`;
  `pingpong_window=6,pingpong_net=0`; sigma_0 = 5000 (clamped to 1000); sigma_0 = 1e-4 with
  `sigma_min=1e-2,sigma_max=1e2,factor=1.5`; `freeze_iteration=5000,pingpong_min_iter=100000,freeze_bias=0.5,
  freeze_value=band_centre`; `factor=1.2,freeze_iteration=20000,pingpong_window=12`. On mater-1 at sigma_0 = 1: the
  defaults; `factor=1.2,step_schedule=halve_on_reversal`; `freeze_iteration=20000,pingpong_window=12`;
  `freeze_iteration=50000,pingpong_window=8,pingpong_net=2,freeze_bias=2,freeze_value=last`.
- Screen design (plan Table 2, fractional): Stage 1 = factor {2, 1.2} x step_schedule {constant, halve_on_reversal}
  x K_f {1e4, 2e4, 5e4} (12 settings) at sigma_0 in {0.5, 1, 2} on the nine small PLATO problems (trto1, trto2,
  buck1, buck2, vibra1, vibra2, mater-1, mater-2, shmup1): 324 runs. Comparators: old (legacy_adaptive) and new
  (residual_balancing) at the three sigma_0 values, fixed sigma = 1: 63 runs. Stage 2 = freeze_value {band centre,
  last} x freeze_bias {1, 2, 4} for the two best Stage-1 settings at sigma_0 = 1 (90 runs), then the best two of
  Stage 2 at sigma_0 in {0.5, 2}. Protocol of the sigma study: tolerance 1e-4, validation every 50 iterations,
  strict stage max|DIMACS| <= 1e-6, iteration cap 200,000 (DECISIONS 18). Driver `tools/sigma_screen/screen.sh`:
  one 1-core Slurm array task per run (seas_compute, cascadelake nodes as in the sigma study, 15 minutes, checkpoints
  every 120 s, resumes on a second call); outputs outside the repository in
  `$HOME/cuadmm-experiments/2026-10-21_phase4b_screen/` (the pinned replica copy and its sha256, `runs/`, `logs/`,
  `results/`, `specs/*.pending.*`). Gates and tables: `tools/sigma_screen/analyze.py` (G1 no problem lost against
  fixed sigma = 1 and the old rule at the same sigma_0; G2 the changes stop; G3 no change followed by a >= 10x rise
  of the validated error; G4 a better shifted geometric mean (shift 10 s) of the estimated GPU time than both
  residual_balancing and fixed sigma = 1; G5 robust across sigma_0).
- Commands (from `cuADMM-reorg/tools/sigma_screen`, Python = `~/cuadmm-analysis-env/bin/python -B`):
  `make_specs.py stage1 specs/stage1.txt`; `make_specs.py comparators specs/comparators.txt`;
  `SCREEN_ROOT=$HOME/cuadmm-experiments/2026-10-21_phase4b_screen ./screen.sh specs/comparators.txt 24 00:15:00`;
  `... ./screen.sh specs/stage1.txt 40 00:15:00`; `srun -c 1 ... analyze.py specs/stage1.txt specs/comparators.txt`.
- Submitted: smoke test (array 54488454, two short runs, both finished in 13 s); comparators array 54489077 (63 tasks,
  24 at a time) and Stage 1 array 54489148 (324 tasks, 40 at a time), 2026-10-21 21:16. Projected cost: at most
  about 20 core-hours for Stage 1 if every run hit the cap, less in practice.
- Stage 1 and comparators: all 387 runs finished (21:16-21:53, 32 core-hours of 1-core cascadelake tasks including
  queue overlap; results `results/stage1_{runs,settings}.csv`, `results/stage1_report.md` in the screen directory).
  Comparators at the 200,000 cap: fixed sigma = 1 solves trto1 (60,150) and buck1 (119,850); the old rule
  (legacy_adaptive) solves vibra1, mater-1, mater-2 at every sigma_0 (sigma_0 = 1: 68,650 / 99,850 / 43,550, through
  Monitor1 jumps of sigma to 100 on vibra1); residual_balancing solves mater-1 and mater-2 (16,300 / 36,650); nobody
  solves trto2, buck2, vibra2, shmup1. Shifted geometric means of the estimated H200 time (unsolved runs censored at
  the cap): fixed 132 s, residual_balancing 109-111 s, old 114-132 s; spikes >= 10x after a change: old 71-74 per
  sigma_0 (Monitor1), residual_balancing 2 at sigma_0 = 0.5, none otherwise.
  balance_freeze: `freeze_iteration` never matters (every run froze long before 10,000: by ping-pong around
  iteration 1,000-1,250 with the constant step, by the "step" rule at 150-200 with halve_on_reversal), so the twelve
  settings collapse to four. factor 1.2 + halve_on_reversal: solves trto1, buck1, mater-1, mater-2 at every sigma_0
  (40-42k, 93-116k, 15.5-16.1k, 46-49k iterations), total variation <= 9.4, largest spike 2.6x, shifted geometric
  mean 89.7-91.9 s (better than both comparators: G4 yes), but loses vibra1 (G1 no). factor 2 + halve_on_reversal:
  the same four problems, 110-113 s (G4 no). factor 2 + constant: loses vibra1 and, at sigma_0 = 1, trto1
  (115-125 s). factor 1.2 + constant: loses buck1 and vibra1 (98-110 s). G2 and G3 hold for every setting (all
  runs frozen or solved, no spike >= 10x). vibra1 is the problem of the plan's Table 2 remark (the duality gap
  dominates and the best fixed sigma is 10-39x above the rule's late average): every balance_freeze run froze at
  sigma in [0.47, 2.4] and did not reach the strict stage within 200,000 iterations (best long-horizon error 6e-4
  with factor 1.2 + halve), while fixed sigma = 1 does not solve it within the cap either.
- Stage 2 (freeze_value {band centre, last} x freeze_bias {1, 2, 4} for the two best Stage-1 settings,
  factor 1.2 + halve_on_reversal and factor 2 + halve_on_reversal, at sigma_0 = 1; 90 runs, array 54498335,
  21:55-22:12; `results/stage2_*`): the bias is what matters. factor 1.2 + halve_on_reversal + bias 4 solves all
  five problems the comparators solve (trto1 97,750, buck1 121,900, vibra1 80,800, mater-1 73,300, mater-2
  163,100 iterations): G1 yes, G2 yes (every run frozen by the "step" rule at iterations 150-200; total variation
  <= 11.4), G3 yes (largest spike 7.4x, none >= 10x). Its shifted geometric mean of the estimated H200 time is 111.5 s
  against residual_balancing 111.0 s and fixed sigma = 1 131.7 s, i.e. G4 fails by 0.4% on a measure in which the
  comparators are censored at the cap on 7 of 9 problems (their true times are larger) and the four problems nobody
  solves enter identically; on the five solvable problems alone: bias 4 45.0 s, residual_balancing 44.7 s (three of
  the five censored), old rule 46.7 s (two censored), fixed 62.7 s (three censored). Band centre and last value give
  the same runs here (the step freeze comes before eight changes). Bias 2 does not rescue vibra1 (frozen at
  sigma = 2.69 the long-horizon error stalls at 2.7; at 1.34 it reaches 6e-4, at 5.38 it converges) and bias 4 costs
  2-4x more iterations on trto1, mater-1 and mater-2 than bias 1 (whose geometric mean is 89.7 s). factor 2 +
  halve_on_reversal with bias 2 or 4 loses mater-2 (and mater-1 at bias 4): the coarser band freezes too high.
- Outcome of the CPU screen (plan: "if no setting passes all gates, adopt the one that loses the fewest problems,
  document the losses"): `balance_freeze` with factor 1.2, `step_schedule halve_on_reversal`, `freeze_bias 4`
  (`freeze_value band_centre`, the other parameters at their defaults; K_f irrelevant) loses no problem against the
  comparators at the 200,000-iteration cap at sigma_0 in {0.5, 1}, keeps the variation bounded, has no spike, and
  ties the time gate (DECISIONS 36). Robustness (36 runs, array 54504903, 22:14-22:24; `results/final_*`): at
  sigma_0 = 0.5 the same five problems are solved (trto1 106,900, buck1 120,600, vibra1 81,400, mater-1 70,200,
  mater-2 170,150; shifted geometric mean 113 s against residual_balancing 109 s); at sigma_0 = 2 vibra1 is lost
  again (the step freeze comes at iteration 676 with sigma = 2.50, 4.9-5.4 at the other sigma_0; the long-horizon
  error stalls at 4.4) and mater-2 takes 181,900 iterations, so G5 (robust across sigma_0) fails on vibra1 at
  sigma_0 = 2; band centre and last value give identical runs at every sigma_0. The default `penalty.policy` stays
  `residual_balancing` (user instruction: the promotion waits for the GPU gates, which need approval); the GPU gate
  scripts are generated for this setting. Cost of the screen: 513 runs, 28.8 core-hours of replica time (the plan's
  estimate was 15-20) on 1-core cascadelake tasks.
- Table 2's sanity check of the ping-pong window W in {6, 8, 12}: not run as separate runs, because for the chosen
  setting (halve_on_reversal) the "step" freeze acts at iterations 150-700, before `pingpong_min_iter` = 1000, so
  W cannot change those runs; with the constant step the ping-pong freeze acted at 1000-1250 with W = 8 (the
  earliest allowed), and a smaller or larger W would move it only later. `freeze_iteration` likewise never acted.
- GPU gates (plan: G0-G3, at most 12.6 H200-hours): prepared, not submitted. `tools/sigma_screen/gpu_gates.sh
  <cuadmm_exe> "<flags of the chosen setting>"` writes one sbatch file per gate (partition `gpu_h200`,
  `--gres=gpu:nvidia_h200:1`, 16 cores, 96-128 GB; the campaign protocol: tolerance 1e-4, validation every 50
  iterations, strict stage max|DIMACS| <= 1e-6, time caps) and prints the projection. Caps: G0 trto1, buck1,
  pendulum N=80 at fixed sigma = 1 (600, 600, 900 s: 0.58 h); G1 trto1, buck1 (600 s) and vibra1, mater-1, mater-2,
  shmup1 (1800 s) x {balance_freeze, residual_balancing, fixed sigma = 1} (7.0 h); G2 pendulum N=80 at sigma_0 in
  {1, 100} x {balance_freeze, residual_balancing} (900 s: 1.0 h; the comparators 124 s old rule and 625 s fixed
  sigma = 1 are the paper-replication runs); G3 mater-3..6 x {balance_freeze, residual_balancing} (1800 s: 4.0 h).
  Sum of the caps 12.58 H200-hours; runs that converge stop earlier. Generated for the screen's choice
  (`gpu_gates.sh $HOME/cuadmm-builds/reorg-f10e149/cuadmm_exe "--factor 1.2 --step-schedule halve_on_reversal
  --freeze-bias 4"`) into `$HOME/cuadmm-experiments/2026-10-21_phase4b_screen/gpu_gates/G{0,1,2,3}.sbatch`.
  Expected from the screen's iteration counts and the campaign's H200 seconds per iteration: G0 about 0.2 h
  (trto1 23 s, buck1 69 s, pendulum 625 s); G1 about 4.3 h (balance_freeze 0.6 h: trto1 38 s, buck1 70 s, vibra1
  46 s, mater-1 23 s, mater-2 61 s, shmup1 at its 1800 s cap; residual_balancing 1.7 h and fixed sigma = 1 2.0 h,
  mostly at their caps); G2 and G3 unknown (at most 1.0 h and 4.0 h): about 6 to 9.5 H200-hours in total. The jobs
  need the user's approval before submission (H200, and G1/G3 exceed one hour).

### GPU evidence of the Pre-0/Phase 1 and Phase 2 commits (clean per-commit builds, MIG configuration)

- f8de01f (Phase 1), `tools/slurm/build_and_compare.sh f8de01f phase1 ... legacy@pre0` (16:39-16:59): comparison with
  the references on every baseline case 49 of 49 identical, 0 different, 0 failed runs (fixed 21 against 2d8a755,
  legacy_adaptive 10 against 9bb4ba5, residual_balancing 10 against 2d8a755, sGS 8 against 2d8a755-sgs). Tests:
  101 ran, 99 passed, 2 failed: `FixedSigmaCli.BackwardCompatibleAndNewFlags` and `HybridSwitchCli.SgsIterationsOption`,
  both expectation mismatches of that commit's own CLI output format (the 17-digit printing writes `1.3500000000000001`
  and `1.6180000000000001` where the tests expect `1.35` and `1.618`); the golden test `FixedSigma.
  LegacyAdaptiveAndPlainAdmmUnchanged` passes (B14 fixed by Pre-0). The formatting is fixed from Phase 2 on by the
  shortest round-trip printing (`std::to_chars`).
- b44596b (Phase 2), `build_and_compare.sh b44596b phase2 ... reorg` (the new CLI, 16:43-17:03): 49 of 49 identical,
  0 different, 0 failed runs; tests 101 ran, 99 passed, the same 2 formatting expectations failed (the commit's tests
  still expected the 17-digit form in one place and the shortest form in the other; both fixed in the working tree
  that the Phase 4 commit carries, where `cuadmm_cpu_tests` and the CLI tests use the shortest round-trip form).
- 59e943a (Phase 3 + 4), `build_and_compare.sh 59e943a phase4 ... reorg` (20:50-21:00): 53 of 53 identical, 0
  different, 0 failed runs (fixed 21 against 2d8a755, legacy_adaptive 10 against 9bb4ba5, residual_balancing 10
  against 2d8a755, sGS 12 against 2d8a755-sgs: 8 fixed-sigma sGS cases, the sGS-phase adaptive rule on PSD_Free,
  two hybrid cases with the switch at 101 and 301, and one sGS run to tolerance 1e-5 with the reduction rule on).
  Tests: 101 ran, 99 passed, 2 failed, the same two CLI formatting expectations as before
  (`FixedSigmaCli.BackwardCompatibleAndNewFlags` expected `"sigma_max": 1000000` where `std::to_chars` wrote `1e+06`;
  `HybridSwitchCli.SgsIterationsOption` expected `1.618` in the history CSV, which prints 17 significant digits).
  Both fixed in d6720b6 (`json_number` prints integral values in fixed notation; the hybrid test parses tau;
  CHANGES.md Phase 4b): the two tests pass on the MIG with the rebuilt working tree (21:17).
- 92dca50 (d6720b6 + the Phase 4b preparation), `build_and_compare.sh 92dca50 phase4b ... reorg` (21:22-21:35,
  MIG configuration): 53 of 53 identical, 0 different, 0 failed runs; tests 101 ran, 101 passed.
- 92dca50 on the full A100 (`compare_build.sh ... phase4b reorg` in the same allocation 54480621, 21:28-21:50):
  53 of 53 identical, 0 different, 0 failed runs; tests 101 ran, 101 passed (21:41).
- 4640e8a (Phase 2b merged, B23, the JSON fix): the clean per-commit build failed to compile (`H5Treclaim` is
  not in the system HDF5 1.10.5 that the AUTO detection finds without `HDF5_ROOT`); fixed by the Phase 2b review
  fixes (9907cda) before any run.
- f10e149 (the second merge of `reorg-io`: review fixes), `build_and_compare.sh f10e149 phase2b2 ... reorg`
  (22:06-22:27, MIG configuration, clean configure without `HDF5_ROOT`: HDF5 1.10.5 and zlib from the system, no
  matio): 53 of 53 identical, 0 different, 0 failed runs; tests 101 ran, 101 passed. The format acceptance with
  that build (`check_formats.py`, `$HOME/cuadmm-baselines/<MIG config>/f10e149_formats/`): the 9 cases its formats
  support pass (trto1 and mater-1 objectives from TXT, `.dat-s`, `.h5`; buck3 `.dat-s` and `.h5` bit for bit the
  TXT run); the two SeDuMi cases need matio (`--formats` of that build: txt sdpa sdpa.gz h5) and passed with the
  matio build of the same tree (`dev_36e8552/formats3`, 11 of 11, recorded under Phase 2b).
- f10e149 on the full A100 (`a100-sxm4-80gb_cuda12.9_gcc12.2_ss7.10.1_t16`, batch job 54695120, 15:09-15:24,
  holygpu8a22206): 53 of 53 identical, 0 different; tests 101 ran, 101 passed; the format acceptance with the
  matio build 11 of 11 (SDPA, HDF5 and SeDuMi histories bit for bit the TXT runs). Both pinned configurations
  therefore cover the final Milestone 1 commit.
- 1cca25e (Phase 5, the cone layer), `build_and_compare.sh 1cca25e phase5 ... reorg` (2026-10-22 21:24-21:36, MIG
  configuration): 53 of 53 identical, 0 different, 0 failed runs; tests 118 ran, 118 passed (the 17 cone-layer tests
  included).
- 14178fa (Phase 6a step 1, the kernels moved in), `build_and_compare.sh 14178fa phase6a1 ... reorg` (2026-10-22
  21:45-21:56, MIG configuration): 53 of 53 identical, 0 different, 0 failed runs; tests 118 ran, 118 passed.
- Note on the summary comparison of `compare_runs.py`: the `options` object of the new CLI's summary was invalid
  JSON from Phase 2 until the fix recorded in CHANGES.md ("Phase 2b follow-up"), and the tool skipped the unreadable
  summary silently; the summary comparison is informational, the acceptance (histories and solutions, bit for bit)
  never depended on it. The tool now warns; the summaries of every later run parse.
- Second configuration `a100-sxm4-80gb_cuda12.9_gcc12.2_ss7.10.1_t16` (partition `gpu`, job 54480621, from 21:08,
  `tools/slurm/full_a100_session.sh 54480621 59e943a phase4 reorg`): Phase 0 baselines 59 of 59 identical
  (6 self-checks and the 53 reference cases, 16 threads); the comparison of 59e943a is in progress and is recorded
  here when it completes.
