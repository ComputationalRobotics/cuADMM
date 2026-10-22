# Behavior changes relative to cuADMM `main` (c23e67e)

Plan principle 4 and the developer prompt: no silent behavior change. Every change of results, option meaning or
default is listed here with the phase that introduced it and the new name or value that selects it. Structural
changes that reproduce the reference iterates bit for bit are not listed (they are verified by
`tools/compare_runs.py`; see `docs/PROGRESS.md`).

## Phase 0 (2026-10-21)

No behavior change: `cuADMM-reorg/` is a verbatim copy of `src/`, `include/`, `src/main.cu`, `MATLAB/` and the
used subset of `psd_projection` (b566590), built as its own project. The MEX target links the `cuadmm` library
instead of recompiling the sources.

## Pre-0 (2026-10-21): penalty policy names restored to the references

Inside `cuADMM-reorg/src/legacy/` (the copied `SDPSolver`), the CLI and the tests; plan Pre-0, decision Q1.

- `legacy_adaptive` is again the historical rule of commit 9bb4ba5, bit for bit: no guard (a change whenever the
  winner's count exceeds 1.35 times the other's), factor 2, sigma kept in [1e-6, 1e6], and the Monitor1 "chasing"
  correction after iteration 5000 every 25 iterations. On `main` since 2d8a755 the same name ran the October 13 rule,
  and since c23e67e with factor 1.2 (B18).
- `residual_balancing` (new name) is the October 13 rule of 2d8a755: guard `max(1, 1.35 other)`, factor
  `penalty.factor` (CLI `--penalty-factor`, default 2), sigma kept in [1e-3, 1e3], no Monitor1. The factor 1.2 of
  c23e67e is the option value `--penalty-factor 1.2`, never the meaning of a policy name.
- The default policy is `residual_balancing` (plan O1, until Phase 4b). A run of the old default (`main` at c23e67e,
  no `--sigma-policy`) therefore now runs the 2d8a755 rule with factor 2 instead of the 1.2-factor variant; the 1.2
  variant stays available as `--penalty-factor 1.2`.
- `fixed_sgs_then_legacy_admm` (the hybrid-only policy of `main`: fixed sigma during the sGS phase, then the
  plain-ADMM rule counted from the switch) is removed; the name is an error. It was neither reference rule, and the
  hybrid mode itself (`--sgs-iterations`, `--switch-admm`) stays available with every remaining policy. The two tests
  of that policy in `hybrid_switch_test.hpp` are removed with it.
- The bounds of the adaptive updates can be overridden (`--sigma-min`, `--sigma-max`; `penalty.sigma_min/max`).
  `sigma_0` is not clamped (as in both references; `balance_freeze`, Phase 4, clamps it as its pseudocode says).
- The summary JSON gains `sigma_policy_parameters` (factor, bounds, win ratio, guard, residual weight, Monitor1, sGS
  rule), `sigma_final_scaled`, `sigma_final_original`, `sigma0_original`, `build_project`, `build_commit` and
  `build_dirty` (the commit comes from a header generated at every build).
- The sigma log rule names: `admm_schedule`, `monitor1` (legacy_adaptive only), `sgs_rule`.

## Phase 1 (2026-10-21): cleanup and one-line fixes (plan Phase 1; results unchanged on the exact paths)

- **B5**: `CUADMM_PLAIN_ADMM_ONLY` is gone. sGS-ADMM and the hybrid are runtime options of the same build
  (`--algorithm sgs`, `--switch-admm`, `--sgs-iterations`), for the CLI, the tests and the MEX alike. The summary key
  `plain_admm_only_build` and the "Build: plain ADMM only" line are removed; `y_solves_per_iteration` reports 1 for
  plain ADMM and 2 for sGS-ADMM.
- **Q3 / B1**: the projection methods `COMPOSITE_FP32` and `COMPOSITE_FP32_EMULATED` are removed (their kernels are
  not copied from the submodule); the composite handle uses `CUBLAS_DEFAULT_MATH` instead of
  `CUBLAS_TENSOR_OP_MATH` (the FP16 filter sets its own compute type; the plan measured its output bitwise identical
  under both modes). No default path used either filter (the CLI and the MEX hard-code `EIG_FP64`).
- **B22**: a composite result whose Frobenius norm exceeds `(1 + 1e-2)` times the input's is rejected and the block
  falls back to the exact EVD (as a non-finite result already did). Composite path only.
- **B2**: the copies of the normalized `At` into `A_csr` wait on an event recorded after `get_normA` (no observable
  change when the race did not occur).
- **B6**: the dead decay `max(2 exp(-k/50000), 2)` of the legacy rule is the constant 2 (bitwise identical).
- **B16**: every kernel launch is followed by `CHECK_KERNEL_LAUNCH()` (`cudaGetLastError`), and the text reader throws
  `std::runtime_error` instead of calling `exit(1)` when a file cannot be opened.
- **B17**: `init()` rejects `eig_stream_num_per_gpu < 1`; the MEX validates every scalar input (real, finite,
  integers whole and in range, tolerances and sigma positive).
- **B21**: `init()` rejects PSD blocks larger than 32768 with a clear message (the exact-EVD limit of CUDA 12.9).
- **B12**: documentation: `init()`'s sigma default, the sigma-log residuals (original units), `lobpcg.h`'s `D`.
- **P4 (dead code)**: `analyze_blk`, `get_maps`, `get_eig_rank_mask`, `check_gpus`, `MatrixSizes::init`,
  `max_dense_vector_zero_mask` and the `mask`/`use_mask` parameters of `dense_matrix_mul_diag_batch`, the members
  `bscale2`, `Cscale2`, `sgs_KKT`, and the three no-op writes at the hybrid switch (`sig_update_stage_2 / 2`,
  `sigscale * 1.23`, `sgs_KKT`) are deleted. `analyze_blk`, `get_maps` and `MatrixSizes::init` survive as the test
  oracle `test/legacy_block_layout.hpp` of `block_structure_test.hpp`.

## Phase 2 (2026-10-21): options and front-ends

- **One option table** (`include/cuadmm/options.h`, `docs/OPTIONS.md`): the CLI flags are the table's names
  (`--policy`, `--sigma0`, `--scheme`, `--psd-medium-streams`, ...); today's flags are accepted as aliases for one
  release (`--sigma-policy`, `--sig`, `--algorithm admm|sgs`, `--switch-admm`, `--streams`, `--lobpcg`,
  `--switch-proj-iter`, `--switch-proj-tol`, `--save-solution`, `--validate-*`, `--sigscale`, `--sig-update-*`).
  Unknown names and values are errors.
- **Defaults that change (plan Section 15, "Defaults that change"; B3, B4):** the C++ API now has the CLI's
  defaults: `sigma0` 100 (the old `SDPSolver::init` default was 1.0), no composite warm-up (`psd.large.warmup none`;
  the old API default was `COMPOSITE_FP16` on every large block), the low-rank analysis from iteration 2
  (`psd.lowrank.start_iter 2`; the old API and MEX switched at 5000), the validation tolerance from `algorithm.tol`.
  The CLI's own behaviour is unchanged for equivalent options.
- **The warm-up and the low-rank start are separate options** (`psd.large.warmup.switch_iter/switch_tol` end a
  warm-up; `psd.lowrank.start_iter` starts the low-rank analysis of blocks without one). Today's single flag
  `--switch-proj-iter` is an alias of the warm-up end and has no effect without a warm-up.
- **`algorithm.tau`** is an option (default 1.618); until Phase 4 the ported engine accepts only 1.618 for `admm`
  and 1.95 for `sgs_admm` (the CLI sets 1.95 for `--scheme sgs_admm` unless `--tau` is given) and rejects others.
- **History and sigma-log CSV headers** use the plan's names: `iter,time_s,pobj,dobj,eta_p,eta_d,eta_g,sigma,tau,phase`
  and `iter,old_sigma,new_sigma,rule,eta_p,eta_d,eta_g,feasratio,prim_win,dual_win` (today: `errRp,errRd,relgap,sig`).
  Numbers are printed as the shortest decimal that round-trips exactly (B25).
- **Summary JSON**: the legacy keys are kept (`sigma_policy`, `switch_admm`, `errRp`, ...; `plain_admm_only_build` is
  gone since Phase 1) and new ones added (`scheme`, `tau`, `eta_*`, `format`, `sense`, `objective_offset`,
  `linsys_backend`, `sigma_changes_after_1e4`, `sigma_total_variation_log2`, `freeze_iteration`, `freeze_reason`,
  `options` with every option, `build_*`). `--sgs-iterations` must be non-negative (as today).
- **MATLAB** (not built here): `cuadmm.solve(At, b, C, K, opts)` on the option table with nested fields, and a
  positional shim `+cuadmm/solve.m` for the signature of `cuadmm_MATLAB` of main (one release, Q6). The old MEX source
  is replaced by `matlab/cuadmm_mex.cu`.
- **Reported objectives** carry the file's sense and offset (`pobj = sign * <C, X> + offset`); for TXT input (always a
  minimization without offset) nothing changes.

## Phase 3 (2026-10-21): thin core

- **One set of checks (B15)**: `src/core/check.h` defines `CUADMM_CHECK_CUDA/CUBLAS/CUSOLVER/CUSPARSE`,
  `CUADMM_CHECK_LAUNCH()` and the `_NOTHROW` variants, all throwing `cuadmm::Error` (a `std::runtime_error` with the
  call site and the failing call). The legacy header and the vendored submodule header map their old names onto
  them: the vendored kernels (LOBPCG, Lanczos, the FP16 filter) now throw on a library error instead of printing and
  continuing. Error paths only.
- `src/core/device_buffer.h` (`DeviceBuffer`, `DeviceSpan`, `PinnedHostBuffer`, host/device copies with `size_t`
  sizes) and `src/core/gpu_context.{h,cu}` (`GpuContext`: device, properties, main handles with the math mode set
  explicitly; `ExecContext`). The ported engine still creates its own streams and handles (the stream pool and the
  arena come with Phase 6b).

## Phase 4 (2026-10-21): algorithm layer

- `SDPSolver::solve()` of main is replaced by `cuadmm::admm::AdmmSolver::run` on `AdmmState` (the steps, moved
  unchanged), `IterationScheme` (`AdmmScheme`, `SgsAdmmScheme`), `PenaltyPolicy` (`FixedPenalty`, `LegacyPenalty`
  with Monitor1, `ResidualBalancingPenalty` for `residual_balancing` and `balance_freeze`), `StoppingRule`
  (`InternalKktRule`, `ValidatedRule` with the strict DIMACS stage), `BestIterateTracker`, and observers (console
  table, history, the engine's info arrays). The legacy `solve()` signature stays as a wrapper for the copied tests.
- **`balance_freeze`** is implemented (plan Section 3.5, Table 2: `penalty.freeze_iteration`, `.pingpong_*`,
  `.freeze_value`, `.freeze_bias`, `.step_schedule`, `.schedule`, `.win_ratio`, `.guard`, `.residual_weight`; `sigma_0`
  clamped). It is not the default (Phase 4b decides); the sigma log records every change and the freeze
  (`freeze_<reason>`), the summary the freeze iteration and reason.
- **The hidden sGS tau rule is an option**: `algorithm.sgs_tau_reduction` (default off) reproduces main's
  `tau = max(1.618, tau / 1.1)` at sGS iterations whose incoming `eta_d < tol`; off, `tau` is what the option says
  (plan Section 3.2). The legacy `SDPSolver::solve()` wrapper keeps it on, as main did, for the golden histories of
  the copied tests.
- **`algorithm.sgs_tau`**: the step length of the sGS iterations of the hybrid mode (main: 1.95 before the switch,
  1.618 after); default the same as `algorithm.tau`; the CLI sets 1.95 for `--sgs-iterations`/`--switch-admm` and
  for `--scheme sgs_admm` unless `--tau`/`--sgs-tau` are given, so that today's command lines give today's iterates.
- **Non-finite residuals** in the non-validating mode now throw at the end of the iteration that produced them, as
  before, but after the sigma update of that iteration was skipped (the error message is the same).
- **Best-iterate ranking with validations (B23)**: unchanged in this phase (the validated snapshots are ranked by
  `ValidationResult::better_than`: practical merit first, then `max_abs_dimacs` in the strict stage); the DIMACS-based
  ranking of plan B23 for the non-strict validation mode is recorded as pending in `docs/PROGRESS.md`.

## Phase 4b (2026-10-21): screen tools and two output-format fixes

- `report::json_number` prints integral values below 1e16 in fixed notation (`"sigma_max": 1000000`, previously
  `1e+06` from the shortest round-trip form); other values are unchanged (`1e-06`, `1.35`). Affects the JSON
  summary only (`sigma_policy_parameters`, `options`, timings that happen to be integral); histories and sigma logs
  keep 17 significant digits. Numerical results unchanged (d6720b6).
- `tools/sigma_screen/`: the CPU replica of plain ADMM (v1.2 = the sigma study's v1.1 plus the `balance_freeze`
  rule), its tests, the C++/replica cross-check generator, the screen driver and analysis (no solver change).
- Tests: `HybridSwitchCli.SgsIterationsOption` parses sigma and tau from the history CSV instead of matching the
  17-digit text; `PenaltyPolicy.BalanceFreezeAgreesWithTheReplicaOn{PsdFree,Mater1}` added with
  `test/data/golden/balance_freeze_cases_*.csv`.
