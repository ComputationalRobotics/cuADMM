# Behavior changes relative to cuADMM `main` (c23e67e)

Plan principle 4 and the developer prompt: no silent behavior change. Every change of results, option meaning or
default is listed here with the phase that introduced it and the new name or value that selects it. Structural
changes that reproduce the reference iterates bit for bit are not listed (they are verified by
`tools/compare_runs.py`; the evidence: `c4eb89b:cuADMM-reorg/docs/PROGRESS.md`, at that commit only).

> Written during the reorganization, when the project lived in `cuADMM-reorg/` beside the 1.x tree:
> `cuADMM-reorg/<path>` is today's `<path>`. The working records named below (`docs/PROGRESS.md`, the phase reports
> `docs/phase*.md` with their data, `tools/sigma_screen/`, `tools/slurm/build_and_compare.sh` and the other campaign
> scripts) were removed from the tree by the 2.0 cleanup; they exist at commit `c4eb89b`, for example
> `git show c4eb89b:cuADMM-reorg/docs/PROGRESS.md`.

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

## Phase 2b (2026-10-21): input and output formats (plan Section 12; B27, B28, B29)

- **New input formats**, all producing the same `ProblemData` (64-bit sizes, CSC of At filled directly, a
  conversion report in `source.report` printed by the CLI and stored in the summary as `conversion_report`):
  native HDF5 `.h5` (schema version 1), SDPA sparse `.dat-s` / `.dat-s.gz` (the checks of the PLATO converter
  `sdpa_to_cuadmm.py`: duplicates, non-finite values, indices out of range and empty constraint matrices reject the
  file; explicit zeros are dropped; lower-triangle entries are mirrored), and through matio SeDuMi, SDPT3 and MOSEK
  `prob` `.mat` files (MATLAB v5 and v7.3). Detection from the extension or the file signature and, for `.mat`,
  from the variables present; `io.format` (`--format`) forces one. The legacy TXT reader is unchanged (bitwise).
- **Objective sense and offset of the file** (`ProblemData::sense`, `objective_offset`): SDPA files are read as the
  maximization of `<F_0, Y>` that the published values refer to, so the reported `pobj`/`dobj` of a `.dat-s` run
  equal the published optimal value, whereas the TXT conversion of the same file (a minimization of `<-F_0, Y>`)
  reports its negative, as today. SeDuMi and SDPT3 files are minimizations. MOSEK files take the sense from a
  variable `sense` saved next to `prob` or from `io.sense` (`--sense min|max`) and are rejected without one; `cfix`
  and the bound shifts go to the offset. B28 is fixed: every scalar bound (free, nonnegative, shifted, negated,
  ranged, fixed) and every row kind (equality, one-sided, ranged, free) is handled.
- **Exact simplifications at read time** (`io.simplify`, `--simplify`, default on, never for TXT): 1x1 PSD blocks
  and PSD blocks whose A and C data touch only the diagonal (including untouched blocks) become `l` blocks; each is
  reported, and solutions are mapped back to the file's blocks. This changes the block structure (and so the
  iteration history) of problems read from `.mat`, `.dat-s` or `.h5` relative to a TXT conversion that kept the
  blocks: PushT_N=10_MOMENT (SPOT) goes from 6015 PSD blocks to 1202 PSD + 4813 `l` blocks; `--simplify 0` keeps
  the file's blocks.
- **Solution output in the input's convention** (`--save-dir`, `io.save_format` / `--save-format` to force one):
  TXT (as today), `.h5` (group `/solution`), SDPA (sparse initial-point layout, `x = -y`, `X_SDPA = S`,
  `Y_SDPA = X`, plus a JSON with `objValPrimal` / `objValDual`), SeDuMi (`x`, `y`, `z` full vectors), SDPT3
  (`X`, `y`, `Z` cell arrays, `obj`), MOSEK (`res.sol.itr` with `barx`, `bars` lower triangles, `xx`, `slx`,
  `sux`, `y`, `slc`, `suc`, `pobjval`, `dobjval`; for a maximization MOSEK's dual variables are negated). B29 is
  fixed: `l` blocks are written in every convention.
- **Warm starts** from a `.h5` file (`/start`) as well as a TXT directory (`io::read_warm_start`).
- **Tools**: `cuadmm-convert <input> <output.h5 | directory>` (any input to the native file or a TXT directory),
  `cuadmm_io_bench` (synthetic 1e8-nonzero problems and read rates). Build options `CUADMM_WITH_HDF5`,
  `CUADMM_WITH_MATIO`, `CUADMM_WITH_ZLIB` (AUTO/ON/OFF); `cuadmm_exe --formats` lists the formats compiled in.

## Phase 2b review fixes (2026-10-21)

- **Native format**: string attributes and `/meta/report` are read with the stored character set, so files
  written by h5py (variable-length UTF-8 strings, its default) and by MATLAB (fixed-length ASCII) both load;
  svec indices of `/At/rowind`, `/b/index`, `/C/index` are range-checked before anything else touches them (a
  malformed file used to crash the simplification scan); a missing `sense` attribute is an error unless
  `io.sense` is given (DECISIONS 35); `/solution` and `/start` are in the layout of the file's blocks, with
  `/solution/blocks` (DECISIONS 34); the HDF5 files are closed explicitly and a failed close is an error; the
  code builds against HDF5 1.10 as well as 1.12+ (`H5Dvlen_reclaim` / `H5Treclaim`).
- **MOSEK solution writer**: `slx` / `sux` of ranged variables come from the dual slacks of `x'` and of the bound
  slack `s` (`slx - sux = S_x' - S_s = c - a'y`), and those of eliminated fixed variables from `c_j - a_j'y`
  (`SourceInfo` keeps the bound-slack positions, the costs and the columns of the fixed variables; the `.h5`
  metadata too); every variable kind now satisfies MOSEK's dual constraint, for maximizations with MOSEK's
  non-positive sign convention (DECISIONS 32).
- **MOSEK reader**: quadratic terms (`qosubi/qosubj/qoval`, `qcsubk/qcsubi/qcsubj/qcval`) are rejected instead of
  being ignored.
- `.cbf` / `.cbf.gz` input is rejected by name ("CBF comes with Phase 9"); read errors on an SDPA text source are
  errors, not an end of file; the SDPA report counts a block as touched only by nonzero entries; the SDPA JSON
  sidecar and `certificate.txt` streams are checked; `solution_blocks.txt` accompanies the TXT solution.
- **Per-iteration values are in the solver's own sense.** The history CSV, the iteration log and
  `SolveResult::best.pobj / dobj` hold `<C, X>` and `b'y` of the minimization the engine solves, without the
  offset (as today, `result.h`); the final `pobj` / `dobj` of the summary and of the solution files are in the
  file's sense with the offset (`sign * value + offset`, sign -1 for SDPA `.dat-s` input and MOSEK
  maximizations). The two differ by that sign and offset only.

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

## Phase 4, B23 (2026-10-21): best-iterate ranking by the maximum DIMACS error

- `ValidationResult::better_than` (the ranking of the externally evaluated snapshots, which decides the iterate a
  validating solve returns when it ends without a validated snapshot, and the `best_validation` record) ranks every
  snapshot by its maximum absolute DIMACS error, validated ones first; the practical merit (normalizers
  `1 + ||b||_2`, `1 + ||C||_2`) decides only between snapshots without a finite DIMACS error. Before, non-validated
  snapshots were ranked by the merit (B23: the two measures differ by about 12x on the long-horizon problems, and
  3 of 4 long-horizon runs returned an iterate whose validated error was 1.19-6.1x that of the best one seen).
  Affects only solves with validation on (`validation.interval > 0`) that end at a limit; the bitwise comparisons
  run without validation. The restore message prints both measures. The internal tracker (no validation) still
  ranks by the solver's own max(eta_p, eta_d, eta_g).

## Phase 2b follow-up (2026-10-21): the JSON summary is always valid

- `options_json` (the `options` object of the summary, Phase 2) wrote the symbolic defaults of numeric options
  unquoted (`"algorithm.sgs_tau": tau`), so every summary of the new CLI since Phase 2 was invalid JSON. Numbers and
  booleans are now written verbatim only when they parse as such; every other value is a JSON string
  (`"tau"`, `"policy"`, `"inf"`, `"none"`, `"algorithm.tol"`). Found by the first version of the GPU objective
  check (since replaced by `tools/io/check_formats.py`), which parses the summary. `Options.OptionsJsonIsValidJson` checks the defaults and set values with a small JSON
  scanner. `tools/compare_runs.py` now prints a warning for a summary it cannot parse instead of skipping the
  (informational) summary comparison silently; the acceptance comparisons (histories, solutions) were never affected.

## Phase 7 (backends, branch reorg-linsys, 2026-10-22)

No behavior change: the new backends (`DenseCusolverSolver`, `CudssSolver`), `NormalSystem` and the presolve are
standalone components with their own tests and are not wired into the y-step; the engine's CHOLMOD path, the facade
and the existing tests are unchanged.

- New build option `CUADMM_WITH_CUDSS` (AUTO, ON, OFF; `-DCUDSS_ROOT=<prefix>`; `cmake/FindCUDSS.cmake`, minimum
  cuDSS 1.0): with AUTO the library links `libcudss.so[.1]` when found, which adds `libcudss.so.1`, `libcublas.so.12`
  and `libnvrtc.so.12` to the runtime dependencies of `libcuadmm.so`; OFF leaves cuDSS out and `CudssSolver` does not
  exist (`CUADMM_HAVE_CUDSS` undefined).
- New test executables `cuadmm_linsys_tests` (GPU) and `cuadmm_presolve_tests` (CPU) under `CUADMM_BUILD_TESTS`.
## Phase 5 (2026-10-22): cone layer

- The cone step runs through the product cone (`src/cone/`): the PSD cone projects the PSD entries with the
  unchanged tiers, the nonnegative and free cones their own entries with exact elementwise kernels (max(z, 0),
  0). Results unchanged (bitwise acceptance).
- Certificate (B24): `certificate.block_bounds` accepts a negative entry meaning "no bound for this block"; the
  certified lower bound is then unavailable (NaN) and the summary lists `certificate_missing_bounds` and a
  `certificate_note`; the summary also prints `certificate_block_bounds` and the bound computed from them.
  `certificate_conditional_on_free_bounds` is now set whenever a free block's bound is in force (also with explicit
  block bounds), not only with `trace_bound` / `trace_bound_scale`.
- `ConeRegistry` rejects the cones of Phase 9 ('q', 'r') and the exponential and power cones with a message naming
  the phase, before any cone object is built.

## Phase 6a, step 1 (2026-10-22): the PSD kernels moved in

- The composite warm-up is deterministic (B8): the Lanczos start vector of the norm estimate (and the LOBPCG cold
  start, which the solver never uses) comes from a counter-based generator seeded by (`seed`, block index) instead
  of `curand` with a time-based seed. Runs with `psd.large.warmup = composite_fp16` are reproducible; their
  iterates differ from the reference's, which were not reproducible. No baseline uses the warm-up (`none` is the
  CLI default): the exact and LOBPCG paths are bit for bit.
- `psd.lowrank.tol` and `psd.lowrank.max_iter` are honoured (previously fixed at 1e-5 and 20; the defaults keep
  those values).

## Phase 6a, step 2 (2026-10-23): the projector classes and the per-block policy

- `psd.large.warmup.min_size` is honoured: the composite warm-up applies to the large blocks with n >= min_size
  (Milestone 1 applied it to every large block, as the C++ API of main). Blocks below the threshold start their
  low-rank analysis at `psd.lowrank.start_iter` while the others warm up (plan Section 10.6). No baseline uses
  the warm-up.
- `psd.lowrank.{reevaluate, ratio, relaxation, rank_rtol, rank_stop_factor}` and `psd.composite.{margin,
  lanczos_steps, norm_slack}` are honoured (previously fixed at the values of main; the defaults keep them).
  `psd.large.final_product` exists with its Phase 6a value `gemm_compat`; `syrk` is rejected until Phase 6b.
- The result and the summary JSON carry `warmup_switch_iteration` (the first iteration after the composite
  warm-up; -1 without one).
- The console lines "Switching projection method" and "Number of low rank matrices / Ranks" follow the verbosity
  (`output.verbosity` 0 prints neither); the CLI output is unchanged.

## Phase 6a, step 3 (2026-10-23): the warm-up default

- `psd.large.warmup` defaults to `composite_fp16`: the large blocks of at least `psd.large.warmup.min_size` (5000)
  are projected with the FP16 composite filter until the KKT residual falls below `warmup.switch_tol` (1e-2) or
  `warmup.switch_iter` (5000) iterations, then exactly (plan Q2/Q3; the Phase 6a benchmark measured the filter 6-8x
  faster than the exact EVD from n = 2000 with a 1e-3 projection error and no rejection). Today's CLI and MATLAB
  never warmed up (`EIG_FP64`); the C++ API warmed up every large block. `--psd-warmup none` restores the exact
  projections. Runs with blocks of at least 5000 change (deterministically: B8); no baseline case has such a block.

## Phase 6b (2026-10-23): the PSD memory model

- The final products of the PSD projections change within rounding: the large blocks form P = B B^T with one SYRK
  on the smaller eigen-side (`psd.large.final_product = syrk`, the default; the negative side gives Z + P'), the
  LOBPCG result likewise, and the small and medium blocks scale the eigenvectors in place by sqrt(max(lambda, 0))
  and form P with one strided-batched GEMM (`psd.batched.final_product = scaled_gemm`, the default). The products
  of main are `gemm_compat` for both options (bit for bit the reference); the C++ API `SDPSolver::solve()` keeps
  them. The strided-batched product of equal-size large groups of main (one GEMM per group when every block
  took the exact path) is gone: the large blocks are projected one at a time through one arena region.
- Memory: the large tier holds one block at a time (M, W, the warm starts) instead of three buffers of
  sum n_b^2 doubles and one EVD workspace per block (B19); the medium tier holds one streamed workspace per
  stream instead of one per block and decomposes batched groups in chunks (B20); LOBPCG and the Lanczos estimate
  allocate nothing per call (B7). Setup prints the plan at verbosity >= 1, fails early above
  `psd.large.memory_budget` of the free memory, and the result / summary carry `memory_plan_bytes`
  (`memory_plan_mib`) next to the measured peak.
- New options: `psd.batched.final_product`, `psd.medium.max_batch`. The warm-up blocks rejected by the norm test
  are unpacked again from the input (no n^2 copy); the termination measures run on the tiers' decompositions
  (the engine's `cone_measures` is gone).

## Phase 7 (2026-10-23): the linear systems

- The y-step runs through the `NormalEquationSolver` interface. `linsys.backend` selects the backend: `auto` and
  `cholmod` are the CHOLMOD backend of main (host CHOLMOD, or the dense factor on the device by today's rule; bit for
  bit), `dense` is the cuSOLVER Cholesky on the device, `cudss` the cuDSS backend when the build has it (both within
  rounding of CHOLMOD: the same optimum, the iteration counts within a few percent). `linsys.regularization` (eps of
  A A^T + eps I) is honoured; 1e-15 remains the default.
- `linsys.presolve` removes dependent constraint rows before the factorization (off by default): the reduced
  problem is solved, y is reported on the full problem with zeros on the removed rows, and the result / summary
  carry `presolve_removed_rows` and the presolve's message.
- `linsys.autotune` is rejected (not implemented: DECISIONS 57).

## 2.0 cleanup (2026-10-23): the empty objective vector

- A problem with an empty `C` (or `b`), such as the PLATO instance `cnhil10` of `examples/plato/TXT`, failed at setup
  since the launch checks of Pre-0 (`CUDA API failed with error: invalid configuration argument` in
  `src/legacy/kernels/sparse_scalar.cu`): the scalar and sparse-dense kernels launched a block of size nnz = 0, which
  1.x issued and silently ignored. The three kernels now return early on an empty vector; the iterates are those of
  1.x bit for bit (cnhil10, 200 iterations of `residual_balancing` against the 2d8a755 reference, with the compat
  products). Test `EmptyObjective.SolverSetsUpAndIteratesWithoutC`.

## 2.0.1 (2026-10-26): acceptance tooling, provenance, license

- `tools/compare_runs.py` rejects what it previously skipped or forgave: a nonfinite value in a history or a
  solution, a history without one of the mandatory columns (`iter`, `pobj`, `dobj`, `eta_p`, `eta_d`, `eta_g`,
  `sigma`), a malformed line, a missing or unparsable summary, a missing solution file and a run whose
  `exit_code.txt` is nonzero are INVALID (exit code 2). The rows of two histories are aligned by their iteration
  id, and a mismatch is a difference. The bitwise comparison compares the bit patterns of the doubles (`-0.0` and
  `0.0` differ, NaN never equals NaN), separately from the tolerance comparison (`--rtol`).
  `tools/slurm/compare_build.sh` returns the worst batch exit code and 1 after a failed test suite (it returned 0);
  `run_tests.sh` prints the gtest summary lines. `tools/tests/test_compare_runs.py` (16 negative tests, standard
  library only) proves the rejections; `tools/recheck_retained.sh` rechecks retained outputs without the solver.
  The retained outputs of the three latest verification tags were rechecked with the repaired tool on 2026-10-26:
  53 of 53 identical with the compat products (`phase7-compat`, `phase6b-compat2`), and for the 2.0 defaults
  (`phase6b-default2`, rtol 1e-6) the same 8 identical and 45 different cases as recorded (52 of 53 final
  objectives within 1e-6, the exception a near-zero objective whose solution agrees to 4e-12 absolute); no invalid
  run, so no case needed rerunning.
- Historical references name immutable commits with exact paths (`c4eb89b:cuADMM-reorg/docs/PROGRESS.md` and the
  phase reports beside it, `c23e67e:reorg-plan/cuadmm_reorg_plan.tex`,
  `c23e67e:experiments/2026-09-29_plato_cold_plain_admm_sigma/scripts/sdpa_to_cuadmm.py`,
  `dd4babe:experiments/2026-09-28_fixed_sigma_to_tolerance/scripts/gen_legacy_golden.cu`, the last one on a local
  branch that is not on GitHub); the deleted tag `v2.0.0-rc1` is not referenced any more.
- `examples/max-cut/LICENCE`: the BSD 2-Clause license of Richard Y. Zhang's `dual_ctc` repository, from which the
  scripts come, reproduced verbatim; `examples/max-cut/README.md` gives the provenance file by file.
- No solver change; the defaults of 2.0 (Decision 37, `residual_balancing`) are unchanged. Project version 2.0.1.

