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
