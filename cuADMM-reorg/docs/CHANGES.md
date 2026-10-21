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
