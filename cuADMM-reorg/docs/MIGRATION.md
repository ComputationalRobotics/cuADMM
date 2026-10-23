# Migrating from cuADMM main to 2.0

What changes for a user of the solver of `main` (c23e67e): the entry points, the option names, and the defaults
that moved. `docs/CHANGES.md` lists every behavior change with the option that restores the old behavior;
`docs/OPTIONS.md` is the option table.

## Entry points

| main | 2.0 |
|---|---|
| `cuadmm_exe <problem dir> --sig 100 --sigma-policy legacy_adaptive ...` | `cuadmm_exe <problem> --sigma0 100 --policy legacy_adaptive ...`; the flags of main are accepted as aliases (`--sig`, `--sigma-policy`, `--switch-admm`, `--streams`, `--lobpcg`, `--switch-proj-iter`, `--switch-proj-tol`, `--algorithm`) |
| `SDPSolver solver; solver.init(...); solver.solve(max_iter, stop_tol, ...)` (one class, 20 positional arguments) | `cuadmm::Solver solver(options); solver.setup(problem); cuadmm::SolveResult r = solver.solve();` with `cuadmm::SolverOptions` (the option paths of the table, `set_option(options, "psd.large.warmup", "none")`) and `cuadmm::ProblemData` (`cuadmm::io::read_problem(path, io_options)` reads every format) |
| `cuadmm_MATLAB(eig_stream_num_per_gpu, max_iter, stop_tol, At_stack, b, C_stack, blk_types, blk_vec, X_new, y_new, S_new, sig_new, sig_update_threshold, sig_update_stage_1, sig_update_stage_2, switch_admm, switch_proj_iter, switch_proj_tol, sigscale)` | `[X, y, S, info] = cuadmm.solve(At, b, C, K, opts)` (the `+cuadmm` package of `matlab/`, built with `CUADMM_BUILD_MATLAB`): `K` is the cone description, `opts` a struct of option paths (`opts.algorithm.max_iter`, `opts.penalty.sigma0`, ...) with the warm start as `opts.warm_start.X/y/S` |

The problem formats: the TXT directory of main, the native HDF5 file (`cuadmm-convert` writes it), SDPA
`.dat-s[.gz]`, SeDuMi / SDPT3 / MOSEK `.mat` (`--formats` lists the readers of the build; `--sense`, `--simplify`,
`--save-format` in `docs/OPTIONS.md`). The outputs: `--summary` (JSON), `--history` (CSV), `--sigma-log`,
`--save-solution` (X, y, S in original units).

## Option names

| main | 2.0 | Note |
|---|---|---|
| `--sig` | `penalty.sigma0` (`--sigma0`) | |
| `--sigma-policy fixed / legacy_adaptive / fixed_sgs_then_legacy_admm` | `penalty.policy fixed / legacy_adaptive / residual_balancing / balance_freeze` | `legacy_adaptive` is the rule of 9bb4ba5 with Monitor1 (the published PLATO runs); `residual_balancing` the October 13 rule of 2d8a755 (the default); the hybrid mode's fixed-then-legacy policy is `penalty.policy fixed` with `algorithm.sgs_iterations` |
| `--sigscale`, `--sig-update-*` | `penalty.sgs_factor`, `penalty.sgs_update_threshold / stage_1 / stage_2` | the sGS phase's own rule of `legacy_adaptive` |
| `--switch-admm n` | `algorithm.scheme_switch_iteration n` or `algorithm.sgs_iterations n-1` | sGS-ADMM for the iterations before n, plain ADMM afterwards; `algorithm.scheme sgs_admm` runs sGS-ADMM throughout |
| `--lobpcg 0/1` | `psd.lowrank.enabled` | the rank analysis and LOBPCG of the large blocks (`psd.lowrank.*`: ratio, relaxation, reevaluate, max_iter, tol, rank tolerances) |
| `--switch-proj-iter`, `--switch-proj-tol` | `psd.large.warmup.switch_iter`, `psd.large.warmup.switch_tol` | the end of the composite warm-up; the low-rank analysis starts at `psd.lowrank.start_iter` (2) for the blocks without a warm-up |
| `--streams` | `psd.medium_streams` | at least 1 (B17) |
| `init(..., COMPOSITE_FP16, ...)` (C++ API only) | `psd.large.warmup composite_fp16` with `psd.large.warmup.min_size` | the default of 2.0 for every front-end, on the blocks of at least 5000 |
| (none) | `psd.large.final_product`, `psd.batched.final_product` | `gemm_compat` restores main's products bit for bit |
| (none) | `linsys.backend`, `linsys.presolve`, `linsys.regularization` | the normal-equation backends and the presolve of Phase 7 |
| (none) | `validation.*`, `certificate.*`, `output.*`, `seed`, `determinism` | the external validation, the certified lower bound, the outputs |

## Defaults that changed

- The penalty rule: `residual_balancing` (2d8a755) instead of the legacy adaptive rule; `--policy legacy_adaptive
  --sigma0 1` reproduces the published PLATO command. The step length tau is 1.618 for plain ADMM and the sGS
  phase's hidden reduction rule is off (`algorithm.sgs_tau_reduction`).
- The composite warm-up is on for the PSD blocks of at least 5000 (main's CLI and MATLAB never warmed up; main's
  C++ API warmed up every large block): `--psd-warmup none` restores the exact projections.
- The final products of the projections: SYRK on the smaller eigen-side and the scaled strided GEMM (within
  rounding of main): `--psd-final-product gemm_compat --psd-batched-final-product gemm_compat` restores main's.
- The LOBPCG tolerance and the rank-analysis parameters are options (the values of main are the defaults).
- The dense-factor rule of the y-step and the CHOLMOD factorization are unchanged (`linsys.backend auto`).
- The sigma bounds and the win-ratio hysteresis of the adaptive rules are options with the values of main.

## The C++ API and the engine

`SDPSolver` of main survives as the ported engine (`src/legacy/`) behind the facade: its `init()` / `solve()`
signatures are kept for the copied tests and the MEX of main, with main's arithmetic (the compat products), but
new code should use `cuadmm::Solver`. The pieces that moved out of it: the ADMM loop (`admm::AdmmSolver`), the
penalty policies, the stopping rules and the best-iterate tracker, the cone layer with the PSD tiers and their
memory (`cone::`), the normal-equation backends (`linsys::`), the readers and writers. Phase 8 (the engine's
remaining vectors, scaling and histories) and Phase 9 (the second-order, rotated, exponential and power cones)
are deferred.

## Verifying a migration

The reference builds of main reproduce the 2.0 runs bit for bit with `--psd-warmup none --psd-final-product
gemm_compat --psd-batched-final-product gemm_compat` on the baseline cases (`tools/compare_runs.py`,
`docs/PROGRESS.md`); with the 2.0 defaults the histories agree within rounding (`compare_runs.py ... --rtol 1e-6`).
