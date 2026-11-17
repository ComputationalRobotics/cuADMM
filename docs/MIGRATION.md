# Migrating to cuADMM 2.1

What changes for a user of the 1.x solver (`main` at c23e67e) or of 2.0.x: the entry points, the option names and
the defaults. `docs/OPTIONS.md` is the option table; `docs/CHANGES.md` lists the changes of every version.

## Entry points

| 1.x | 2.1 |
|---|---|
| `cuadmm_exe <problem dir> --sig 100 --sigma-policy ... ` | `cuadmm_exe <problem> --sigma0 100 --policy balance_freeze ...` (the option flags of `docs/OPTIONS.md`; the flags of 1.x are not accepted) |
| `SDPSolver solver; solver.init(...); solver.solve(max_iter, stop_tol, ...)` | `cuadmm::Solver solver(options); solver.setup(problem); cuadmm::SolveResult r = solver.solve();` with `cuadmm::SolverOptions` (the option paths of the table, `set_option(options, "psd.large.warmup", "none")`) and `cuadmm::ProblemData` (`cuadmm::io::read_problem(path, io_options)` reads every format). `SDPSolver` is the internal engine; its headers are not part of the public `include/` directory. |
| `cuadmm_MATLAB(eig_stream_num_per_gpu, max_iter, stop_tol, At_stack, b, C_stack, blk_types, blk_vec, X_new, y_new, S_new, sig_new, ...)` | `[X, y, S, info] = cuadmm.solve(At, b, C, K, opts)` (the `+cuadmm` package of `matlab/`, built with `CUADMM_BUILD_MATLAB`): `K` is the cone description, `opts` a struct of option paths (`opts.algorithm.max_iter`, `opts.penalty.sigma0`, ...) with the warm start as `opts.warm_start.X/y/S` |

The problem formats: the TXT directory of 1.x, the native HDF5 file (`cuadmm-convert` writes it), SDPA
`.dat-s[.gz]`, SeDuMi / SDPT3 / MOSEK `.mat` (`--formats` lists the readers of the build; `--sense`, `--simplify`,
`--save-format` in `docs/OPTIONS.md`). The outputs: `--summary` (JSON), `--history` (CSV), `--sigma-log`,
`--save-dir` (X, y, S in original units).

## Option names

| 1.x flag | 2.1 option | Note |
|---|---|---|
| `--sig` | `penalty.sigma0` (`--sigma0`) | |
| `--sigma-policy` | `penalty.policy` (`--policy`): `balance_freeze` or `fixed` | the adaptive rules of 1.x are not available |
| `--switch-admm`, `--sigscale`, `--sig-update-*`, `--algorithm sgs` | none | 2.1 runs plain ADMM only |
| `--lobpcg 0/1` | `psd.lowrank.enabled` (`--psd-lowrank-enabled`) | with `psd.lowrank.*`: ratio, relaxation, reevaluate, max_iter, tol, rank tolerances |
| `--switch-proj-iter`, `--switch-proj-tol` | `psd.large.warmup.switch_iter`, `psd.large.warmup.switch_tol` | the end of the composite warm-up |
| `--streams` | `psd.medium_streams` (`--psd-medium-streams`) | at least 1 |
| `--validate-interval`, `--validate-tol`, `--validate-threads` | `validation.interval`, `validation.tol`, `validation.threads` (`--validation-*`) | |
| `--checkpoint-iters`, `--save-final-dir`, `--save-solution` | `validation.checkpoint_iterations`, `validation.final_save_dir`, `io.save_dir` (`--checkpoint-iterations`, `--final-save-dir`, `--save-dir`) | |

Options of 2.0.x removed in 2.1 (they are errors now): `algorithm.scheme`, `algorithm.sgs_tau`,
`algorithm.scheme_switch_iteration`, `algorithm.sgs_iterations`, `algorithm.sgs_tau_reduction` (sGS-ADMM and the
hybrid mode); the policies `legacy_adaptive` and `residual_balancing` with `penalty.sgs_update_*` and
`penalty.sgs_factor`; `penalty.step_schedule`, `penalty.win_ratio`, `penalty.guard`, `penalty.residual_weight`,
`penalty.schedule`, `penalty.pingpong_*`, `penalty.freeze_value`, `penalty.freeze_bias` (fixed at the values of the
2.0.x defaults: constant step, ratio 1.35 with the guard, weight 1, the check schedule
200:10,1000:25,5000:50,10000:200,inf:1000, no ping-pong freeze, the current value at the freeze);
`psd.large.final_product`, `psd.batched.final_product` (the SYRK / scaled products only); `psd.jacobi_tol`,
`psd.jacobi_max_sweeps`, `scaling.enabled`, `determinism`, `linsys.autotune` (fixed or not implemented). The defaults
of 2.0.1 at main 8fd203c run unchanged in 2.1: the same iterates, bit for bit.

## Defaults compared with 1.x

- The penalty rule: `balance_freeze` (factor 1.35 in [1e-3, 1e3], frozen at iteration 10000) instead of the
  adaptive rule of 1.x; the step length tau is 1.618.
- The composite warm-up is on for the PSD blocks of at least 5000 (the CLI and MATLAB of 1.x never warmed up):
  `--psd-warmup none` keeps the exact projections.
- The final products of the projections: SYRK on the smaller eigen-side and the scaled strided GEMM (within rounding
  of 1.x).
- The LOBPCG low-rank threshold is 0.05 n (1.x: 0.03 n); the LOBPCG tolerance and the rank-analysis parameters are
  options.
- The dense-factor rule of the y-step and the CHOLMOD factorization are those of 1.x (`linsys.backend auto`).
