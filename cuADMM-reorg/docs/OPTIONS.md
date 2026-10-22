# Options of cuADMM-reorg

Generated from the option table of `src/options/options.cc` (`cuadmm --print-options`). Every front-end uses the same table; unknown names are errors.

| Option | CLI flag (aliases) | Type | Default | Phase | Meaning |
|---|---|---|---|---|---|
| `algorithm.scheme` | `--scheme` (`--algorithm`) | enum (admm, sgs_admm) | `admm` | now | iteration scheme: plain two-block ADMM, or sGS-ADMM (one extra normal-equation solve per iteration); the alias --algorithm accepts admm and sgs |
| `algorithm.tau` | `--tau` | double | `1.618` | now | multiplier step length, in (0, 2); the legacy sGS path used 1.95 |
| `algorithm.sgs_tau` | `--sgs-tau` | double | `tau` | now | step length of the sGS iterations of the hybrid mode (the sGS phase of main used 1.95, then 1.618); tau: the same as algorithm.tau |
| `algorithm.scheme_switch_iteration` | `--scheme-switch-iteration` (`--switch-admm`) | int | `0` | now | hybrid mode (plan Q5): sGS-ADMM for the iterations before this one, then plain ADMM from the same X, y, S; 0: off |
| `algorithm.sgs_iterations` | `--sgs-iterations` | int | `0` | now | hybrid mode as a count: exactly n sGS-ADMM iterations, then plain ADMM (= scheme_switch_iteration n + 1); 0: plain ADMM |
| `algorithm.max_iter` | `--max-iter` | int | `1000000` | now | maximum number of iterations |
| `algorithm.time_limit` | `--time-limit` | double | `inf` | now | seconds of iterations after which the solve stops, returning the best iterate |
| `algorithm.tol` | `--tol` | double | `1e-4` | now | internal stopping tolerance on eta = max(eta_p, eta_d, eta_g) |
| `algorithm.sgs_tau_reduction` | `--sgs-tau-reduction` | bool | `false` | now | today's hidden sGS rule, tau = max(1.618, tau / 1.1) at sGS iterations whose incoming eta_d is below tol; off: tau is the option (plan) |
| `penalty.policy` | `--policy` (`--sigma-policy`) | enum (fixed, legacy_adaptive, residual_balancing, balance_freeze) | `residual_balancing` | now | fixed: sigma_k = sigma0; legacy_adaptive: the rule of 9bb4ba5 (no guard, factor 2, [1e-6, 1e6], Monitor1); residual_balancing: the October 13 rule of 2d8a755 (guard, factor, [1e-3, 1e3]); balance_freeze: residual balancing that freezes sigma (plan Section 3.5; Phase 4) |
| `penalty.sigma0` | `--sigma0` (`--sig`) | double | `100` | now | initial sigma (scaled units) |
| `penalty.sigma_min` | `--sigma-min` | double | `policy` | now | lower bound of the adaptive updates (policy: the policy's own) |
| `penalty.sigma_max` | `--sigma-max` | double | `policy` | now | upper bound of the adaptive updates (policy: the policy's own) |
| `penalty.factor` | `--factor` (`--penalty-factor`) | double | `2` | now | multiplicative step of residual_balancing and balance_freeze (1.2 = the October 20 edit of main) |
| `penalty.step_schedule` | `--step-schedule` | enum (constant, halve_on_reversal) | `constant` | 4 | balance_freeze: constant factor, or the factor's square root at every reversal |
| `penalty.win_ratio` | `--win-ratio` | double | `1.35` | now | hysteresis of the win counts (legacy_adaptive and residual_balancing: fixed at 1.35) |
| `penalty.guard` | `--guard` | bool | `true` | now | winner above max(1, win_ratio * other) (true) or above win_ratio * other (false); legacy_adaptive: false, residual_balancing: true |
| `penalty.residual_weight` | `--residual-weight` | double | `1` | now | weight on eta_p in the vote (today's ratioconst) |
| `penalty.schedule` | `--schedule` | schedule | `200:10,1000:25,5000:50,10000:200,inf:1000` | 4 | check iterations as (up to, period) pairs: a check at every k = 1 mod period while k <= up to (balance_freeze; the reference policies use their own fixed schedule) |
| `penalty.freeze_iteration` | `--freeze-iteration` | int | `policy` | 4 | balance_freeze: freeze sigma at this iteration at the latest (policy: 10000); residual_balancing: off |
| `penalty.pingpong_window` | `--pingpong-window` | int | `8` | 4 | balance_freeze: window W of the ping-pong test |
| `penalty.pingpong_net` | `--pingpong-net` | int | `2` | 4 | balance_freeze: freeze when |net movement over the window| <= this |
| `penalty.pingpong_min_iter` | `--pingpong-min-iter` | int | `1000` | 4 | balance_freeze: no ping-pong freeze before this iteration |
| `penalty.freeze_value` | `--freeze-value` | enum (last, band_centre) | `band_centre` | 4 | balance_freeze: frozen value when sigma bounces in a band |
| `penalty.freeze_bias` | `--freeze-bias` | double | `1` | 4 | balance_freeze: factor applied to the frozen value |
| `penalty.sgs_update_threshold` | `--sgs-update-threshold` (`--sig-update-threshold`) | int | `500` | now | sGS-phase rule of the adaptive policies: checks every stage_1 iterations up to this iteration, every stage_2 after |
| `penalty.sgs_update_stage_1` | `--sgs-update-stage-1` (`--sig-update-stage-1`) | int | `50` | now | sGS-phase rule: period before the threshold |
| `penalty.sgs_update_stage_2` | `--sgs-update-stage-2` (`--sig-update-stage-2`) | int | `100` | now | sGS-phase rule: period after the threshold |
| `penalty.sgs_factor` | `--sgs-factor` (`--sigscale`) | double | `2` | now | sGS-phase rule: factor of the updates |
| `scaling.enabled` | `--scaling` | bool | `true` | now | row normalization of A and the two scalars b_s, C_s (Milestone 1: always on) |
| `linsys.backend` | `--backend` | enum (auto, cholmod, cudss, dense) | `auto` | 7 | normal-equation backend (Milestone 1: auto = cholmod with today's dense-factor rule) |
| `linsys.regularization` | `--regularization` | double | `1e-15` | 7 | added to the diagonal of AA^T (Milestone 1: fixed) |
| `linsys.presolve` | `--presolve` | bool | `false` | 7 | dependent-row removal (Phase 7; off in the bitwise phases) |
| `linsys.memory_fraction` | `--memory-fraction` | double | `0.25` | 7 | budget of a dense or cuDSS factor as a fraction of free memory |
| `linsys.autotune` | `--autotune` | bool | `false` | 7 | time the two best backends and keep the faster |
| `certificate.block_bounds` | `--block-bounds` | double_list | `none` | now | per-block bounds R_b of the certified lower bound (comma-separated, one per block) |
| `certificate.trace_bound` | `--trace-bound` | double | `none` | now | the same bound R for every block (paper eq. 30) |
| `certificate.trace_bound_scale` | `--trace-bound-scale` | double | `none` | now | R_b = f n_b for PSD blocks of size n_b, f for the entries of l and u blocks |
| `validation.interval` | `--validation-interval` (`--validate-interval`) | int | `0` | now | validation-aware stopping: validate every n iterations on the CPU with the original data and stop only on a validated snapshot (0: off) |
| `validation.tol` | `--validation-tol` (`--validate-tol`) | double | `algorithm.tol` | now | tolerance of every external criterion |
| `validation.threads` | `--validation-threads` (`--validate-threads`) | int | `1` | now | CPU threads of the external validation |
| `validation.strict_dimacs_tol` | `--strict-dimacs-tol` | double | `0` | now | after the validated criteria, continue until max |DIMACS error| <= x (0: stop at the criteria) |
| `validation.checkpoint_iterations` | `--checkpoint-iterations` (`--checkpoint-iters`) | int_list | `none` | now | iterations with an extra validation snapshot |
| `validation.checkpoint_dir` | `--checkpoint-dir` | string | `none` | now | save X.txt, y.txt, S.txt of those snapshots in <dir>/iter_<k> |
| `validation.final_save_dir` | `--final-save-dir` (`--save-final-dir`) | string | `none` | now | save the final snapshot of a solve that ends at a safety limit |
| `validation.history_file` | `--validation-history` | string | `none` | now | one CSV line per external validation |
| `io.format` | `--format` | enum (auto, h5, sdpa, sedumi, sdpt3, mosek, txt) | `auto` | 2b | input format (auto: from the extension and the .mat variables) |
| `io.sense` | `--sense` | enum (min, max) | `from the file` | 2b | objective sense (required for MOSEK files without a sense variable) |
| `io.simplify` | `--simplify` | bool | `true` | 2b | 1x1 and diagonal-only PSD blocks become l blocks (reported); never for TXT input |
| `io.warm_start_dir` | `--warm-start-dir` | string | `none` | now | read X.txt, y.txt, S.txt (dense, one value per line) from this directory |
| `io.warm_start` | `--warm-start` | bool | `false` | now | read X.txt, y.txt, S.txt from the problem directory (TXT input) |
| `io.save_dir` | `--save-dir` (`--save-solution`) | string | `none` | now | write the returned X, y, S (and certificate.txt) in the input's convention |
| `io.save_format` | `--save-format` | string | `input's` | 2b | convention of the written solution |
| `output.verbosity` | `--verbosity` | int | `1` | now | 0: quiet, 1: the iteration table, 2: details |
| `output.history` | `--history` | string | `none` | now | per-iteration CSV (17 significant digits) |
| `output.summary` | `--summary` | string | `none` | now | append a one-line JSON summary of the run |
| `output.sigma_log` | `--sigma-log` | string | `none` | now | one CSV line per sigma change |
| `output.tag` | `--tag` | string | `none` | now | label copied to the summary |
| `psd.small_max` | `--psd-small-max` | int | `32` | 6a | largest small PSD block (batched Jacobi); Milestone 1: fixed |
| `psd.medium_max` | `--psd-medium-max` | int | `1000` | 6a | largest medium PSD block; Milestone 1: fixed |
| `psd.medium_batch_min` | `--psd-medium-batch-min` | int | `2` | now | medium size groups with at least this many blocks use the batched EVD (0: never) |
| `psd.medium_streams` | `--psd-medium-streams` (`--streams`) | int | `15` | now | CUDA streams of the medium eigendecompositions (>= 1) |
| `psd.jacobi_tol` | `--psd-jacobi-tol` | double | `1e-16` | 6a | tolerance of the batched Jacobi EVD; Milestone 1: fixed |
| `psd.jacobi_max_sweeps` | `--psd-jacobi-max-sweeps` | int | `15` | 6a | sweeps of the batched Jacobi EVD; Milestone 1: fixed |
| `psd.large.memory_budget` | `--psd-large-memory-budget` | double | `0.9` | 6b | fraction of free memory for the large tier (Phase 6b) |
| `psd.large.warmup` | `--psd-warmup` | enum (none, composite_fp16) | `none` | now | low-precision warm-up of the large blocks (composite_fp16 becomes the default in Phase 6a) |
| `psd.large.warmup.min_size` | `--psd-warmup-min-size` | int | `5000` | 6a | smallest block that gets the warm-up (Milestone 1: the warm-up applies to every large block, as the C++ API did) |
| `psd.large.warmup.switch_tol` | `--psd-warmup-switch-tol` (`--switch-proj-tol`) | double | `1e-2` | now | the warm-up ends at the first iteration with eta below this |
| `psd.large.warmup.switch_iter` | `--psd-warmup-switch-iter` (`--switch-proj-iter`) | int | `5000` | now | or after this many iterations |
| `psd.lowrank.enabled` | `--psd-lowrank-enabled` (`--lobpcg`) | bool | `true` | now | LOBPCG for large blocks of low positive or negative rank |
| `psd.lowrank.start_iter` | `--psd-lowrank-start-iter` | int | `2` | now | first iteration of the low-rank analysis for blocks without a warm-up (warm-up blocks: the end of the warm-up) |
| `psd.lowrank.reevaluate` | `--psd-lowrank-reevaluate` | int | `100` | 6a | exact EVD with rank analysis every this many iterations; Milestone 1: fixed |
| `psd.lowrank.ratio` | `--psd-lowrank-ratio` | double | `0.03` | 6a | low-rank threshold as a fraction of n; Milestone 1: fixed |
| `psd.lowrank.relaxation` | `--psd-lowrank-relaxation` | double | `1.2` | 6a | eigenpairs requested: ceil(relaxation * rank); Milestone 1: fixed |
| `psd.lowrank.max_iter` | `--psd-lowrank-max-iter` | int | `20` | 6a | LOBPCG iterations; Milestone 1: fixed |
| `psd.lowrank.tol` | `--psd-lowrank-tol` | double | `1e-5` | 6a | LOBPCG tolerance (absolute today, B13); Milestone 1: fixed |
| `psd.lowrank.rank_rtol` | `--psd-lowrank-rank-rtol` | double | `1e-8` | 6a | rank tolerance relative to max |lambda|; Milestone 1: fixed |
| `psd.lowrank.rank_stop_factor` | `--psd-lowrank-rank-stop-factor` | double | `1e-3` | 6a | rank tolerance floor as a fraction of the stopping tolerance; Milestone 1: fixed |
| `psd.composite.margin` | `--psd-composite-margin` | double | `1.05` | 6a | safety margin on the Lanczos estimate; Milestone 1: fixed |
| `psd.composite.lanczos_steps` | `--psd-composite-lanczos-steps` | int | `20` | 6a | Lanczos steps of the norm estimate; Milestone 1: fixed |
| `psd.composite.norm_slack` | `--psd-composite-norm-slack` | double | `1e-2` | 6a | norm test slack of the composite result (B22); Milestone 1: fixed |
| `seed` | `--seed` | int | `0` | now | seed of the deterministic generators (the Lanczos start from Phase 6a) |
| `determinism` | `--determinism` | enum (strict, fast) | `strict` | now | strict: deterministic SpMV and reductions; fast: opt-in nondeterministic modes (none yet) |
| `gpu.device` | `--device` | int | `0` | now | CUDA device of the solver |
