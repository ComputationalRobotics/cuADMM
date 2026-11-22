# Options of cuADMM

Generated from the option table of `src/options/options.cc` (`cuadmm_exe --print-options markdown`). The command line, MATLAB and the C++ API use the same table; unknown names are errors.

| Option | CLI flag | Type | Default | Meaning |
|---|---|---|---|---|
| `algorithm.tau` | `--tau` | double | `1.618` | multiplier step length, in (0, 2) |
| `algorithm.max_iter` | `--max-iter` | int | `1000000` | maximum number of iterations |
| `algorithm.time_limit` | `--time-limit` | double | `inf` | seconds of iterations after which the solve stops, returning the best iterate |
| `algorithm.tol` | `--tol` | double | `1e-4` | internal stopping tolerance on eta = max(eta_p, eta_d, eta_g) |
| `penalty.policy` | `--policy` | enum (fixed, balance_freeze) | `balance_freeze` | balance_freeze: sigma follows the residual vote at the checks of the schedule 200:10,1000:25,5000:50,10000:200 (up to iteration: period; one check per 1000 iterations after that) and is frozen at penalty.freeze_iteration; fixed: sigma = penalty.sigma0 throughout |
| `penalty.sigma0` | `--sigma0` | double | `100` | initial sigma (scaled units) |
| `penalty.sigma_min` | `--sigma-min` | double | `1e-3` | lower bound of the updates (balance_freeze) |
| `penalty.sigma_max` | `--sigma-max` | double | `1e3` | upper bound of the updates (balance_freeze) |
| `penalty.factor` | `--factor` | double | `1.35` | step of an update: sigma * factor up, sigma / factor down (balance_freeze) |
| `penalty.freeze_iteration` | `--freeze-iteration` | int | `10000` | sigma keeps its current value from this iteration on (balance_freeze) |
| `linsys.backend` | `--backend` | enum (auto, cholmod, cudss, dense) | `auto` | normal-equation backend: auto = cholmod (CHOLMOD on the host; a dense factor on the GPU when it fits), dense (cuSOLVER Cholesky on the device), cudss (cuDSS, when compiled in) |
| `linsys.regularization` | `--regularization` | double | `1e-15` | added to the diagonal of AA^T |
| `linsys.presolve` | `--presolve` | bool | `false` | remove dependent constraint rows before the factorization |
| `linsys.memory_fraction` | `--memory-fraction` | double | `0.25` | budget of a dense factor as a fraction of free memory |
| `certificate.block_bounds` | `--block-bounds` | double_list | `none` | per-block bounds R_b of the certified lower bound (comma-separated, one per block; a negative entry: no bound for that block, the bound is then unavailable) |
| `certificate.trace_bound` | `--trace-bound` | double | `none` | the same bound R for every block (paper eq. 30) |
| `certificate.trace_bound_scale` | `--trace-bound-scale` | double | `none` | R_b = f n_b for PSD blocks of size n_b, f for the entries of l and u blocks |
| `validation.interval` | `--validation-interval` | int | `0` | validation-aware stopping: validate every n iterations on the CPU with the original data and stop only on a validated snapshot (0: off) |
| `validation.time_interval` | `--validation-time-interval` | double | `0` | also validate every this many seconds of net solve time (the time spent in validations is excluded from the solve clock and the time limit; 0: off); alone it enables the validated stopping like validation.interval |
| `validation.tol` | `--validation-tol` | double | `algorithm.tol` | tolerance of every external criterion |
| `validation.threads` | `--validation-threads` | int | `1` | CPU threads of the external validation |
| `validation.strict_dimacs_tol` | `--strict-dimacs-tol` | double | `0` | after the validated criteria, continue until max |DIMACS error| <= x (0: stop at the criteria) |
| `validation.checkpoint_iterations` | `--checkpoint-iterations` | int_list | `none` | iterations with an extra validation snapshot |
| `validation.checkpoint_dir` | `--checkpoint-dir` | string | `none` | save X.txt, y.txt, S.txt of those snapshots in <dir>/iter_<k> |
| `validation.final_save_dir` | `--final-save-dir` | string | `none` | save the final snapshot of a solve that ends at a safety limit |
| `validation.history_file` | `--validation-history` | string | `none` | one CSV line per external validation |
| `io.format` | `--format` | enum (auto, h5, sdpa, sedumi, sdpt3, mosek, txt) | `auto` | input format (auto: from the extension and the .mat variables) |
| `io.sense` | `--sense` | enum (min, max) | `from the file` | objective sense (required for MOSEK files without a sense variable) |
| `io.simplify` | `--simplify` | bool | `true` | 1x1 and diagonal-only PSD blocks become l blocks (reported); never for TXT input |
| `io.warm_start_dir` | `--warm-start-dir` | string | `none` | read X.txt, y.txt, S.txt (dense, one value per line) from this directory |
| `io.warm_start` | `--warm-start` | bool | `false` | read X.txt, y.txt, S.txt from the problem directory (TXT input) |
| `io.save_dir` | `--save-dir` | string | `none` | write the returned X, y, S (and certificate.txt) in the input's convention |
| `io.save_format` | `--save-format` | string | `input's` | convention of the written solution: txt, h5, sdpa, sedumi, sdpt3, mosek (default: the input's) |
| `output.verbosity` | `--verbosity` | int | `1` | 0: quiet, 1: the iteration table, 2: details |
| `output.history` | `--history` | string | `none` | per-iteration CSV (17 significant digits) |
| `output.summary` | `--summary` | string | `none` | append a one-line JSON summary of the run |
| `output.sigma_log` | `--sigma-log` | string | `none` | one CSV line per sigma change |
| `output.tag` | `--tag` | string | `none` | label copied to the summary |
| `psd.small_max` | `--psd-small-max` | int | `32` | largest small PSD block (batched Jacobi; at most 32) |
| `psd.medium_max` | `--psd-medium-max` | int | `1000` | largest medium PSD block (batched or streamed EVD); the larger blocks form the large tier |
| `psd.medium_batch_min` | `--psd-medium-batch-min` | int | `2` | medium size groups with at least this many blocks use the batched EVD (0: never) |
| `psd.medium_streams` | `--psd-medium-streams` | int | `15` | CUDA streams of the medium eigendecompositions (>= 1) |
| `psd.large.memory_budget` | `--psd-large-memory-budget` | double | `0.9` | fraction of free memory for the large tier |
| `psd.large.warmup` | `--psd-warmup` | enum (none, composite_fp16, composite_fp32) | `composite_fp16` | low-precision warm-up of the large blocks of at least warmup.min_size: the FP16 (22 GEMMs, Tensor Cores) or native FP32 (31 GEMMs) composite filter until eta < switch_tol |
| `psd.large.warmup.min_size` | `--psd-warmup-min-size` | int | `5000` | smallest large block that gets the warm-up |
| `psd.large.warmup.switch_tol` | `--psd-warmup-switch-tol` | double | `1e-2` | the warm-up ends at the first iteration with eta below this |
| `psd.large.warmup.switch_iter` | `--psd-warmup-switch-iter` | int | `5000` | or after this many iterations |
| `psd.medium.max_batch` | `--psd-medium-max-batch` | int | `0` | blocks per batched EVD call of a medium size group (0: the whole group) |
| `psd.lowrank.enabled` | `--psd-lowrank-enabled` | bool | `true` | LOBPCG for large blocks of low positive or negative rank |
| `psd.lowrank.start_iter` | `--psd-lowrank-start-iter` | int | `2` | first iteration of the low-rank analysis for blocks without a warm-up (warm-up blocks: the end of the warm-up) |
| `psd.lowrank.reevaluate` | `--psd-lowrank-reevaluate` | int | `100` | exact EVD with rank analysis every this many iterations |
| `psd.lowrank.ratio` | `--psd-lowrank-ratio` | double | `0.05` | low-rank threshold as a fraction of n (LOBPCG while 0 < rank < ratio * n); sizes the warm start, ceil(n * ratio * relaxation) columns |
| `psd.lowrank.relaxation` | `--psd-lowrank-relaxation` | double | `1.2` | eigenpairs requested: ceil(relaxation * rank) |
| `psd.lowrank.max_iter` | `--psd-lowrank-max-iter` | int | `20` | LOBPCG iterations |
| `psd.lowrank.tol` | `--psd-lowrank-tol` | double | `1e-5` | LOBPCG convergence tolerance (absolute) |
| `psd.lowrank.rank_rtol` | `--psd-lowrank-rank-rtol` | double | `1e-8` | rank tolerance relative to max |lambda| |
| `psd.lowrank.rank_stop_factor` | `--psd-lowrank-rank-stop-factor` | double | `1e-3` | rank tolerance floor as a fraction of the stopping tolerance |
| `psd.composite.margin` | `--psd-composite-margin` | double | `1.05` | safety margin on the Lanczos estimate of the composite filter |
| `psd.composite.lanczos_steps` | `--psd-composite-lanczos-steps` | int | `20` | Lanczos steps of the norm estimate |
| `psd.composite.norm_slack` | `--psd-composite-norm-slack` | double | `1e-2` | norm test slack of the composite result |
| `psd.composite.on_reject` | `--psd-composite-on-reject` | enum (evd, keep, abort) | `evd` | a composite result that is not finite or fails the norm test: evd replaces it by the exact EVD, keep uses it anyway (counted as a rejection), abort stops the solve with an error |
| `psd.composite.dump_dir` | `--psd-composite-dump-dir` | string | `none` | directory of FP64 dumps of the composite projection input and output of every large block (column-major binary, a JSON line of metadata) at psd.composite.dump_iterations |
| `psd.composite.dump_iterations` | `--psd-composite-dump-iterations` | int_list | `none` | the iterations whose composite projections are dumped |
| `psd.trace` | `--psd-trace` | string | `none` | one JSON line per rank analysis of a large PSD block (eigenvalue counts of the projection input, mode), LOBPCG and composite rejection and warm-up switch, and per-block totals at the end |
| `seed` | `--seed` | int | `0` | seed of the deterministic generators (the Lanczos start vectors) |
| `gpu.device` | `--device` | int | `0` | CUDA device of the solver |
