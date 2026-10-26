# Examples

Command lines on the new options (every option is in `docs/OPTIONS.md`; `cuadmm_exe --print-options` prints the
table of the build). `<problem>` is a TXT directory (the format of 1.x and of the data below), a native `.h5` file, an SDPA `.dat-s[.gz]` file, or a
SeDuMi / SDPT3 / MOSEK `.mat` file; `--formats` lists the readers compiled in. The scripts of this directory run the
same lines on the generator's problems (`cuadmm_gen_problems <dir>` writes them).

## The defaults of 2.0

    cuadmm_exe <problem> --tol 1e-6 --summary summary.json --history history.csv --save-solution solution/

Plain ADMM with the residual-balancing penalty rule (`penalty.policy = residual_balancing`, sigma0 100), the FP16
composite warm-up of the PSD blocks of at least 5000 (`psd.large.warmup = composite_fp16`,
`psd.large.warmup.min_size = 5000`), LOBPCG for the large blocks of low rank (`psd.lowrank.enabled`), the SYRK /
scaled products of the projections, CHOLMOD on the host for the normal equations (the dense factor on the GPU when
it is dense: `linsys.backend = auto`), the arena-sized PSD workspaces (the plan printed at `--verbosity 1`).

## The arithmetic of main, bit for bit

    cuadmm_exe <problem> --psd-warmup none --psd-final-product gemm_compat --psd-batched-final-product gemm_compat \
        --policy residual_balancing --sigma0 100

The reference builds of 1.x reproduce these runs bit for bit on the 53 baseline cases (the GPU evidence:
`c4eb89b:cuADMM-reorg/docs/PROGRESS.md`, at that commit only).
`--policy legacy_adaptive --sigma0 1` is the rule of the published PLATO runs (9bb4ba5, with Monitor1);
`--policy fixed --sigma0 <x>` keeps sigma.

## The warm-up on smaller blocks

    cuadmm_exe <problem> --psd-warmup composite_fp16 --psd-warmup-min-size 2000 --psd-warmup-switch-tol 1e-2

The projector benchmark (`cuadmm_psd_projector_bench`; its recorded run:
`c4eb89b:cuADMM-reorg/docs/phase6a_projector_benchmark.md`) measured
the filter 6x faster than the exact EVD at n = 2000; the warm-up ends at the first iteration with the KKT residual below `switch_tol` (or after
`switch_iter` iterations), then the exact projections and the low-rank analysis take over. The summary reports
`warmup_switch_iteration` and `composite_fallbacks` (the results the norm test rejected, projected exactly).

## The normal-equation backends

    cuadmm_exe <problem> --backend cudss          # cuDSS on the device (when the build has it: CUADMM_WITH_CUDSS)
    cuadmm_exe <problem> --backend dense          # cuSOLVER Cholesky on the device (8 m^2 bytes)
    cuadmm_exe <problem> --backend cholmod        # CHOLMOD on the host, the dense factor on the GPU by today's rule

`cuadmm_linsys_bench` measures the factorization time and the time per solve of every backend on problem
directories; `linsys.presolve` removes dependent constraint rows before the factorization:

    cuadmm_exe <problem> --presolve 1 --verbosity 1

## The hybrid scheme and the external validation

    cuadmm_exe <problem> --scheme sgs_admm --tau 1.95                 # sGS-ADMM throughout
    cuadmm_exe <problem> --sgs-iterations 5000 --sgs-tau 1.95         # 5000 sGS-ADMM iterations, then plain ADMM
    cuadmm_exe <problem> --validation-interval 100 --tol 1e-6          # stop on the residuals recomputed on the CPU

## The penalty rule of Phase 4 (not the default: DECISIONS 37)

    cuadmm_exe <problem> --policy balance_freeze --factor 1.2 --step-schedule halve_on_reversal --freeze-bias 4

The CPU screen of Phase 4b chose this setting; the H200 gates kept `residual_balancing` as the default
(DECISIONS 37).

## The C++ API

```cpp
#include <cuadmm/solver.h>
cuadmm::SolverOptions options;                         // the defaults above; cuadmm::set_option(options, "psd.large.warmup", "none")
cuadmm::ProblemData problem = cuadmm::io::read_problem(path, cuadmm::IoOptions()); // any supported format
cuadmm::Solver solver(options);
solver.setup(problem);                                 // scale, factorize, plan the PSD workspaces, check the memory
cuadmm::SolveResult result = solver.solve();           // result.best.X / y / S, result.pobj, result.converged, ...
```

## MATLAB

    [X, y, S, info] = cuadmm.solve(At, b, C, K, opts)   % the +cuadmm package of matlab/; opts: the option paths

The gateway is built with `-DCUADMM_BUILD_MATLAB=ON` (`matlab/`); it has not been built against MATLAB on the
development machines yet. See `docs/MIGRATION.md` for the signature of the 1.x MEX function and the options that
changed.

## Problem data

- `SPOT/data/`: the SPOT moment and SOS relaxations (PlanarHand, PushBox, PushT) as MOSEK `.mat` files (`MOSEK/`) and
  TXT directories (`TXT/`); the format tests read them (`CUADMM_SPOT_DATA_DIR`).
- `pendulum/`: the pendulum swing-up relaxation with N = 80 (`TXT/N=80_licols/`, `MATLAB/N=80_licols.mat`); the loose
  `At.txt`, `b.txt`, `C.txt`, `blk.txt`, `con_num.txt` are a second variant of the same size.
- `plato/`: PLATO / Kocvara instances as SDPA `.dat-s` and `.mat` files (`MATLAB/`) and TXT directories (`TXT/`).
- `dimacs/data/`: the DIMACS instances hinf12, hinf13, truss5 and truss8 (`MATLAB/`, `TXT/`).
- `max-cut/`: MATLAB scripts that generate max-cut SDPs in SeDuMi form (Richard Y. Zhang's `dual_ctc`, BSD 2-Clause:
  `max-cut/LICENCE` and `max-cut/README.md`).

A TXT directory holds `At.txt`, `b.txt`, `C.txt`, `blk.txt` and `con_num.txt`; `cuadmm-convert` writes it from any
format the build reads (the MATLAB conversion scripts of 1.x are gone).
