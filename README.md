# cuADMM 2.0 (cuADMM-reorg)

The reorganized cuADMM solver: a GPU ADMM solver for semidefinite programs with PSD, nonnegative and free blocks,
built as a self-contained CMake project (`cuadmm` library, CLI, MATLAB gateway, tests, benchmarks). It reproduces
the reference solver of `main` bit for bit on its exact paths and adds the options of the reorganization plan
(`../reorg-plan/cuadmm_reorg_plan.tex`). The original tree stays the reference implementation; switching it over
is the maintainers' decision.

- `docs/OPTIONS.md`: every option (one table for the CLI, the C++ API and MATLAB).
- `docs/MIGRATION.md`: what changed for users of main (the MEX signature, the defaults, the option names).
- `docs/CHANGES.md`: every behavior change relative to main, with the option that restores the old behavior.
- `docs/DECISIONS.md`, `docs/PROGRESS.md`: the decisions and the acceptance evidence, phase by phase.
- `examples/`: command lines and scripts on the new options.

## Build

    cmake --preset release && cmake --build --preset release        # Release, sm_80 (build-release/)
    cmake --preset relwithdebinfo && cmake --build --preset relwithdebinfo   # the development configuration (build/)
    tools/build.sh <build dir> [cmake args]                          # the script of the acceptance runs (RelWithDebInfo)

Requirements: CUDA 12.x (cuBLAS, cuSOLVER, cuSPARSE; cuDSS 1.0 optional), SuiteSparse (CHOLMOD, CAMD, CCOLAMD),
a BLAS/LAPACK, CMake 3.21, a C++17 compiler. CMake options: `CUADMM_BUILD_CLI`, `CUADMM_BUILD_TESTS`,
`CUADMM_BUILD_BENCH`, `CUADMM_BUILD_MATLAB` (needs MATLAB), the formats `CUADMM_WITH_HDF5`, `CUADMM_WITH_MATIO`,
`CUADMM_WITH_ZLIB` and the backend `CUADMM_WITH_CUDSS` (`AUTO` / `ON` / `OFF`, located with `-DHDF5_ROOT=`,
`-DMATIO_ROOT=`, `-DZLIB_ROOT=`, `-DCUDSS_ROOT=`), `CMAKE_CUDA_ARCHITECTURES` (`80-real` by default; `90-real` for
H100/H200). `cmake --build <dir> --target cuadmm_docs` renders the API documentation with Doxygen (docs/api/html).

## Run

    cuadmm_exe <problem> [--option value ...]      # <problem>: a TXT directory, a native .h5 file, SDPA .dat-s[.gz],
                                                   # or a SeDuMi / SDPT3 / MOSEK prob .mat file (--formats lists the readers)
    cuadmm_exe --print-options [markdown]          # the option table; today's flags of main are accepted as aliases
    cuadmm-convert <input> <output.h5 | directory> # any input to the native HDF5 file or a TXT directory
    cuadmm_tests, cuadmm_cpu_tests                 # the GPU suite (run from the build directory) and the CPU-only suite
    cuadmm_linsys_tests, cuadmm_presolve_tests     # the backend and presolve suites
    cuadmm_io_bench, cuadmm_psd_projector_bench, cuadmm_linsys_bench   # the benchmarks (CUADMM_BUILD_BENCH)
    cuadmm_gen_problems <dir>                      # the known-optimum generator problems as TXT

Outputs: `--summary` (JSON: the status, residuals, objectives, timings, the memory plan and peak, the backend and
factorization statistics, the options), `--history` (CSV per iteration), `--sigma-log`, `--save-solution`
(X, y, S in original units); per-iteration values are the engine's minimization values without the objective
offset, the final objectives are in the file's sense with the offset.

## What the solver does

- **Scheme** (`algorithm.scheme`): plain two-block ADMM (default) or sGS-ADMM; the hybrid mode runs sGS-ADMM for
  the first iterations (`algorithm.sgs_iterations`) and plain ADMM afterwards from the same iterate.
- **Penalty** (`penalty.policy`): `residual_balancing` (default: the rule of main's 2d8a755), `legacy_adaptive`
  (the historical rule of 9bb4ba5 with Monitor1: the published PLATO runs), `fixed`, `balance_freeze` (the rule of
  plan Section 3.5, screened in Phase 4b; not the default: DECISIONS 37).
- **PSD projections**: the small blocks (n <= 32) with the batched Jacobi EVD, the medium blocks (n <= 1000) with
  the batched or streamed cuSOLVER EVD, the large blocks one at a time through one workspace arena with the exact
  EVD, the warm-started LOBPCG of the blocks of low rank (`psd.lowrank.*`), and the FP16 composite filter as the
  warm-up of the blocks of at least `psd.large.warmup.min_size` (5000; 6-8x faster than the EVD) until the KKT
  residual falls below `psd.large.warmup.switch_tol`. The final products use SYRK on the smaller eigen-side;
  `psd.*.final_product = gemm_compat` keeps main's products bit for bit. `Solver::setup` plans the PSD
  workspaces, prints the plan at verbosity 1 and fails early above `psd.large.memory_budget` of the free memory;
  the measured peak matches the plan within 0.2% on the problems measured.
- **Normal equations** (`linsys.backend`): CHOLMOD on the host with the dense factor on the GPU when it is dense
  (`auto`, bit for bit main), the cuSOLVER dense Cholesky (`dense`), cuDSS (`cudss`; `docs/phase7_linsys_benchmark.md`);
  `linsys.presolve` removes dependent constraint rows.
- **Stopping**: the internal KKT residual max(eta_p, eta_d, eta_g) below `algorithm.tol`; with
  `validation.interval` the residuals are recomputed on the CPU in the original units and the solve stops only on
  them (external validation, the practical and strict DIMACS tolerances); `algorithm.time_limit` and
  `algorithm.max_iter` return the best iterate. The summary carries the cone violations of X and S and, with
  `certificate.block_bounds` (or the trace bound), a certified lower bound on the optimal value.
- **Determinism**: every run is reproducible bit for bit on the same GPU and build (`seed` fixes the Lanczos start
  of the composite warm-up; the medium-tier streams and the batched EVD chunks do not change the bits).

## Guarantees and limits

- The exact paths (the EVD projections, LOBPCG, the CHOLMOD y-step, the residual-balancing and legacy penalty
  rules, the sGS and hybrid schemes) reproduce the reference builds bit for bit on 53 baseline cases of two pinned
  configurations with `--psd-warmup none` and the compat products (`docs/PROGRESS.md`). With the 2.0 defaults the
  runs differ within rounding; the composite warm-up changes the iterates of runs with blocks of at least 5000
  (deterministically).
- PSD blocks are limited to n <= 32768 by the cuSOLVER EVD of CUDA 12.9 (checked at setup); the large tier holds one
  block of the largest size (M, about 5 n^2 doubles with the workspace) plus the LOBPCG warm starts.
- Second-order, rotated, exponential and power cones (plan Phase 9) and the removal of the ported engine (Phase 8)
  are deferred: the cones are rejected at setup with a message. `linsys.autotune` is not implemented (DECISIONS 57).
- The penalty-rule redesign with a freeze (plan Section 3.5) remains open: the Phase 4b gates kept
  `residual_balancing` (DECISIONS 37, `docs/phase4b_gates_report.md`).

## Repository layout

    include/cuadmm/   the public API (options, problem data, solver facade, result, io)
    src/core          GpuContext, StreamPool, WorkspaceArena, buffers, checks
    src/cone          the cone layer: product, free, nonnegative, PSD (tiers, projectors, policy)
    src/linsys        NormalSystem and the backends (CHOLMOD, dense cuSOLVER, cuDSS); src/problem/presolve
    src/admm          AdmmState, AdmmSolver, the penalty policies; src/termination  the stopping rules, the certificate
    src/options, src/report, src/io, src/solver   the option table, the writers, the readers, the facade
    src/legacy        the ported engine of main (its vectors, scaling, y-step data, histories), shrinking phase by phase
    apps/cli, matlab/ the CLI and the MATLAB gateway (+cuadmm/solve.m); test/, bench/, tools/ the suites, benchmarks and scripts
