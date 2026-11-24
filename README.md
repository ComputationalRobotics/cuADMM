# cuADMM 2.1

cuADMM is a GPU ADMM solver for SDP (semidefinite programs) with PSD, nonnegative and free blocks, the solver of
[arXiv:2406.05846](https://arxiv.org/abs/2406.05846). It solves

    min <C, X>   s.t.   <A_i, X> = b_i  (i = 1..m),   X in K

with K a product of PSD cones, nonnegative orthants and free blocks. It is one CMake project with the `cuadmm`
library, the command-line interface, the MATLAB gateway, the tests and the benchmarks; the ported engine of 1.x
lives in `src/legacy/` behind the facade `cuadmm::Solver`.

- `docs/OPTIONS.md`: every option (one table for the CLI, the C++ API and MATLAB).
- `docs/MIGRATION.md`: what changed for users of 1.x and of 2.0 (entry points, option names, defaults).
- `docs/CHANGES.md`: the changes of every version.
- `docs/DECISIONS.md`: the design decisions of the 2.0 reorganization.
- `examples/`: command lines and scripts on the options, and the problem data (SPOT, PLATO, pendulum, DIMACS,
  max-cut).

## Build

    cmake --preset release && cmake --build --preset release                 # Release, sm_80 (build-release/)
    cmake --preset relwithdebinfo && cmake --build --preset relwithdebinfo   # the development configuration (build/)
    tools/build.sh <build dir> [cmake args]                                  # configure once and build (RelWithDebInfo)

Requirements: CUDA 12.x (cuBLAS, cuSOLVER, cuSPARSE; cuDSS 1.0 optional), SuiteSparse (CHOLMOD, CAMD, CCOLAMD),
a BLAS/LAPACK, CMake 3.21, a C++17 compiler. CMake options: `CUADMM_BUILD_CLI`, `CUADMM_BUILD_TESTS`,
`CUADMM_BUILD_BENCH`, `CUADMM_BUILD_MATLAB` (needs MATLAB), the formats `CUADMM_WITH_HDF5`, `CUADMM_WITH_MATIO`,
`CUADMM_WITH_ZLIB` and the backend `CUADMM_WITH_CUDSS` (`AUTO` / `ON` / `OFF`, located with `-DHDF5_ROOT=`,
`-DMATIO_ROOT=`, `-DZLIB_ROOT=`, `-DCUDSS_ROOT=`), `CMAKE_CUDA_ARCHITECTURES` (`80-real` by default; `90-real` for
H100/H200). `cmake --build <dir> --target cuadmm_docs` renders the API documentation with Doxygen (docs/api/html).

## Run

    cuadmm_exe <problem> [--option value ...]      # <problem>: a TXT directory, a native .h5 file, SDPA .dat-s[.gz],
                                                   # or a SeDuMi / SDPT3 / MOSEK prob .mat file (--formats lists the readers)
    cuadmm_exe --print-options [markdown]          # the option table
    cuadmm-convert <input> <output.h5 | directory> # any input to the native HDF5 file or a TXT directory
    cuadmm_tests, cuadmm_cpu_tests                 # the GPU suite (run from the build directory) and the CPU-only suite
    cuadmm_linsys_tests, cuadmm_presolve_tests     # the backend and presolve suites
    cuadmm_io_bench, cuadmm_psd_projector_bench, cuadmm_linsys_bench   # the benchmarks (CUADMM_BUILD_BENCH)
    cuadmm_gen_problems <dir>                      # the known-optimum generator problems as TXT

For example, from the repository root after `cmake --preset release && cmake --build --preset release`:

    build-release/cuadmm_exe examples/SPOT/data/TXT/PlanarHand_N=1_MOMENT --summary summary.json --history history.csv

Outputs: `--summary` (JSON: the status, residuals, objectives, timings, the memory plan and peak, the backend and
factorization statistics, the options), `--history` (CSV per iteration), `--sigma-log`, `--save-dir` (X, y, S in
original units); per-iteration values are the engine's minimization values without the objective offset, the
final objectives are in the file's sense with the offset.

In MATLAB, `[X, y, S, info] = cuadmm.solve(At, b, C, K, opts)` (the `+cuadmm` package of `matlab/`, built with
`-DCUADMM_BUILD_MATLAB=ON`; the gateway has not been built against MATLAB on the development machines yet).

## What the solver does

- **Algorithm**: plain two-block ADMM with the step length `algorithm.tau` (1.618).
- **Penalty** (`penalty.policy`): `balance_freeze` (default) balances the primal and dual residuals: after every
  iteration the larger one gets a vote, and at the checks (every 10 iterations up to 200, every 25 up to 1000, every
  50 up to 5000, every 200 up to 10000, every 1000 after that) sigma is multiplied or divided by `penalty.factor`
  (1.35) when one side leads by the ratio 1.35, within [`penalty.sigma_min`, `penalty.sigma_max`] = [1e-3, 1e3],
  starting from `penalty.sigma0` (100); from `penalty.freeze_iteration` (10000) on, sigma keeps its value.
  `fixed` keeps sigma at `penalty.sigma0`.
- **PSD projections**: the small blocks (n <= 32) with the batched Jacobi EVD, the medium blocks (n <= 1000) with
  the batched or streamed cuSOLVER EVD, the large blocks one at a time through one workspace arena with the exact
  EVD, the warm-started LOBPCG of the blocks of low rank (`psd.lowrank.*`), and the FP16 composite filter as the
  warm-up of the blocks of at least `psd.large.warmup.min_size` (5000) until the KKT residual falls below
  `psd.large.warmup.switch_tol`. The final products use SYRK on the smaller eigen-side. `Solver::setup` plans the
  PSD workspaces, prints the plan at verbosity 1 and fails early above `psd.large.memory_budget` of the free memory.
- **Normal equations** (`linsys.backend`): CHOLMOD on the host with the dense factor on the GPU when it is dense
  (`auto`), the cuSOLVER dense Cholesky (`dense`), cuDSS (`cudss`); `linsys.presolve` removes dependent constraint
  rows.
- **Stopping**: the internal KKT residual max(eta_p, eta_d, eta_g) below `algorithm.tol` (1e-4); with
  `validation.interval` the residuals are recomputed in double precision in the original units and the solve stops only on
  them (external validation, the practical and strict DIMACS tolerances); `algorithm.time_limit` and
  `algorithm.max_iter` return the best iterate. The summary carries the cone violations of X and S and, with
  `certificate.block_bounds` (or the trace bound), a certified lower bound on the optimal value.
- **Determinism**: every run is reproducible bit for bit on the same GPU model and build (`seed` fixes the Lanczos
  start of the composite warm-up; the medium-tier streams and the batched EVD chunks do not change the bits).

## Limits

- PSD blocks are limited to n <= 32768 by the cuSOLVER EVD of CUDA 12.9 (checked at setup); the large tier holds one
  block of the largest size (M, about 5 n^2 doubles with the workspace) plus the LOBPCG warm starts.
- Second-order, rotated, exponential and power cones are not supported: they are rejected at setup with a message.

## Repository layout

    include/cuadmm/   the public API (options, problem data, solver facade, result, io)
    src/core          GpuContext, StreamPool, WorkspaceArena, buffers, checks
    src/cone          the cone layer: product, free, nonnegative, PSD (tiers, projectors, policy, kernels)
    src/linsys        NormalSystem and the backends (CHOLMOD, dense cuSOLVER, cuDSS); src/problem/presolve
    src/admm          AdmmState, AdmmSolver, the penalty policies; src/termination  the stopping rules, the certificate
    src/options, src/report, src/io, src/solver   the option table, the writers, the readers, the facade
    src/legacy        the ported engine of 1.x (its vectors, scaling, y-step data, histories)
    apps/cli          cuadmm_exe, cuadmm-convert, cuadmm_gen_problems;  matlab/  the MEX gateway and +cuadmm/solve.m
    test/, bench/     the suites (test/data: fixtures and golden data) and the benchmarks
    tools/            build.sh, the SLURM environment and test scripts, the generators of the I/O test files
    examples/         the problem data and the example scripts;  docs/  the documentation;  cmake/  the find modules

## Citation

The method and the solver are described in [arXiv:2406.05846](https://arxiv.org/abs/2406.05846).
