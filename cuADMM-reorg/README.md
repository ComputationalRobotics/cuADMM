# cuADMM-reorg

The reorganized cuADMM solver (plan: `../reorg-plan/cuadmm_reorg_plan.tex`, revision 2; instructions:
`../reorg-plan/developer_prompt.md`), built as a self-contained CMake project. The original tree is untouched and
serves as the reference implementation.

- `docs/PROGRESS.md`: status per phase, acceptance commands and their output, pending checks.
- `docs/DECISIONS.md`: choices, deviations from the plan and open questions.
- `docs/CHANGES.md`: every behavior change relative to `main`.
- `tools/compare_runs.py`: bitwise comparison of runs (reference builds vs this project).
- `tools/reference/`: patches applied to the reference worktrees and the pinned configuration.
- `tools/slurm/`: GPU job scripts (baselines, tests).

## Build

    tools/build.sh <build dir>            # CPU node: configure (RelWithDebInfo, sm_80) and build
    # or directly:
    cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80-real -DSUITESPARSE_INCLUDE_DIRECTORIES=... \
          -DCHOLMOD_LIB=... -DCMAD_LIB=... -DCCOLAMD_LIB=... -DBLAS_LIBRARIES=...
    cmake --build build -j16

Options: `CUADMM_BUILD_CLI`, `CUADMM_BUILD_TESTS`, `CUADMM_BUILD_BENCH`, `CUADMM_BUILD_MATLAB`, and the input formats
`CUADMM_WITH_HDF5`, `CUADMM_WITH_MATIO`, `CUADMM_WITH_ZLIB` (`AUTO`: on when the library is found; `ON`; `OFF`), located with
`-DHDF5_ROOT=`, `-DMATIO_ROOT=`, `-DZLIB_ROOT=` (`cmake/FindHDF5Lib.cmake`, `cmake/FindMatio.cmake`).

## Run

    build/cuadmm_exe <problem> [--option value ...]             # the CLI on the option table (docs/OPTIONS.md);
                                                               # today's flags are accepted as aliases
    # <problem>: a TXT directory, a native .h5 file, SDPA .dat-s[.gz], or a SeDuMi / SDPT3 / MOSEK prob .mat file
    # (--formats lists the formats compiled in; --format, --sense, --simplify, --save-format: plan Section 12)
    build/cuadmm-convert <input> <output.h5 | directory>       # any input to the native HDF5 file or a TXT directory
    build/cuadmm_io_bench generate|read ...                    # synthetic problems and read rates of the formats
    build/cuadmm_exe --print-options [markdown]                # the option table
    build/cuadmm_tests                                         # GPU tests; run from the build directory
    build/cuadmm_cpu_tests                                     # CPU-only tests (options, policies, readers)
    build/cuadmm_gen_problems <dir>                            # writes the known-optimum generator problems as TXT

Penalty policies (`--policy`): `fixed`, `legacy_adaptive` (the historical rule of 9bb4ba5), `residual_balancing`
(the October 13 rule of 2d8a755, the default until plan Phase 4b), `balance_freeze` (Phase 4). Schemes
(`--scheme`): `admm` (default), `sgs_admm`.
