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

Options: `CUADMM_BUILD_CLI`, `CUADMM_BUILD_TESTS`, `CUADMM_BUILD_MATLAB`, `CUADMM_WITH_HDF5`, `CUADMM_WITH_MATIO`.

## Run

    build/cuadmm_exe <problem directory> [options]     # Phase 0: the copied CLI (same flags as main)
    build/cuadmm_tests                                 # GPU; run from the build directory
    build/cuadmm_gen_problems <dir>                    # writes the known-optimum generator problems as TXT
