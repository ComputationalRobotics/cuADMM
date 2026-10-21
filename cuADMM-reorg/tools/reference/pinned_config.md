# Pinned configuration for the bitwise comparisons

Bitwise results hold for one configuration only (plan Section 17). Every reference build and every cuADMM-reorg
build compared with it uses this configuration; `tools/slurm/phase0_baselines.sh` records the live values of each
GPU job in `$CUADMM_BASELINES/<config>/config.txt`.

| Item | Value |
|---|---|
| Config name | `a100-sxm4-80gb_cuda12.9_gcc12.2_ss7.10.1` |
| Cluster, partition | FASRC Cannon, partition `gpu` (`--gres=gpu:nvidia_a100-sxm4-80gb:1 -c 16 --mem=96G`, allocations under one hour) |
| GPU | NVIDIA A100-SXM4-80GB, one device (`gpu.device` 0), driver as recorded per job |
| CUDA | 12.9.86 (module `cuda/12.9.1-fasrc01`), code generation `-gencode arch=compute_80,code=sm_80` (CMake `80-real`) |
| Host compiler | GCC 12.2.0 (module `gcc/12.2.0-fasrc01`), C++17, build type RelWithDebInfo (`-O2 -g -DNDEBUG`) |
| CMake | 3.31.6 |
| SuiteSparse | 7.10.1, CHOLMOD 5.3.1 (conda env `$HOME/cuadmm-env`; `CHOLMOD_AUTO` supernodal with simplicial LDL^T fallback) |
| BLAS / LAPACK | OpenBLAS 0.3.34 (`libblas.so` of the same env; LAPACK `dsyevd` from OpenBLAS) |
| Threads | `OMP_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=16` (16 cores per job) |
| cuSPARSE SpMV | `CUSPARSE_SPMV_CSR_ALG2` (deterministic; in the sources since 3d3b274) |
| Presolve | none (the references have none; `linsys.presolve` stays off in the bitwise phases) |
| Normal equations | CHOLMOD on the CPU (dense-factor mode only under today's conditions, none of the test problems trigger it) |
| Build machines | CPU nodes (`sapphire`/`shared`, 16 cores) for compiling; the GPU node only runs |

## Reference builds

Worktrees under `$HOME/cuadmm-worktrees/`, built into `$HOME/cuadmm-builds/<name>/` with
`$HOME/cuadmm-tools/build_ref.sh` (same CMake arguments as `tools/build.sh`; the references set
`CUDA_ARCHITECTURES OFF`, so the architecture is passed through `CMAKE_CUDA_FLAGS`).

| Name | Commit | Submodule | Patches applied (reference worktrees only) | Used for |
|---|---|---|---|---|
| `ref-9bb4ba5` | 9bb4ba5 | b566590 | `print17.patch` | `fixed`, `legacy_adaptive` (Monitor1 included) |
| `ref-2d8a755` | 2d8a755 | b566590 | `print17.patch` | `fixed`, `residual_balancing` |
| `ref-2d8a755-sgs` | 2d8a755 | b566590 | `print17.patch`, `sgs_no_plain_only.patch` | sGS-ADMM (`--algorithm sgs`), fixed sigma, tau 1.95 |

`print17.patch` changes printing only: the three `std::setprecision(10)` of `src/main.cu` (history CSV, JSON summary,
`json_number`) become 17 significant digits, so that histories round-trip exactly. `--save-solution` already writes 17
digits. `sgs_no_plain_only.patch` comments out `target_compile_definitions(... CUADMM_PLAIN_ADMM_ONLY)` in the root
`CMakeLists.txt`, which is the "build without `CUADMM_PLAIN_ADMM_ONLY`" of the plan's Phase 4 criterion.

Both patches were applied with `sed` on 2026-10-21 and recorded with `git diff` from the worktrees
(`git -C $HOME/cuadmm-worktrees/ref-9bb4ba5 diff` reproduces `print17.patch`).

## Data

- Generator problems: `cuadmm_gen_problems $CUADMM_DATA/generated` (deterministic, `std::mt19937` raw outputs); the
  SHA-256 of the written files is recorded next to each baseline (`generated_sha256.txt`).
- PLATO: `trto1` and `mater-1` TXT directories converted by
  `experiments/2026-09-29_plato_cold_plain_admm_sigma/scripts/sdpa_to_cuadmm.py` (`$CUADMM_DATA/plato_kocvara/txt/`).
