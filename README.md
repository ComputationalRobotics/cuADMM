# cuADMM
A CUDA-based implementation of the Alternating Direction Method of Multipliers (ADMM) algorithm to solve Semi-Definite Programming (SDP) problems.

cuADMM solves multi-block SDP problems of the form:
```math
\min_X \left\langle C,X\right\rangle \quad\text{s.t.}\quad \begin{cases}
        \left\langle A_i,X\right\rangle = b_i, \quad i\in [m]\\
        X\in\Omega_+
    \end{cases}
```
where $\Omega_+$ is the cartesian product of symmetric cones corresponding to the symmetric blocks.

## Dependencies
The following dependencies are required to build and run the project:
- [`CMake`](https://cmake.org/download/)
- [`CUDA`](https://developer.nvidia.com/cuda-downloads)
- [`BLAS`](https://www.netlib.org/blas/) (for basic linear algebra operations on CPU)
- [`SuiteSparse`](https://github.com/DrTimothyAldenDavis/SuiteSparse) (for Cholesky factorization)
- [`MATLAB`](https://www.mathworks.com/products/matlab.html) (for the bindings)

This project has been tested on Linux.

## Execution
Build the project using CMake:
```bash
mkdir build
cmake -S . -B build
cmake --build build
```

Run the project:
```bash
./build/cuadmm_exe [dir_name] [options]
```
where `dir_name` is the directory containing the input files. See below for the expected input format.

Without options, the executable runs plain (two-block) ADMM from the first iteration with a KKT tolerance of `1e-4`. The main options are (`./build/cuadmm_exe --help` lists all of them):
| Option | Description |
|--------|-------------|
| `--tol <x>`, `--max-iter <n>` | KKT stopping tolerance (default `1e-4`) and maximum number of iterations |
| `--switch-admm <n>` | run sGS-ADMM (the update of Algorithm 1 of [the paper](https://arxiv.org/abs/2406.05846)) for the first `n-1` iterations, then plain ADMM; `0` (default) is plain ADMM throughout, `n` larger than `--max-iter` is pure sGS-ADMM (= Algorithm 1). With `1 < n <= --max-iter` it is the hybrid mode (see `--sgs-iterations`). In the sGS phase τ = 1.95, reduced to max(1.618, τ/1.1) = 1.7727 at iterations whose incoming dual residual is below `--tol`; τ = 1.618 in plain ADMM |
| `--sig <x>`, `--lobpcg <0\|1>` | initial penalty `sigma` (default `100`) and LOBPCG for low-rank blocks larger than 1000 (default on) |
| `--sigscale <x>`, `--sig-update-*` | factor (default `2`) and periods of the `sigma` updates of the sGS phase, with `--sigma-policy legacy_adaptive` only; the plain ADMM phase has its own schedule (factor 2). Use `--sigma-policy fixed` for a fixed σ |
| `--algorithm <admm\|sgs>` | plain ADMM throughout, or **pure sGS-ADMM** throughout (the switch to plain ADMM cannot happen); an alternative to `--switch-admm` |
| `--sgs-iterations <n>` | **hybrid mode** of this code (not Algorithm 1 of the paper, which is pure sGS-ADMM, and not its kNN warm start): exactly `n` sGS-ADMM iterations, then plain ADMM continued from the same `X`, `y`, `S`, σ and workspaces in the same solve (`--switch-admm n+1`; `0` is plain ADMM). The log and the JSON summary report the phases (iterations, times, the switch iteration, σ and the KKT residual at the switch), and `--history` marks the phase and τ of every iteration |
| `--sigma-policy <fixed\|legacy_adaptive\|fixed_sgs_then_legacy_admm>` | `fixed`: σ = `--sig` for every iteration (Algorithm 1 of the paper); `legacy_adaptive` (default): the historical adaptive σ rules below; `fixed_sgs_then_legacy_admm`: σ fixed during the sGS phase of a hybrid run, then the legacy plain-ADMM rules exactly as a plain-ADMM solve started at the switch applies them (schedule counted from the first plain-ADMM iteration, primal/dual win counters from 0) |
| `--validate-interval <n>`, `--validate-tol <x>`, `--validate-threads <n>` | validation-aware stopping: every `n` iterations and when the internal KKT residual crosses `--tol`, recompute on the CPU from the original data the primal/dual residuals, the gap, the per-block X cone violation max(0, −λmin(X_b))/(1+‖X_b‖_F) and the dual cone violation, and stop only when all are ≤ the tolerance (default `--tol`); without success the best externally evaluated iterate is returned |
| `--checkpoint-iters <list>`, `--checkpoint-dir <dir>`, `--validation-history <file>`, `--time-limit <s>` | extra validation snapshots (saved to `<dir>/iter_<k>`), one CSV line per validation, wall-time cap of the iterations |
| `--warm-start-dir <dir>` | start from `X.txt`, `y.txt`, `S.txt` in `<dir>` |
| `--save-solution <dir>` | write the returned `X.txt`, `y.txt`, `S.txt` and `certificate.txt` to `<dir>` |
| `--summary <file>`, `--history <file>`, `--tag <label>` | append a one-line JSON summary of the run, write the per-iteration history as CSV |
| `--trace-bound <R>`, `--trace-bound-scale <f>` | print the certified lower bound $\langle b,y\rangle + \sum_\beta R_\beta \min(0, \lambda_{\min}((C - A^\top y)_\beta))$ of the paper (eq. 30), with $R_\beta = R$ or $R_\beta = f\cdot n_\beta$ |

The summary reports, besides the KKT residuals, the cone violation of the returned `X` (`X` is not a projection, and with `--switch-admm` it can be noticeably outside the cone while the KKT residuals are below the tolerance), and the quantities of the certificate. The certified lower bound is valid when every PSD block of an optimal `X` has trace at most $R_\beta$ (and the entries of `l` / `u` blocks lie in $[0, R_\beta]$ / $[-R_\beta, R_\beta]$). For the moment matrix of order $\kappa$ of a probability measure on the box $|x_i| \le R$ ($n$ variables), $\mathrm{tr}(M) \le \sum_{j \le \kappa} \binom{n+j-1}{j} R^{2j} \le s(n,\kappa) \max(1,R)^{2\kappa}$, where $s(n,\kappa)$ is the size of the block; for a localizing matrix of $g$, multiply by $\max |g|$ over the box. So for POP variables scaled to $[-1,1]$, $R_\beta$ is the size of the block (`--trace-bound-scale 1`, the bound of Theorem 2 of the paper) for moment matrices, and for localizing matrices if $|g| \le 1$ there. The paper's $s(n,\kappa) R^2$ holds only for $R = 1$: `--trace-bound-scale` with $f = R^2$ is not a valid bound for $R > 1$.

At initialization the solver prints how eps*I + AA^T was factorized (supernodal or simplicial LDL^T, nnz(L)) and its pivots. Tiny pivots mean that A has dependent rows, as in the SPOT examples. The supernodal factorization then fails and the slower simplicial one is used. Removing the dependent rows (as in the pendulum `licols` data) avoids this. The y-step solves run on the CPU, except when the factor is dense (few constraints, e.g. `neosfrbr25`); then they run on the GPU. Medium blocks of equal size are eigendecomposed in one batched call. Runs are bitwise reproducible. Measurements: `experiments/2026-09-25_sgs_admm/code_benchmarks/`; the sGS-ADMM vs ADMM comparison on the shipped problems: `experiments/2026-09-25_sgs_admm/README.md`.

## Input format
cuADMM can be called in two ways: in the command line by specifying a directory containing `TXT` input files, or by using the MATLAB bindings.

### From `TXT`
When using the executable, you need to provide a directory containing the input files, in a format close to SDPT3. The expected files are:
- `At.txt`: the transpose of the constraint matrix in sparse `svec` COO format
- `b.txt`: the right-hand side constraint vector in sparse COO format
- `blk.txt`: a file containing the size of the blocks; if a line contains both a character and a number (e.g. `s 3`), the block is interpreted using the table below; if it contains only a number (e.g. `2`), it indicates a PSD block (identical to `s 2`)
- `C.txt`: the cost matrix in sparse `svec` COO format
- `con_num.txt`: a file containing the number of constraints (which cannot be inferred from the other files)

Additionally, an initial guess can be provided with `--warm-start-dir <dir>` (or with `--warm-start`, from the problem directory itself) as the following files, which contain dense vectors (one value per entry, in `svec` form for `X` and `S`), as written by `--save-solution`:
- `X.txt`: the primal variable
- `y.txt`: the dual variable of the equality constraints
- `S.txt`: the dual slack variable

`X`, `C` and `A` use the `svec` version of the multi-block matrices, obtained by stacking the upper triangular part of each block in a vector, where non-diagonal elements are multiplied by $\sqrt{2}$. The sparse COO format stores the non-zero elements of the `svec` vector by storing the row indices, column indices, and values on the same line, separated by spaces.

Examples files are provided in the `examples` directory, in the `TXT` subfolders. See for instance [this example](examples/SPOT/data/TXT/PlanarHand_N=1_MOMENT).

### Block types
The `blk.txt` file can contain the following block types:
| Character | Description                                           |
|-----------|-------------------------------------------------------|
| `s`       | PSD matrix of size `n` by `n`                         |
| `u`       | Unconstrained vector of size `n` (free variables)     |
| `l`       | Non-negative vector of size `n` (linear cone)         |

### From other formats
We provide in `examples` a few MATLAB scripts to convert from other formats to the expected `TXT` format:
- `mosek_to_txt.m`: converts a problem in MOSEK format to the `TXT` format
- `sedumi_to_txt.m`: converts a problem in SeDuMi format to the `TXT` format

## MATLAB Bindings
In the `MATLAB` directory, you can find the bindings to use cuADMM from MATLAB. To use them, you need to compile the MEX files:
```bash
cd MATLAB
mkdir build
cmake -S . -B build
cmake --build build
```
The signature of the MEX function is the following:
```matlab
cuadmm_MATLAB(eig_stream_num_per_gpu,...
              max_iter, stop_tol,...
              At_stack, b, C_stack, blk_types, blk_vec,...
              X_new, y_new, S_new, sig_new,...
              sig_update_threshold, sig_update_stage_1, sig_update_stage_2,...
              switch_admm,switch_proj_iter,switch_proj_tol,...
              sigscale);
```
The file [`cuadmm_MATLAB.cu`](MATLAB/cuadmm_MATLAB.cu) defines the MEX function, and can be used as a reference for the input format when interfacing with other languages or libraries.

A few examples of how to use the bindings are also provided, such as [`example_mosek.m`](MATLAB/example_mosek.m) which shows how to solve a problem in MOSEK format.

## Testing
After building, you can execute the unit tests:
```bash
cd build && ctest
```

> [!NOTE]
> Some tests require the `CUADMM_SOLVER_TEST_PATH` environment variable to be set to the path of some test data. You can  export it in your terminal session using `export CUADMM_SOLVER_TEST_PATH="/path/to/test/data"`. If the environment variable is not set, the tests will be skipped.
