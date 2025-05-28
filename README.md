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
- [`LAPACK`](https://www.netlib.org/lapack/) (for linear algebra operations)
- [`BLAS`](https://www.netlib.org/blas/) (for basic linear algebra operations)
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
./build/cuadmm_exe [dir_name]
```
where `dir_name` is the directory containing the input files. See below for the expected input format.

## Input format
### From `TXT`
When using the executable, you need to provide a directory containing the input files, in a format close to SDPT3. The expected files are:
- `At.txt`: the transpose of the constraint matrix in sparse `svec` COO format
- `b.txt`: the right-hand side constraint vector in sparse COO format
- `blk.txt`: a file containing the size of the symmetric blocks
- `C.txt`: the cost matrix in sparse `svec` COO format
- `con_num.txt`: a file containing the number of constraints (which cannot be inferred from the other files)

Additionally, the following optional files can be provided:
- `X.txt`: an initial guess for the primal variable in sparse `svec` COO format
- `y.txt`: an initial guess for the dual variable in sparse `svec` COO format
- `S.txt`: an initial guess for the dual variable in sparse `svec` COO format

`X`, `C` and `A` use the `svec` version of the multi-block matrices, obtained by stacking the upper triangular part of each block in a vector, where non-diagonal elements are multiplied by $\sqrt{2}$. The sparse COO format stores the non-zero elements of the `svec` vector by storing the row indices, column indices, and values on the same line, separated by spaces.

Examples files are provided in the `examples` directory, in the `TXT` subfolders. See for instance [this example](examples/SPOT/data/TXT/PlanarHand_N=1_MOMENT).

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
The signature of the MEX function is the same as the C++ one, that is:
```matlab
cuadmm_MATLAB(eig_stream_num_per_gpu,...
              max_iter, stop_tol,...
              At_stack, b, C_stack, blk_vec,...
              X_new, y_new, S_new, sig_new);
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