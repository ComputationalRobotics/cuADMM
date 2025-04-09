# cuADMM
A CUDA-based implementation of the Alternating Direction Method of Multipliers (ADMM) algorithm to solve Semi-Definite Programming (SDP) problems.

## Execution
Build the project using CMake:
```bash
cmake -S . -B build
cmake --build build
```

Run the project:
```bash
./build/cuadmm_exe
```

## Testing
After building, you can execute the unit tests:
```bash
cd build && ctest
```

> [!NOTE]
> Some tests require the `CUADMM_SOLVER_TEST_PATH` environment variable to be set to the path of some test data. You can  export it in your terminal session using `export CUADMM_SOLVER_TEST_PATH="/path/to/test/data"`. If the environment variable is not set, the tests will be skipped.