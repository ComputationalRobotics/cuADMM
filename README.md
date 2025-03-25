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

Execute the tests:
```bash
cd build && ctest
```