# MATLAB MEX gateway of cuADMM-reorg (optional: -DCUADMM_BUILD_MATLAB=ON, needs MATLAB).
# Phase 0: the gateway source is the verbatim copy of MATLAB/cuadmm_MATLAB.cu; it links the cuadmm library of this
# project instead of recompiling the sources (as the original MATLAB/CMakeLists.txt did).
# Phase 2 rebuilds the gateway on SolverOptions (cuadmm.solve(At, b, C, K, opts) with a positional shim).
find_package(Matlab REQUIRED)

matlab_add_mex(
  NAME cuadmm_MATLAB
  SRC cuadmm_MATLAB.cu
  OUTPUT_NAME cuadmm_MATLAB
  LINK_TO cuadmm ${Matlab_LIBRARIES})
# The external validator needs LP64 LAPACK (dsyevd); MATLAB ships an ILP64 LAPACK, so the gateway (which never enables
# validation) is compiled without it, as today.
target_compile_definitions(cuadmm_MATLAB PRIVATE CUADMM_NO_LAPACK)
