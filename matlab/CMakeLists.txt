# MATLAB MEX gateway of cuADMM 2.0 (optional: -DCUADMM_BUILD_MATLAB=ON, needs MATLAB): cuadmm_mex(At, b, C, K, opts),
# called through the package function +cuadmm/solve.m, which also serves the positional signature of the 1.x gateway
# cuadmm_MATLAB for one release. Add this directory to the MATLAB path (the MEX file next to the +cuadmm package).
# The gateway links the cuadmm library of this project. It has not been built against MATLAB on the development
# machines yet (no MATLAB there); cuadmm_mex.cu compiles against stub declarations of the MEX API.
find_package(Matlab REQUIRED)

matlab_add_mex(
  NAME cuadmm_mex
  SRC cuadmm_mex.cu
  OUTPUT_NAME cuadmm_mex
  LINK_TO cuadmm ${Matlab_LIBRARIES})
target_include_directories(cuadmm_mex PRIVATE ${CMAKE_BINARY_DIR}/generated)
add_dependencies(cuadmm_mex cuadmm_version)
# The external validator needs LP64 LAPACK (dsyevd); MATLAB ships an ILP64 LAPACK, so the gateway (which never enables
# validation) is compiled without it.
target_compile_definitions(cuadmm_mex PRIVATE CUADMM_NO_LAPACK)
