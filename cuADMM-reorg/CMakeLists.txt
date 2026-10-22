cmake_minimum_required(VERSION 3.18)

# cuADMM-reorg: the reorganized cuADMM solver, built as a self-contained project (reorg-plan/developer_prompt.md).
# Nothing here includes or links the parent tree. Phase 0 (plan Section 17): the sources of main at c23e67e are
# copied verbatim into src/legacy/ and third_party/psd_projection/ (the subset of the frozen submodule b566590 that
# the solver uses), with the original tests; later phases move the code into the target layout of plan Section 6.1.

project(cuadmm_reorg LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
# Explicit CUDA architectures (plan Section 16, CI): the references are built with -gencode arch=compute_80,code=sm_80;
# a bitwise comparison needs the same code generation on both sides.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 80-real CACHE STRING "CUDA architectures" FORCE)
endif()
message(STATUS "cuADMM-reorg: CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}, build type ${CMAKE_BUILD_TYPE}")

option(CUADMM_BUILD_CLI "Build the cuadmm command-line executable" ON)
option(CUADMM_BUILD_TESTS "Build the test runner" ON)
option(CUADMM_BUILD_MATLAB "Build the MATLAB MEX gateway (needs MATLAB)" OFF)
option(CUADMM_WITH_HDF5 "HDF5 reader and writer (Phase 2b)" OFF)
option(CUADMM_WITH_MATIO "MATLAB .mat readers through matio (Phase 2b)" OFF)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)

find_package(CUDAToolkit 12.8 REQUIRED)

# SuiteSparse / CHOLMOD (required): -DSUITESPARSE_INCLUDE_DIRECTORIES=... -DCHOLMOD_LIB=... -DCMAD_LIB=... -DCCOLAMD_LIB=...
find_path(SUITESPARSE_INCLUDE_DIRECTORIES
  NAMES cholmod.h
  PATHS /usr/include /usr/local/include /opt/local/include
  PATH_SUFFIXES suitesparse SuiteSparse)
if(NOT SUITESPARSE_INCLUDE_DIRECTORIES)
  message(FATAL_ERROR "SuiteSparse include directory not found: set -DSUITESPARSE_INCLUDE_DIRECTORIES=<dir with cholmod.h>")
endif()
find_library(CMAD_LIB NAMES camd PATHS /usr/local/lib)
find_library(CCOLAMD_LIB NAMES ccolamd PATHS /usr/local/lib)
find_library(CHOLMOD_LIB NAMES cholmod PATHS /usr/local/lib)
foreach(SUITESPARSE_LIB CMAD_LIB CCOLAMD_LIB CHOLMOD_LIB)
  if(NOT ${SUITESPARSE_LIB})
    message(FATAL_ERROR "${SUITESPARSE_LIB} not found: set -D${SUITESPARSE_LIB}=/path/to/lib")
  endif()
endforeach()

# BLAS and LAPACK (dsyevd of the external validator); OpenBLAS provides both
find_package(BLAS REQUIRED)
if(NOT LAPACK_LIBRARIES)
  find_package(LAPACK)
endif()
if(NOT LAPACK_LIBRARIES)
  message(STATUS "LAPACK not found: dsyevd must be provided by BLAS_LIBRARIES (e.g. OpenBLAS)")
endif()

# ---------------------------------------------------------------------------------------------- psd kernels
# Subset of the frozen psd_projection submodule (b566590): composite filters, Lanczos estimate, LOBPCG, helpers.
# Phase 6a moves them to src/cone/psd/kernels/; until then they keep their own include guard and check macros.
# SHARED, as the submodule builds it: cuadmm and the kernels both define max_dense_vector_zero_kernel (B10, fixed in
# Phase 6a); a static library would give a duplicate symbol at link time.
add_library(cuadmm_psd_kernels SHARED
  third_party/psd_projection/src/composite_FP16.cu
  third_party/psd_projection/src/lanczos.cu
  third_party/psd_projection/src/lobpcg.cu
  third_party/psd_projection/src/utils.cu)
target_include_directories(cuadmm_psd_kernels PUBLIC third_party/psd_projection/include PRIVATE src)
target_link_libraries(cuadmm_psd_kernels PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)

# ---------------------------------------------------------------------------------------------- build commit
# generated/cuadmm/version.h holds the commit of the source tree (plan Pre-0: policy, parameters and build commit in the
# summary); regenerated at every build
set(CUADMM_VERSION_HEADER "${CMAKE_BINARY_DIR}/generated/cuadmm/version.h")
add_custom_target(cuadmm_version ALL
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DOUTPUT=${CUADMM_VERSION_HEADER}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GitVersion.cmake
  BYPRODUCTS ${CUADMM_VERSION_HEADER}
  COMMENT "cuADMM-reorg: recording the build commit")

# ---------------------------------------------------------------------------------------------- solver library
file(GLOB CUADMM_LEGACY_SOURCES
  src/legacy/*.cu
  src/legacy/kernels/*.cu
  src/legacy/utils/*.cu)
# the reorganized layers (plan Section 6.1); src/legacy/ holds what later phases still have to move
set(CUADMM_SOURCES
  src/options/options.cc
  src/io/problem_data.cc
  src/io/txt.cc
  src/io/io.cc
  src/report/writers.cc
  src/report/observers.cc
  src/core/gpu_context.cu
  src/admm/penalty_policy.cc
  src/admm/admm_state.cu
  src/admm/admm_solver.cu
  src/termination/stopping_rule.cu
  src/termination/best_iterate_tracker.cu
  src/solver/solver_facade.cu)
add_library(cuadmm SHARED ${CUADMM_LEGACY_SOURCES} ${CUADMM_SOURCES})
target_include_directories(cuadmm
  PUBLIC include ${CMAKE_BINARY_DIR}/generated
  PRIVATE src src/legacy/include ${SUITESPARSE_INCLUDE_DIRECTORIES})
add_dependencies(cuadmm cuadmm_version)
target_link_libraries(cuadmm PUBLIC
  CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver
  ${CMAD_LIB} ${CCOLAMD_LIB} ${CHOLMOD_LIB}
  ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
  cuadmm_psd_kernels)

# ---------------------------------------------------------------------------------------------- executables
if(CUADMM_BUILD_CLI)
  # the command-line front-end on SolverOptions (apps/cli/cuadmm.cu); keeps the executable name of main
  add_executable(cuadmm_exe apps/cli/cuadmm.cu)
  target_link_libraries(cuadmm_exe PRIVATE cuadmm)
  target_include_directories(cuadmm_exe PRIVATE src ${CMAKE_BINARY_DIR}/generated)
  add_dependencies(cuadmm_exe cuadmm_version)
  # generator of the known-optimum TXT problems (test/regression_problems.hpp), for the reference comparisons
  add_executable(cuadmm_gen_problems apps/cli/gen_problems.cpp)
  target_include_directories(cuadmm_gen_problems PRIVATE test)
  # the option table as documentation (docs/OPTIONS.md is committed; `make cuadmm_options_md` regenerates it)
  add_custom_target(cuadmm_options_md
    COMMAND cuadmm_exe --print-options markdown > ${CMAKE_CURRENT_SOURCE_DIR}/docs/OPTIONS.md
    DEPENDS cuadmm_exe
    COMMENT "cuADMM-reorg: writing docs/OPTIONS.md from the option table")
endif()

if(CUADMM_BUILD_MATLAB)
  add_subdirectory(matlab)
endif()

# ---------------------------------------------------------------------------------------------- tests
if(CUADMM_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)
  # GPU tests (the copied suite of main and the integration tests)
  add_executable(cuadmm_tests test/tests.cpp)
  target_include_directories(cuadmm_tests PRIVATE test src src/legacy/include ${SUITESPARSE_INCLUDE_DIRECTORIES})
  target_link_libraries(cuadmm_tests PRIVATE GTest::gtest_main cuadmm CUDA::nvJitLink)
  target_compile_definitions(cuadmm_tests PRIVATE CUADMM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
  # CPU-only tests (options, penalty policies on scripted residuals, readers and writers): run on any machine
  add_executable(cuadmm_cpu_tests test/cpu_tests.cpp)
  target_include_directories(cuadmm_cpu_tests PRIVATE test src)
  target_link_libraries(cuadmm_cpu_tests PRIVATE GTest::gtest_main cuadmm)
  target_compile_definitions(cuadmm_cpu_tests PRIVATE CUADMM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
  gtest_discover_tests(cuadmm_cpu_tests DISCOVERY_MODE PRE_TEST)
  set(CUADMM_REGRESSION_DATA_DIR "${CMAKE_BINARY_DIR}/regression_data" CACHE PATH
      "Directory of the TXT problems generated by the RegressionSolver tests")
  gtest_discover_tests(cuadmm_tests
    PROPERTIES ENVIRONMENT "CUADMM_REGRESSION_DATA_DIR=${CUADMM_REGRESSION_DATA_DIR}" TIMEOUT 1800
    DISCOVERY_MODE PRE_TEST)
endif()
