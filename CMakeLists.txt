cmake_minimum_required(VERSION 3.18)

# cuADMM 2.0: a GPU ADMM solver for semidefinite programs, built as a self-contained project (the cuadmm library, the
# command-line interface, the MATLAB gateway, the tests and the benchmarks). The ported engine of cuADMM 1.x lives in
# src/legacy/ behind the facade cuadmm::Solver; the PSD kernels of src/cone/psd/kernels/ come from the psd_projection
# repository (b566590, formerly a submodule). Layout: plan Section 6.1 (reorg-plan/ in the history of main).

project(cuADMM VERSION 2.0.0 LANGUAGES CXX CUDA)

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
message(STATUS "cuADMM: CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}, build type ${CMAKE_BUILD_TYPE}")

option(CUADMM_BUILD_CLI "Build the cuadmm command-line executable" ON)
option(CUADMM_BUILD_TESTS "Build the test runner" ON)
option(CUADMM_BUILD_MATLAB "Build the MATLAB MEX gateway (needs MATLAB)" OFF)
option(CUADMM_BUILD_BENCH "Build the benchmarks (bench/)" ON)
# Optional input/output dependencies (plan Section 12): AUTO enables a format when its library is found
# (-DHDF5_ROOT=<prefix>, -DMATIO_ROOT=<prefix>, -DZLIB_ROOT=<prefix>), ON requires it, OFF leaves it out.
set(CUADMM_WITH_HDF5 AUTO CACHE STRING "Native HDF5 problem format (AUTO, ON, OFF)")
set(CUADMM_WITH_MATIO AUTO CACHE STRING "SeDuMi, SDPT3 and MOSEK .mat files through matio (AUTO, ON, OFF)")
set(CUADMM_WITH_ZLIB AUTO CACHE STRING "gzip-compressed SDPA files (.dat-s.gz) through zlib (AUTO, ON, OFF)")
set_property(CACHE CUADMM_WITH_HDF5 CUADMM_WITH_MATIO CUADMM_WITH_ZLIB PROPERTY STRINGS AUTO ON OFF)
# Optional linear-system backend (plan Section 11.4, Phase 7): NVIDIA cuDSS for CudssSolver, found by
# cmake/FindCUDSS.cmake (-DCUDSS_ROOT=<prefix> with include/cudss.h and lib/libcudss.so or libcudss.so.1)
set(CUADMM_WITH_CUDSS AUTO CACHE STRING "NVIDIA cuDSS backend of the y-step, CudssSolver (AUTO, ON, OFF)")
set_property(CACHE CUADMM_WITH_CUDSS PROPERTY STRINGS AUTO ON OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# resolves an AUTO/ON/OFF option against a find_package result: CUADMM_HAVE_<name> is TRUE or FALSE afterwards
function(cuadmm_resolve_optional option_name found_var feature)
  set(value "${${option_name}}")
  string(TOUPPER "${value}" value)
  if(value STREQUAL "OFF" OR value STREQUAL "0" OR value STREQUAL "FALSE" OR value STREQUAL "NO")
    set(${option_name}_RESOLVED FALSE PARENT_SCOPE)
  elseif(${found_var})
    set(${option_name}_RESOLVED TRUE PARENT_SCOPE)
  elseif(value STREQUAL "AUTO")
    message(STATUS "cuADMM: ${feature} not found, ${option_name} is OFF")
    set(${option_name}_RESOLVED FALSE PARENT_SCOPE)
  else()
    message(FATAL_ERROR "${option_name}=ON but ${feature} was not found")
  endif()
endfunction()

if(NOT CUADMM_WITH_HDF5 STREQUAL "OFF")
  find_package(HDF5Lib)
endif()
cuadmm_resolve_optional(CUADMM_WITH_HDF5 HDF5Lib_FOUND "HDF5 (set HDF5_ROOT)")
if(NOT CUADMM_WITH_MATIO STREQUAL "OFF")
  find_package(Matio)
endif()
cuadmm_resolve_optional(CUADMM_WITH_MATIO Matio_FOUND "matio (set MATIO_ROOT)")
if(NOT CUADMM_WITH_ZLIB STREQUAL "OFF")
  find_package(ZLIB)
endif()
cuadmm_resolve_optional(CUADMM_WITH_ZLIB ZLIB_FOUND "zlib (set ZLIB_ROOT)")
message(STATUS "cuADMM: HDF5 ${CUADMM_WITH_HDF5_RESOLVED} (${HDF5LIB_VERSION}), matio ${CUADMM_WITH_MATIO_RESOLVED} (${MATIO_VERSION}), zlib ${CUADMM_WITH_ZLIB_RESOLVED} (${ZLIB_VERSION_STRING})")

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)

find_package(CUDAToolkit 12.8 REQUIRED)

# cuDSS (optional; after the toolkit, since the imported target carries CUDA::cublas and CUDA::nvrtc)
if(NOT CUADMM_WITH_CUDSS STREQUAL "OFF")
  find_package(CUDSS 1.0)
endif()
cuadmm_resolve_optional(CUADMM_WITH_CUDSS CUDSS_FOUND "cuDSS (set CUDSS_ROOT)")
message(STATUS "cuADMM: cuDSS ${CUADMM_WITH_CUDSS_RESOLVED} (${CUDSS_VERSION})")

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
# The PSD kernels (psd_projection b566590) are part of the cuadmm library: src/cone/psd/kernels/.

# ---------------------------------------------------------------------------------------------- build commit
# generated/cuadmm/version.h holds the commit of the source tree (plan Pre-0: policy, parameters and build commit in the
# summary); regenerated at every build
set(CUADMM_VERSION_HEADER "${CMAKE_BINARY_DIR}/generated/cuadmm/version.h")
add_custom_target(cuadmm_version ALL
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DOUTPUT=${CUADMM_VERSION_HEADER}
          -DVERSION=${PROJECT_VERSION} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GitVersion.cmake
  BYPRODUCTS ${CUADMM_VERSION_HEADER}
  COMMENT "cuADMM: recording the build commit")

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
  src/io/csc_builder.cc
  src/io/text_source.cc
  src/io/sha256.cc
  src/io/simplify.cc
  src/io/sdpa.cc
  src/io/h5.cc
  src/io/mat_common.cc
  src/io/sedumi.cc
  src/io/sdpt3.cc
  src/io/mosek.cc
  src/io/solution_writers.cc
  src/io/txt_writer.cc
  src/report/writers.cc
  src/report/observers.cc
  src/core/gpu_context.cu
  src/admm/penalty_policy.cc
  src/admm/admm_state.cu
  src/admm/admm_solver.cu
  src/termination/stopping_rule.cu
  src/termination/best_iterate_tracker.cu
  src/cone/cone.cu
  src/cone/cone_registry.cc
  src/cone/slice_table.cu
  src/cone/free_cone.cu
  src/cone/nonnegative_cone.cu
  src/cone/cone_product.cu
  src/cone/psd/psd_cone.cu
  src/cone/psd/psd_kernels.cu
  src/cone/psd/kernels/precision.cu
  src/cone/psd/kernels/lanczos.cu
  src/cone/psd/kernels/lobpcg.cu
  src/cone/psd/kernels/composite_fp16.cu
  src/cone/psd/eig_info.cu
  src/core/workspace_arena.cu
  src/core/stream_pool.cu
  src/cone/psd/batched/batched_tier.cu
  src/cone/psd/large/large_block_policy.cc
  src/cone/psd/large/large_tier.cu
  src/cone/psd/large/rank_analyzer.cu
  src/cone/psd/large/projectors/full_evd.cu
  src/cone/psd/large/projectors/lobpcg_projector.cu
  src/cone/psd/large/projectors/composite_filter.cu
  src/termination/certificate.cu
  src/solver/solver_facade.cu
  src/linsys/normal_system.cc
  src/linsys/direct/cholmod/cholmod_solver.cu
  src/linsys/direct/dense/dense_cusolver_solver.cu
  src/problem/presolve.cc)
# the cuDSS backend (plan Section 11.4) is compiled in only when the library was found
if(CUADMM_WITH_CUDSS_RESOLVED)
  list(APPEND CUADMM_SOURCES src/linsys/direct/cudss/cudss_solver.cu)
endif()
add_library(cuadmm SHARED ${CUADMM_LEGACY_SOURCES} ${CUADMM_SOURCES})
target_include_directories(cuadmm
  PUBLIC include ${CMAKE_BINARY_DIR}/generated
  PRIVATE src src/legacy/include ${SUITESPARSE_INCLUDE_DIRECTORIES})
add_dependencies(cuadmm cuadmm_version)
target_link_libraries(cuadmm PUBLIC
  CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver
  ${CMAD_LIB} ${CCOLAMD_LIB} ${CHOLMOD_LIB}
  ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
  )
# the optional formats (plan Section 12): the readers that are not compiled in fail with a clear message
if(CUADMM_WITH_HDF5_RESOLVED)
  target_compile_definitions(cuadmm PRIVATE CUADMM_HAVE_HDF5=1)
  target_link_libraries(cuadmm PRIVATE cuadmm::hdf5)
endif()
if(CUADMM_WITH_MATIO_RESOLVED)
  target_compile_definitions(cuadmm PRIVATE CUADMM_HAVE_MATIO=1)
  target_link_libraries(cuadmm PRIVATE cuadmm::matio)
endif()
if(CUADMM_WITH_ZLIB_RESOLVED)
  target_compile_definitions(cuadmm PRIVATE CUADMM_HAVE_ZLIB=1)
  target_link_libraries(cuadmm PRIVATE ZLIB::ZLIB)
endif()
# the optional cuDSS backend (plan Section 11.4): CudssSolver exists only with CUADMM_HAVE_CUDSS
if(CUADMM_WITH_CUDSS_RESOLVED)
  target_compile_definitions(cuadmm PRIVATE CUADMM_HAVE_CUDSS=1)
  target_link_libraries(cuadmm PRIVATE cuadmm::cudss)
endif()

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
  # cuadmm-convert: any supported input format to the native HDF5 file or a TXT directory (plan Section 12)
  add_executable(cuadmm_convert apps/cli/cuadmm_convert.cc)
  set_target_properties(cuadmm_convert PROPERTIES OUTPUT_NAME cuadmm-convert)
  target_link_libraries(cuadmm_convert PRIVATE cuadmm)
  target_include_directories(cuadmm_convert PRIVATE src ${CMAKE_BINARY_DIR}/generated)
  add_dependencies(cuadmm_convert cuadmm_version)
  # the option table as documentation (docs/OPTIONS.md is committed; `make cuadmm_options_md` regenerates it)
  add_custom_target(cuadmm_options_md
    COMMAND cuadmm_exe --print-options markdown > ${CMAKE_CURRENT_SOURCE_DIR}/docs/OPTIONS.md
    DEPENDS cuadmm_exe
    COMMENT "cuADMM: writing docs/OPTIONS.md from the option table")
endif()

if(CUADMM_BUILD_MATLAB)
  add_subdirectory(matlab)
endif()

# ---------------------------------------------------------------------------------------------- documentation
# `cmake --build <dir> --target cuadmm_docs` renders the API documentation with Doxygen into docs/api/html (plan Phase 10)
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
  add_custom_target(cuadmm_docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Doxygen: docs/api/html"
    VERBATIM)
else()
  message(STATUS "cuADMM: Doxygen not found, the target cuadmm_docs is unavailable (run `doxygen docs/Doxyfile` with Doxygen installed)")
endif()

# ---------------------------------------------------------------------------------------------- benchmarks
if(CUADMM_BUILD_BENCH)
  # reading rate of the problem formats (plan Phase 2b acceptance: 1e8 nonzeros from .h5 limited by the disk)
  add_executable(cuadmm_io_bench bench/io_read_bench.cc)
  target_link_libraries(cuadmm_io_bench PRIVATE cuadmm)
  # the FP16 composite filter against the exact FP64 EVD on synthetic spectra at n = 2000..20000
  add_executable(cuadmm_psd_projector_bench bench/psd_projector_bench.cu)
  target_include_directories(cuadmm_psd_projector_bench PRIVATE src ${CMAKE_BINARY_DIR}/generated)
  target_link_libraries(cuadmm_psd_projector_bench PRIVATE cuadmm CUDA::cudart CUDA::cublas CUDA::cusolver)
  add_dependencies(cuadmm_psd_projector_bench cuadmm_version)
  # the normal-equation backends on the systems of problem directories
  add_executable(cuadmm_linsys_bench bench/linsys_bench.cu)
  target_include_directories(cuadmm_linsys_bench PRIVATE src src/legacy/include ${CMAKE_BINARY_DIR}/generated ${SUITESPARSE_INCLUDE_DIRECTORIES})
  target_link_libraries(cuadmm_linsys_bench PRIVATE cuadmm CUDA::cudart CUDA::cublas CUDA::cusolver)
  if(CUADMM_WITH_CUDSS_RESOLVED)
    target_compile_definitions(cuadmm_linsys_bench PRIVATE CUADMM_HAVE_CUDSS=1)
  endif()
  add_dependencies(cuadmm_linsys_bench cuadmm_version)
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
  # external data of the format tests (skipped when absent): the PLATO/Kocvara instances in several formats under
  # CUADMM_EXTERNAL_DATA_DIR (plato_sdpa/, plato_sedumi/, plato_kocvara/txt/) and the SPOT samples (MOSEK/, TXT/)
  set(CUADMM_EXTERNAL_DATA_DIR "$ENV{HOME}/cuadmm-data" CACHE PATH "Directory of the external test data (PLATO, Kocvara)")
  set(CUADMM_SPOT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples/SPOT/data" CACHE PATH "Directory of the SPOT sample data (MOSEK/, TXT/)")
  target_compile_definitions(cuadmm_cpu_tests PRIVATE
    CUADMM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data"
    CUADMM_EXTERNAL_DATA_DIR="${CUADMM_EXTERNAL_DATA_DIR}"
    CUADMM_SPOT_DATA_DIR="${CUADMM_SPOT_DATA_DIR}")
  if(CUADMM_WITH_HDF5_RESOLVED)
    target_compile_definitions(cuadmm_cpu_tests PRIVATE CUADMM_HAVE_HDF5=1)
    target_link_libraries(cuadmm_cpu_tests PRIVATE cuadmm::hdf5)
  endif()
  if(CUADMM_WITH_MATIO_RESOLVED)
    target_compile_definitions(cuadmm_cpu_tests PRIVATE CUADMM_HAVE_MATIO=1)
    target_link_libraries(cuadmm_cpu_tests PRIVATE cuadmm::matio)
  endif()
  if(CUADMM_WITH_ZLIB_RESOLVED)
    target_compile_definitions(cuadmm_cpu_tests PRIVATE CUADMM_HAVE_ZLIB=1)
    target_link_libraries(cuadmm_cpu_tests PRIVATE ZLIB::ZLIB)
  endif()
  gtest_discover_tests(cuadmm_cpu_tests DISCOVERY_MODE PRE_TEST)
  set(CUADMM_REGRESSION_DATA_DIR "${CMAKE_BINARY_DIR}/regression_data" CACHE PATH
      "Directory of the TXT problems generated by the RegressionSolver tests")
  gtest_discover_tests(cuadmm_tests
    PROPERTIES ENVIRONMENT "CUADMM_REGRESSION_DATA_DIR=${CUADMM_REGRESSION_DATA_DIR}" TIMEOUT 1800
    DISCOVERY_MODE PRE_TEST)
  # Phase 7 backends (GPU): NormalSystem, DenseCusolverSolver and, when compiled in, CudssSolver against the CHOLMOD
  # wrapper of the ported engine (test/linsys_test.hpp)
  add_executable(cuadmm_linsys_tests test/linsys_tests.cpp)
  target_include_directories(cuadmm_linsys_tests PRIVATE test src src/legacy/include ${SUITESPARSE_INCLUDE_DIRECTORIES})
  target_link_libraries(cuadmm_linsys_tests PRIVATE GTest::gtest_main cuadmm)
  if(CUADMM_WITH_CUDSS_RESOLVED)
    target_compile_definitions(cuadmm_linsys_tests PRIVATE CUADMM_HAVE_CUDSS=1)
    target_compile_definitions(cuadmm_tests PRIVATE CUADMM_HAVE_CUDSS=1) # the end-to-end cuDSS test of the main suite
  endif()
  gtest_discover_tests(cuadmm_linsys_tests DISCOVERY_MODE PRE_TEST)
  # Phase 7 presolve (CPU only): dependent constraint rows (test/presolve_test.hpp)
  add_executable(cuadmm_presolve_tests test/presolve_tests.cpp)
  target_include_directories(cuadmm_presolve_tests PRIVATE test src)
  target_link_libraries(cuadmm_presolve_tests PRIVATE GTest::gtest_main cuadmm)
  gtest_discover_tests(cuadmm_presolve_tests DISCOVERY_MODE PRE_TEST)
endif()
