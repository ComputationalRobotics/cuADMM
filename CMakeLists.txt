cmake_minimum_required(VERSION 3.10)

project(
    cuADMM
    LANGUAGES CXX CUDA
)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER "/usr/bin/gcc")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")

# Explicitly add CUDA library path for linker
link_directories(/usr/local/cuda/lib64)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Find CUDA Toolkit and use imported targets for linking
find_package(CUDAToolkit REQUIRED)
set_property(TARGET ${ROUTINE} PROPERTY CUDA_ARCHITECTURES OFF)

find_package(BLAS REQUIRED)
find_package(Matlab REQUIRED)

set_property(TARGET ${ROUTINE} PROPERTY CUDA_ARCHITECTURES OFF)

set(CUADMM_LIB cuadmm_lib)
set(CUADMM_EXE cuadmm_exe)

find_path(SUITESPARSE_INCLUDE_DIRECTORIES
  NAMES SuiteSparse_config.h
  PATHS /usr/include /usr/local/include /opt/local/include
  PATH_SUFFIXES suitesparse SuiteSparse
)
if(NOT SUITESPARSE_INCLUDE_DIRECTORIES)
  message(FATAL_ERROR "SuiteSparse include directory not found. Please set SUITESPARSE_INCLUDE_DIRECTORIES manually.")
endif()
set(CMAD_LIB /usr/local/lib/libcamd.so)
set(CCOLAMD_LIB /usr/local/lib/libccolamd.so)
set(CHOLMOD_LIB /usr/local/lib/libcholmod.so)

set(CMAKE_INSTALL_PREFIX ../)

set(
    INCLUDE_DIRS
    ${Matlab_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${SUITESPARSE_INCLUDE_DIRECTORIES}
    include
)

file(
    GLOB_RECURSE LIB_SOURCES
    src/*.cu
    src/kernels/*.cu
    src/utils/*.cu
)

### Create library
add_library(${CUADMM_LIB} SHARED ${LIB_SOURCES})
set_property(TARGET ${CUADMM_LIB} PROPERTY CUDA_ARCHITECTURES OFF)
target_include_directories(${CUADMM_LIB} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(
    ${CUADMM_LIB}
    PUBLIC
    cudart cusparse cublas cusolver
    ${CMAD_LIB}
    ${CCOLAMD_LIB}
    ${CHOLMOD_LIB}
    ${Matlab_LIBRARIES}
    ${BLAS_LIBRARIES}
    psd_lib
)

# Disable building tests in psd_projection to avoid target name conflict
set(PSD_PROJECTION_BUILD_TESTS OFF CACHE BOOL "Disable psd_projection tests" FORCE)
# Add psd_projection submodule
add_subdirectory(psd_projection)
list(APPEND INCLUDE_DIRS psd_projection/include)

### Create executable
add_executable(${CUADMM_EXE} src/main.cu)
set_property(TARGET ${CUADMM_EXE} PROPERTY CUDA_ARCHITECTURES OFF)
target_include_directories(${CUADMM_EXE} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(
    ${CUADMM_EXE}
    PUBLIC
    ${CUADMM_LIB}
    psd_lib
)

install(
    TARGETS ${CUADMM_LIB}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

### Configure unit tests
enable_testing()

add_executable(
  tests
  test/tests.cpp
)
target_link_libraries(
  tests
  GTest::gtest_main
  ${CUADMM_LIB}
  nvJitLink # avoid linking error
)

include(GoogleTest)
gtest_discover_tests(tests)