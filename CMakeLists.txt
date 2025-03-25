cmake_minimum_required(VERSION 3.9)

project(
    cuADMM
    LANGUAGES CXX CUDA
)

### GoogleTest for unit testing
# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


### Find LAPACK and BLAS
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
# find_package(Matlab REQUIRED)

set_property(TARGET ${ROUTINE} PROPERTY CUDA_ARCHITECTURES OFF)

set(CUADMM_LIB cuadmm_lib)
set(CUADMM_EXE cuadmm_exe)

set(SUITESPARSE_INCLUDE_DIRECTOIRES /usr/local/include/suitesparse)
set(CMAD_LIB /usr/local/lib/libcamd.so)
set(CCOLAMD_LIB /usr/local/lib/libccolamd.so)
set(CHOLMOD_LIB /usr/local/lib/libcholmod.so)

set(CMAKE_INSTALL_PREFIX ../)

set(
    INCLUDE_DIRS
    # ${Matlab_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${SUITESPARSE_INCLUDE_DIRECTOIRES}
    include
)

file(
    GLOB_RECURSE LIB_SOURCES
    # src/io/*.cu
    src/kernels/*.cu
    # src/solvers/*.cu
    # src/utils/*.cu
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
    # ${Matlab_LIBRARIES}
    ${LAPACK_LIBRARIES} -llapacke
    ${BLAS_LIBRARIES}
)

### Create executable
add_executable(${CUADMM_EXE} src/main.cu)
set_property(TARGET ${CUADMM_EXE} PROPERTY CUDA_ARCHITECTURES OFF)
target_include_directories(${CUADMM_EXE} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(
    ${CUADMM_EXE}
    PUBLIC
    ${CUADMM_LIB}
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
)

include(GoogleTest)
gtest_discover_tests(tests)