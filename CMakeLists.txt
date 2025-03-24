cmake_minimum_required(VERSION 3.9)

project(
    cuADMM
    LANGUAGES CXX CUDA
)

# find LAPACK ans BLAS in MATLAB
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(Matlab REQUIRED)

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
    ${Matlab_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${SUITESPARSE_INCLUDE_DIRECTOIRES}
    include
)

file(
    # GLOB_RECURSE LIB_SOURCES
    # src/io/*.cu
    # src/kernels/*.cu
    # src/solvers/*.cu
    # src/utils/*.cu
)

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
    ${LAPACK_LIBRARIES} -llapacke
    ${BLAS_LIBRARIES}
)

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
