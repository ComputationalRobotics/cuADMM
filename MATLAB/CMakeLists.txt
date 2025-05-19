cmake_minimum_required(VERSION 3.9)

project(
    cuADMM_MATLAB
    LANGUAGES CXX CUDA
)

# find MATLAB, LAPACK, and BLAS
find_package(Matlab REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

set(TOP_SOURCE_NAME cuadmm_MATLAB.cu)
set(TARGET_NAME cuadmm_MATLAB)

set(SUITESPARSE_INCLUDE_DIRECTOIRES /usr/local/include/suitesparse)
set(CMAD_LIB /usr/local/lib/libcamd.so)
set(CCOLAMD_LIB /usr/local/lib/libccolamd.so)
set(CHOLMOD_LIB /usr/local/lib/libcholmod.so)
# set(CHOLMOD_LIB /usr/local/MATLAB/R2023b/bin/glnxa64/libmwcholmod.so)

file(
    GLOB_RECURSE MEX_SOURCES
    ${TOP_SOURCE_NAME}
    # ${CMAKE_CURRENT_SOURCE_DIR}/../src/io/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/kernels/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/../src/solvers/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/*.cu
)

set(
    INCLUDE_DIRS
    ${Matlab_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${SUITESPARSE_INCLUDE_DIRECTOIRES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

include_directories(${TARGET_NAME} PUBLIC ${INCLUDE_DIRS})

matlab_add_mex(
    NAME ${TARGET_NAME}
    SRC ${MEX_SOURCES}
    OUTPUT_NAME ${TARGET_NAME}
    LINK_TO 
    ${Matlab_LIBRARIES} 
    cudart cusparse cublas cusolver 
    ${CMAD_LIB} ${CCOLAMD_LIB} ${CHOLMOD_LIB}
    ${LAPACK_LIBRARIES} -llapacke 
    ${BLAS_LIBRARIES}
)

set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
