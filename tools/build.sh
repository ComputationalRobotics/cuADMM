#!/bin/bash
# usage: tools/build.sh <build_dir> [extra cmake args]   -- configure (once) and build cuADMM
# Pinned configuration of tools/reference/pinned_config.md: paths come from the environment, with the FASRC defaults.
#   CUADMM_SUITESPARSE_DIR  prefix with include/suitesparse and lib/{libcholmod,libcamd,libccolamd}.so  [$HOME/cuadmm-env]
#   CUADMM_BLAS             BLAS library (OpenBLAS provides LAPACK too)                                  [$CUADMM_SUITESPARSE_DIR/lib/libblas.so]
#   CUADMM_GTEST_SRC        googletest source directory for offline builds (FETCHCONTENT_SOURCE_DIR_GOOGLETEST) [optional]
#   CUADMM_CUDA_ARCH        CMAKE_CUDA_ARCHITECTURES                                                       [80-real]
#   BUILD_JOBS              parallel jobs                                                                  [16]
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd)
BLD=$1; shift
SS=${CUADMM_SUITESPARSE_DIR:-$HOME/cuadmm-env}
BLAS=${CUADMM_BLAS:-$SS/lib/libblas.so}
GT=${CUADMM_GTEST_SRC:-$HOME/cuADMM/build/_deps/googletest-src}
GTARG=""
[ -d "$GT" ] && GTARG="-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$GT"
if [ ! -f "$BLD/CMakeCache.txt" ]; then
  mkdir -p "$BLD"
  cmake -S "$SRC" -B "$BLD" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=${CUADMM_CUDA_ARCH:-80-real} \
    -DSUITESPARSE_INCLUDE_DIRECTORIES=$SS/include/suitesparse \
    -DCMAD_LIB=$SS/lib/libcamd.so -DCCOLAMD_LIB=$SS/lib/libccolamd.so -DCHOLMOD_LIB=$SS/lib/libcholmod.so \
    -DBLAS_LIBRARIES=$BLAS $GTARG "$@" > "$BLD/configure.log" 2>&1 || { tail -40 "$BLD/configure.log"; exit 1; }
fi
cmake --build "$BLD" -j${BUILD_JOBS:-16} > "$BLD/last_build.log" 2>&1; rc=$?
grep -E "error|Error|warning: unused" "$BLD/last_build.log" | grep -v "Wno-deprecated\|-Werror" | head -40
grep -E "Built target" "$BLD/last_build.log" | tail -8
echo "build exit code: $rc"
exit $rc
