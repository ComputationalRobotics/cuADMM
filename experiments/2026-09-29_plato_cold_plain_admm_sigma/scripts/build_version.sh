#!/bin/bash
# usage: build_version.sh <label> <worktree> <build_dir>     (on the GPU node; identical flags for every version)
# Writes the build manifest to builds/<label>/ in this experiment directory (PLATO cold-start plain-ADMM sigma campaign; the same flags as the hybrid campaign build hybrid_official, so the two builds are comparable bit for bit).
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
L=$1; SRC=$2; BLD=$3; MAN=$EXP/builds/$L
mkdir -p $BLD $MAN
ENVD=$HOME/cuadmm-env
# EXTRA_INCLUDE (upstream compatibility build only): directory with the lapack.h declaration shim
INC=${EXTRA_INCLUDE:+ -I$EXTRA_INCLUDE}
FLAGS=(-DCMAKE_BUILD_TYPE=Release
  "-DCMAKE_CUDA_FLAGS=-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90$INC"
  "-DCMAKE_CXX_FLAGS=$INC"
  -DCMAKE_DISABLE_FIND_PACKAGE_Matlab=${DISABLE_MATLAB:-TRUE}
  -DSUITESPARSE_INCLUDE_DIRECTORIES=$ENVD/include/suitesparse
  -DCMAD_LIB=$ENVD/lib/libcamd.so -DCCOLAMD_LIB=$ENVD/lib/libccolamd.so -DCHOLMOD_LIB=$ENVD/lib/libcholmod.so
  -DBLAS_LIBRARIES=$ENVD/lib/libblas.so -DLAPACK_LIBRARIES=$ENVD/lib/liblapack.so
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/n/home00/yukuanwei/cuADMM/build/_deps/googletest-src)
{
  echo "label: $L"; echo "worktree: $SRC"; echo "build dir: $BLD"; echo "date: $(date -Iseconds)"; echo "host: $(hostname)"
  echo "parent commit: $(git -C $SRC rev-parse HEAD)"; echo "psd_projection commit: $(git -C $SRC/psd_projection rev-parse HEAD)"
  echo "worktree modifications (git status --short, excluding the submodule placeholder):"; git -C $SRC status --short | grep -v psd_projection
  echo "cmake flags:"; printf '  %s\n' "${FLAGS[@]}"
} > $MAN/manifest.txt
git -C $SRC diff > $MAN/worktree.diff
cmake -S $SRC -B $BLD "${FLAGS[@]}" > $MAN/configure.log 2>&1; rc=$?
echo "configure exit code: $rc" >> $MAN/manifest.txt
if [ $rc -ne 0 ]; then tail -20 $MAN/configure.log; exit $rc; fi
cmake --build $BLD -j${JOBS:-32} -- VERBOSE=1 > $MAN/build.log 2>&1; rc=$?
echo "build exit code: $rc" >> $MAN/manifest.txt
{
  echo "CMakeCache:"; grep -E "^(CMAKE_BUILD_TYPE|CMAKE_CUDA_FLAGS|CMAKE_CUDA_FLAGS_RELEASE|CMAKE_CXX_FLAGS_RELEASE|CMAKE_CUDA_COMPILER|CMAKE_CXX_COMPILER|CHOLMOD_LIB|BLAS_LIBRARIES|SUITESPARSE_INCLUDE_DIRECTORIES|CMAKE_CUDA_ARCHITECTURES)[:=]" $BLD/CMakeCache.txt | sed 's/^/  /'
  echo "sample nvcc command:"; grep -m1 -E "nvcc .*solver\.cu" $MAN/build.log | cut -c1-600 | sed 's/^/  /'
  for f in $BLD/libcuadmm_lib.so $BLD/psd_projection/libpsd_lib.so $BLD/cuadmm_exe $BLD/tests; do
    [ -f $f ] || { echo "missing: $f"; continue; }
    echo "$(sha256sum $f)"
    echo "  cuobjdump --list-elf: $(cuobjdump --list-elf $f 2>/dev/null | grep -o 'sm_[0-9]*' | sort | uniq -c | tr '\n' ' ')"
    echo "  cuobjdump --list-ptx: $(cuobjdump --list-ptx $f 2>/dev/null | grep -o 'sm_[0-9]*\|compute_[0-9]*' | sort | uniq -c | tr '\n' ' ')"
  done
} >> $MAN/manifest.txt 2>&1
grep -E "error|Error" $MAN/build.log | grep -v -E "Wno-deprecated|error_|Werror|errors\.cpp|errno" | head -20
grep -E "exit code|sha256|cuobjdump" $MAN/manifest.txt
exit $rc
