# FindCUDSS.cmake: NVIDIA cuDSS, the sparse direct backend of the y-step (plan Section 11.4, CudssSolver, Phase 7;
# optional dependency CUADMM_WITH_CUDSS). cuDSS is distributed separately from the CUDA toolkit (a tarball, or the
# PyPI wheel nvidia-cudss-cu12, which ships lib/libcudss.so.1 without the unversioned libcudss.so symlink: both
# names are accepted, and linking by the full path records the SONAME libcudss.so.1 and the directory in the RPATH).
#   CUDSS_ROOT (cache or environment): installation prefix with include/cudss.h and lib/libcudss.so[.1]
# Defines CUDSS_FOUND, CUDSS_INCLUDE_DIR, CUDSS_LIBRARY, CUDSS_VERSION and the imported target cuadmm::cudss (which
# carries cuBLAS and NVRTC, the two CUDA libraries libcudss.so.1 needs; find_package(CUDAToolkit) must come first).
find_path(CUDSS_INCLUDE_DIR NAMES cudss.h
  HINTS ${CUDSS_ROOT} $ENV{CUDSS_ROOT} PATH_SUFFIXES include)
find_library(CUDSS_LIBRARY NAMES cudss libcudss.so.1
  HINTS ${CUDSS_ROOT} $ENV{CUDSS_ROOT} PATH_SUFFIXES lib lib64)
set(CUDSS_VERSION "")
if(CUDSS_INCLUDE_DIR AND EXISTS "${CUDSS_INCLUDE_DIR}/cudss.h")
  file(STRINGS "${CUDSS_INCLUDE_DIR}/cudss.h" _cudss_major REGEX "#define CUDSS_VERSION_MAJOR[ \t]+[0-9]+")
  file(STRINGS "${CUDSS_INCLUDE_DIR}/cudss.h" _cudss_minor REGEX "#define CUDSS_VERSION_MINOR[ \t]+[0-9]+")
  file(STRINGS "${CUDSS_INCLUDE_DIR}/cudss.h" _cudss_patch REGEX "#define CUDSS_VERSION_PATCH[ \t]+[0-9]+")
  if(_cudss_major MATCHES "([0-9]+)$" )
    set(_cudss_v_major "${CMAKE_MATCH_1}")
    string(REGEX MATCH "([0-9]+)$" _cudss_v_minor "${_cudss_minor}")
    string(REGEX MATCH "([0-9]+)$" _cudss_v_patch "${_cudss_patch}")
    set(CUDSS_VERSION "${_cudss_v_major}.${_cudss_v_minor}.${_cudss_v_patch}")
  endif()
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDSS REQUIRED_VARS CUDSS_LIBRARY CUDSS_INCLUDE_DIR VERSION_VAR CUDSS_VERSION)
if(CUDSS_FOUND AND NOT TARGET cuadmm::cudss)
  add_library(cuadmm::cudss UNKNOWN IMPORTED)
  set_target_properties(cuadmm::cudss PROPERTIES
    IMPORTED_LOCATION "${CUDSS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDE_DIR}")
  # libcudss.so.1 NEEDS libcublas.so.12 and libnvrtc.so.12 (readelf -d); the toolkit targets put them in the RPATH
  if(TARGET CUDA::nvrtc)
    set_property(TARGET cuadmm::cudss PROPERTY INTERFACE_LINK_LIBRARIES CUDA::cublas CUDA::nvrtc)
  else()
    set_property(TARGET cuadmm::cudss PROPERTY INTERFACE_LINK_LIBRARIES CUDA::cublas)
  endif()
endif()
mark_as_advanced(CUDSS_INCLUDE_DIR CUDSS_LIBRARY)
