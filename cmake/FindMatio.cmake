# FindMatio.cmake: libmatio (MATLAB .mat reader/writer; plan Section 12, optional dependency of Phase 2b).
#   MATIO_ROOT (cache or environment): installation prefix with include/matio.h and lib/libmatio.so
# Defines MATIO_FOUND, MATIO_INCLUDE_DIR, MATIO_LIBRARY, MATIO_VERSION and the imported target cuadmm::matio.
find_path(MATIO_INCLUDE_DIR NAMES matio.h
  HINTS ${MATIO_ROOT} $ENV{MATIO_ROOT} PATH_SUFFIXES include)
find_library(MATIO_LIBRARY NAMES matio
  HINTS ${MATIO_ROOT} $ENV{MATIO_ROOT} PATH_SUFFIXES lib lib64)
set(MATIO_VERSION "")
if(MATIO_INCLUDE_DIR AND EXISTS "${MATIO_INCLUDE_DIR}/matio_pubconf.h")
  file(STRINGS "${MATIO_INCLUDE_DIR}/matio_pubconf.h" _matio_version_line REGEX "#define MATIO_VERSION_STR")
  if(_matio_version_line MATCHES "\"([0-9.]+)\"")
    set(MATIO_VERSION "${CMAKE_MATCH_1}")
  endif()
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Matio REQUIRED_VARS MATIO_LIBRARY MATIO_INCLUDE_DIR VERSION_VAR MATIO_VERSION)
if(Matio_FOUND AND NOT TARGET cuadmm::matio)
  add_library(cuadmm::matio UNKNOWN IMPORTED)
  set_target_properties(cuadmm::matio PROPERTIES
    IMPORTED_LOCATION "${MATIO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MATIO_INCLUDE_DIR}")
endif()
mark_as_advanced(MATIO_INCLUDE_DIR MATIO_LIBRARY)
