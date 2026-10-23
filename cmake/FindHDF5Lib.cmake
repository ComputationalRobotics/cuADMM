# FindHDF5Lib.cmake: the HDF5 C library for the native problem format (plan Section 12, optional dependency of
# Phase 2b). Deliberately simpler than CMake's FindHDF5 (no compiler wrapper is run):
#   HDF5_ROOT (cache or environment): installation prefix with include/hdf5.h and lib/libhdf5.so
# Defines HDF5LIB_FOUND, HDF5LIB_INCLUDE_DIR, HDF5LIB_LIBRARY, HDF5LIB_VERSION and the imported target cuadmm::hdf5.
find_path(HDF5LIB_INCLUDE_DIR NAMES hdf5.h
  HINTS ${HDF5_ROOT} $ENV{HDF5_ROOT} PATH_SUFFIXES include include/hdf5/serial)
find_library(HDF5LIB_LIBRARY NAMES hdf5 hdf5_serial
  HINTS ${HDF5_ROOT} $ENV{HDF5_ROOT} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu/hdf5/serial)
set(HDF5LIB_VERSION "")
if(HDF5LIB_INCLUDE_DIR AND EXISTS "${HDF5LIB_INCLUDE_DIR}/H5public.h")
  file(STRINGS "${HDF5LIB_INCLUDE_DIR}/H5public.h" _h5_version_line REGEX "#define H5_VERS_INFO")
  if(_h5_version_line MATCHES "version: ([0-9.]+)")
    set(HDF5LIB_VERSION "${CMAKE_MATCH_1}")
  endif()
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HDF5Lib REQUIRED_VARS HDF5LIB_LIBRARY HDF5LIB_INCLUDE_DIR VERSION_VAR HDF5LIB_VERSION)
if(HDF5Lib_FOUND AND NOT TARGET cuadmm::hdf5)
  add_library(cuadmm::hdf5 UNKNOWN IMPORTED)
  set_target_properties(cuadmm::hdf5 PROPERTIES
    IMPORTED_LOCATION "${HDF5LIB_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${HDF5LIB_INCLUDE_DIR}")
endif()
mark_as_advanced(HDF5LIB_INCLUDE_DIR HDF5LIB_LIBRARY)
