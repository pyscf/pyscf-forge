cmake_minimum_required (VERSION 3.5)
project (pyscf_forge)

set(CMAKE_VERBOSE_MAKEFILE OFF)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # TODO: If updating to minimum requirement cmake>=3.7, use
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  # see also issue #661
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()

set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")
option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
if (BUILD_MARCH_NATIVE)
  include(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
      message("Add CFLAGS -march=native -unroll-aggressive")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -unroll-aggressive -ipo")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    else()
      message("Add CFLAGS -march=native -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -funroll-loops -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    endif()
  endif()

  include (CheckCXXCompilerFlag)
  # Compile to run of this machine
  check_cxx_compiler_flag (-xHost HAS_XHOST)
  check_cxx_compiler_flag (-march=native HAS_MARCH_NATIVE)
  if (HAS_XHOST)
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")
  elseif (HAS_MARCH_NATIVE)
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif ()

  # Do interparts optimization
  check_cxx_compiler_flag (-ipo HAS_IPO)
  if (HAS_IPO)
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
  endif (HAS_IPO)
else()
  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG("-msse3" COMPILER_SUPPORTS_SSE3)
    if(COMPILER_SUPPORTS_SSE3)
      # Avoids error "SIMDD undeclared here (not in a function)" in the qcint two-electron integral interface
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
    endif()
  endif()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Whether or not to compile with MKL
option (MKL "Compile using the MKL" OFF)
if (MKL)
    set ( ENV{BLA_VENDOR} "Intel10_64lp" )
endif (MKL)
#link_directories ($ENV{LD_LIBRARY_PATH})

find_package(BLAS REQUIRED)
#find_package (LAPACK REQUIRED)

# OpenMP
option(ENABLE_OPENMP "Compiling C extensions with openmp" ON)
set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

option(PYSCF_SOURCE_DIR "Pyscf source directory e.g. /usr/local/lib/python3.7/site-packages/pyscf")
if(NOT PYSCF_SOURCE_DIR)
  execute_process(
    COMMAND python3 -c "import pyscf; print(pyscf.__path__[0], end='')"
    OUTPUT_VARIABLE PYSCF_SOURCE_DIR)
endif()
message(STATUS "Include pyscf source dir: ${PYSCF_SOURCE_DIR}")
include_directories(${PYSCF_SOURCE_DIR}/lib ${PYSCF_SOURCE_DIR}/lib/deps/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
configure_file(
  "${PYSCF_SOURCE_DIR}/lib/config.h.in"
  "${PYSCF_SOURCE_DIR}/lib/config.h")
# to find config.h
link_directories(${PYSCF_SOURCE_DIR}/lib/deps/lib ${PYSCF_SOURCE_DIR}/lib/deps/lib64)
link_directories(${PYSCF_SOURCE_DIR}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)
message(STATUS "${PYSCF_SOURCE_DIR}/lib may need to be put in the environment LD_LIBRARY_PATH")

# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
if (WIN32)
  #?
elseif (APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
  set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
else ()
  set(CMAKE_SKIP_BUILD_RPATH  True)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
endif ()

# Build the occri library (optional - depends on availability of FFTW, OpenMP)
option(BUILD_OCCRI "Build OCCRI C extension (requires FFTW, OpenMP)" ON)

if(BUILD_OCCRI)
    set(OCCRI_SOURCE_FILES
        occri/occri.c
        occri/occri.h
    )
    
    # Check for required dependencies
    set(OCCRI_DEPS_FOUND TRUE)
    
    # Find FFTW
    find_library(FFTW3_LIBRARY fftw3)
    find_library(FFTW3_THREADS_LIBRARY fftw3_threads)
    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    
    if(NOT FFTW3_LIBRARY OR NOT FFTW3_THREADS_LIBRARY OR NOT FFTW3_INCLUDE_DIR)
        message(STATUS "FFTW3 not found - OCCRI C extension will not be built")
        set(OCCRI_DEPS_FOUND FALSE)
    endif()
    
    # Find OpenMP
    find_package(OpenMP)
    if(NOT OPENMP_FOUND)
        message(STATUS "OpenMP not found - OCCRI C extension will not be built")
        set(OCCRI_DEPS_FOUND FALSE)
    endif()
    
    if(OCCRI_DEPS_FOUND)
        message(STATUS "Building OCCRI C extension with FFTW and OpenMP")
        add_library(occri SHARED ${OCCRI_SOURCE_FILES})
        
        # Add include directories
        target_include_directories(occri PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/occri)
        target_include_directories(occri PRIVATE ${FFTW3_INCLUDE_DIR})
        
        # Link against FFTW and OpenMP
        target_link_libraries(occri
            ${OPENMP_C_PROPERTIES}
            ${FFTW3_LIBRARY}
            ${FFTW3_THREADS_LIBRARY}
        )
        
        # Set output directory and name
        set_target_properties(occri PROPERTIES
            CLEAN_DIRECT_OUTPUT 1
            LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_NAME "occri"
        )
    else()
        message(STATUS "OCCRI C extension disabled due to missing dependencies")
        message(STATUS "Pure Python implementation will be used (performance will be reduced)")
        message(STATUS "To enable optimized version, install: FFTW3, BLAS, OpenMP")
    endif()
else()
    message(STATUS "OCCRI C extension build disabled by user")
endif()

# Build the LibXC interface v2 library
set (LIBXC_ITRF2_SOURCE_FILES "dft/libxc_itrf2.c")
add_library (clib_libxc_itrf2 SHARED ${LIBXC_ITRF2_SOURCE_FILES})
target_link_libraries (clib_libxc_itrf2 xc ${OPENMP_C_PROPERTIES})
set_target_properties (clib_libxc_itrf2 PROPERTIES
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "xc_itrf2")

# Build the DSRG library
set (DSRG_SOURCE_FILES "../dsrg_mrpt2/dsrg_helper.c")
add_library (clib_dsrg SHARED ${DSRG_SOURCE_FILES})
set_target_properties (clib_dsrg PROPERTIES
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "dsrg")

# Build the CSFstring shared library
set (CSF_SOURCE_FILES "csf/csfstring.c")
add_library (clib_csf SHARED ${CSF_SOURCE_FILES})
target_link_libraries (clib_csf ${OPENMP_C_PROPERTIES})
set_target_properties (clib_csf PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "csf")

add_subdirectory(sfnoci)

