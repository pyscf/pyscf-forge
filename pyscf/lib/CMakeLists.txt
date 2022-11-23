project (clib_pyscf_mcpdft)

cmake_minimum_required (VERSION 2.8.11)
set (CMAKE_VERBOSE_MAKEFILE OFF)

# Whether or not to compile with MKL
option (MKL "Compile using the MKL" OFF)
if (MKL)
    set ( ENV{BLA_VENDOR} "Intel10_64lp" )
endif (MKL)
link_directories ($ENV{LD_LIBRARY_PATH})

find_package (LAPACK REQUIRED)

# OpenMP
find_package (OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

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

# With debug symbols and highly optimized
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3")

# Build the PDFT library
set (PDFT_SOURCE_FILES "../mcpdft/nr_numint.c")
add_library (clib_pdft SHARED ${PDFT_SOURCE_FILES})
target_link_libraries (clib_pdft ${LAPACK_LIBRARIES})
set_target_properties (clib_pdft PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "pdft")


