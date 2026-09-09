# Build the CSFstring shared library
add_library (clib_csf SHARED csf_string.c csf_op.c csf_bits.c)
target_link_libraries (clib_csf ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
set_target_properties (clib_csf PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "csf")


