# Build the LAS shared library: sparse density-fitting kernels used by
# pyscf/mcscf/las/_sparse_df.py, copied from mrh's libsint (lib/sparsedf.c).
# TODO (migration step 5): refactor or replace.
add_library (clib_las SHARED sparsedf.c)
target_link_libraries (clib_las ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
set_target_properties (clib_las PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "las")
