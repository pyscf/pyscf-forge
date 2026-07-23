# Build the GBCI/GBPDFT helper libraries

add_library (clib_gbpdft SHARED otpd_contract.c)
target_link_libraries (clib_gbpdft ${OPENMP_C_PROPERTIES})
set_target_properties (clib_gbpdft PROPERTIES
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "gbpdft")

add_library (clib_gbci SHARED gbci_contract.c)
target_link_libraries (clib_gbci ${OPENMP_C_PROPERTIES})
set_target_properties (clib_gbci PROPERTIES
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "gbci")
