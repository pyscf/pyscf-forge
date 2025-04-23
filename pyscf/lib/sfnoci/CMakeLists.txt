# Build the SFNOCI library

add_library (clib_sfnoci SHARED SFNOCI_contract.c)
set_target_properties (clib_sfnoci PROPERTIES
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "sfnoci")