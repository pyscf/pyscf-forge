# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Extended nonorthogonal Wick theorem kernels

set(NONORTHWICK_SOURCE_FILES
    nowick.c
    nowick_d.c
    nowick_z.c
)

# Packed AVX2 and AVX-512 kernels are compiled separately and selected at run
# time, so the library does not require either instruction set.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64"
   AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  list(APPEND NONORTHWICK_SOURCE_FILES nowick_avx2.c nowick_avx512.c)
  set_source_files_properties(nowick_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  set_source_files_properties(nowick_avx512.c PROPERTIES COMPILE_OPTIONS "-mavx512f")
  set_source_files_properties(nowick.c PROPERTIES COMPILE_DEFINITIONS NOWICK_X86_SIMD)
endif()

add_library(clib_nonorthwick SHARED ${NONORTHWICK_SOURCE_FILES})

target_include_directories(clib_nonorthwick PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # Inline complex multiplication without the C99 Annex G NaN recovery
  target_compile_options(clib_nonorthwick PRIVATE -O3 -fcx-limited-range)
endif()

target_link_libraries(clib_nonorthwick m ${OPENMP_C_PROPERTIES})

set_target_properties(clib_nonorthwick PROPERTIES
    LINKER_LANGUAGE C
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "nonorthwick"
)
