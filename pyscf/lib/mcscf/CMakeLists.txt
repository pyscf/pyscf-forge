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

set(GAS_FCI_SOURCE_FILES
    fci_string_gas.c
    fci_contract_gas.c
    fci_hdiag_gas.c
    fci_rdm_gas.c
)

add_library(clib_fci_gas SHARED ${GAS_FCI_SOURCE_FILES})

target_include_directories(clib_fci_gas PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(clib_fci_gas PRIVATE c_std_11)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(clib_fci_gas PRIVATE
      -Wall
      -Wextra
      -Wpedantic
  )
endif()

target_link_libraries(clib_fci_gas PRIVATE m)

if(ENABLE_OPENMP AND TARGET OpenMP::OpenMP_C)
  target_link_libraries(clib_fci_gas PRIVATE OpenMP::OpenMP_C)
endif()

set_target_properties(clib_fci_gas PROPERTIES
    LINKER_LANGUAGE C
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    CLEAN_DIRECT_OUTPUT 1
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_NAME "fci_gas"
)
