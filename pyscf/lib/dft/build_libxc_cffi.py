import os
import sys
import cffi
import sysconfig
import shutil

forge_path = os.path.dirname(__file__)
helper_lib_path = os.path.join(forge_path, '..', '..', 'dft2')
sys.path.append(helper_lib_path)
from _libxc_header import preprocess_header, load_file

package_path = sys.argv[1]
header_file_path = os.path.join(package_path, 'lib', 'deps', 'include', 'xc.h')
library_dir = os.path.join(package_path, 'lib', 'deps', 'lib')
include_dir = os.path.join(package_path, 'lib', 'deps', 'include')
itrf_c_path = os.path.join(forge_path, 'libxc_itrf2.c')
itrf_h_path = os.path.join(forge_path, 'libxc_itrf2.h')

ffi = cffi.FFI()
src = preprocess_header(header_file_path)
ffi.cdef(src + load_file(itrf_h_path))
ffi.set_source("libxc_cffi", load_file(itrf_c_path),
               extra_link_args=["-L", library_dir, "-l:libxc.so",
                                "-Wl,-rpath=" + library_dir],
               extra_compile_args=["-I", include_dir])
ffi.compile()
so_name = "libxc_cffi" + sysconfig.get_config_var('EXT_SUFFIX')
shutil.move(so_name, os.path.join(forge_path, '..', '..', 'dft2', so_name))

