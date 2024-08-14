import os
import sys
import cffi
import subprocess
import re
import sysconfig
import shutil

def preprocess_header(path):
    '''Call gcc to preprocess `xc.h` macros. This will let cffi recognize most constants.'''
    command = ['gcc', '-E', '-P', '-xc', '-dD', '-']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    with open(path, 'r') as f:
        for line in f:
            if not line.strip().startswith('#include'):
                process.stdin.write(line)

    process.stdin.close()

    # Process only constants with integer numbers and constants defined with one left shift operator
    # Also exclude builtins that start with underscore
    # Limitations: Flags with other operators are not included
    #     e.g.:  #define XC_FLAGS_HAVE_ALL
    CONST_PATTERN = re.compile(r'#define [^_][^ ]* -?\d+\s*')
    SHIFT_PATTERN = re.compile(r'#define ([^_][^ ]*) \(\s*(\d+)\s*<<\s*(\d+)\s*\)')

    def read():
        for line in process.stdout:
            if line.startswith('#define'):
                pat = SHIFT_PATTERN.match(line)
                if pat:
                    name, a, b = pat.groups()
                    yield f'#define {name} {int(a) << int(b)}\n'
                    continue
                elif not CONST_PATTERN.fullmatch(line):
                    #  print('Warning: ignore #define line:', line.strip())
                    continue
            yield line

    return ''.join(read())

if __name__ == '__main__':
    package_path = sys.argv[1]
    forge_path = os.path.dirname(__file__)
    header_file_path = os.path.join(package_path, 'lib', 'deps', 'include', 'xc.h')
    library_dir = os.path.join(package_path, 'lib', 'deps', 'lib')

    ffi = cffi.FFI()
    src = preprocess_header(header_file_path)
    ffi.cdef(src)
    ffi.set_source("libxc_cffi", src, extra_link_args=["-L", library_dir, "-l:libxc.so",
                                                       "-Wl,-rpath=" + library_dir])
    ffi.compile()
    so_name = "libxc_cffi" + sysconfig.get_config_var('EXT_SUFFIX')
    shutil.move(so_name, os.path.join(forge_path, '..', '..', 'dft2', so_name))

