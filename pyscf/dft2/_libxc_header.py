import sys
import subprocess
import re

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

def load_file(path):
    with open(path) as f:
        return f.read()

