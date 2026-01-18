#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

NAME = 'pyscf_forge'
AUTHOR = 'Pyscf Developer'
AUTHOR_EMAIL = None
DESCRIPTION  = 'Staging ground for PySCF core features'
SO_EXTENSIONS = {
}
DEPENDENCIES = ['pyscf', 'numpy!=2.4.*']
VERSION = '1.0.3'

#######################################################################
# Unless not working, nothing below needs to be changed.
metadata = globals()
import os
import sys
from setuptools import setup, find_namespace_packages, Extension
from setuptools.command.build_py import build_py

topdir = os.path.abspath(os.path.join(__file__, '..'))
modules = find_namespace_packages(include=['pyscf.*'])

def get_platform():
    from distutils.util import get_platform
    platform = get_platform()
    if sys.platform == 'darwin':
        arch = os.getenv('CMAKE_OSX_ARCHITECTURES')
        if arch:
            osname = platform.rsplit('-', 1)[0]
            if ';' in arch:
                platform = f'{osname}-universal2'
            else:
                platform = f'{osname}-{arch}'
        elif os.getenv('_PYTHON_HOST_PLATFORM'):
            # the cibuildwheel environment
            platform = os.getenv('_PYTHON_HOST_PLATFORM')
            if platform.endswith('arm64'):
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'arm64')
            elif platform.endswith('x86_64'):
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'x86_64')
            else:
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'arm64;x86_64')
    return platform

pyscf_lib_dir = os.path.join(topdir, 'pyscf', 'lib')
def make_ext(pkg_name, srcs,
             libraries=[], library_dirs=[pyscf_lib_dir],
             include_dirs=[], extra_compile_flags=[],
             extra_link_flags=[], **kwargs):
    if sys.platform.startswith('darwin'):  # OSX
        from distutils.sysconfig import get_config_vars
        conf_vars = get_config_vars()
        conf_vars['LDSHARED'] = conf_vars['LDSHARED'].replace('-bundle', '-dynamiclib')
        conf_vars['CCSHARED'] = " -dynamiclib"
        conf_vars['EXT_SUFFIX'] = '.dylib'
        soname = pkg_name.split('.')[-1]
        extra_link_flags = extra_link_flags + ['-install_name', f'@loader_path/{soname}.dylib']
        runtime_library_dirs = []
    else:
        extra_compile_flags = extra_compile_flags + ['-fopenmp']
        extra_link_flags = extra_link_flags + ['-fopenmp']
        runtime_library_dirs = ['$ORIGIN', '.']
    os.path.join(topdir, *pkg_name.split('.')[:-1])
    return Extension(pkg_name, srcs,
                     libraries = libraries,
                     library_dirs = library_dirs,
                     include_dirs = include_dirs + library_dirs,
                     extra_compile_args = extra_compile_flags,
                     extra_link_args = extra_link_flags,
                     runtime_library_dirs = runtime_library_dirs,
                     **kwargs)

class CMakeBuildPy(build_py):
    def run(self):
        self.plat_name = get_platform()
        self.build_base = 'build'
        self.build_lib = os.path.join(self.build_base, 'lib')
        self.build_temp = os.path.join(self.build_base, f'temp.{self.plat_name}')

        self.announce('Configuring extensions', level=3)
        src_dir = os.path.abspath(os.path.join(__file__, '..', 'pyscf', 'lib'))
        cmd = ['cmake', f'-S{src_dir}', f'-B{self.build_temp}']
        
        # Allow user to disable OCCRI C extension build
        build_occri = os.getenv('BUILD_OCCRI', 'ON')
        cmd.extend([f'-DBUILD_OCCRI={build_occri}'])
        
        configure_args = os.getenv('CMAKE_CONFIGURE_ARGS')
        if configure_args:
            cmd.extend(configure_args.split(' '))
        self.spawn(cmd)

        self.announce('Building binaries', level=3)
        # Do not use high level parallel compilation. OOM may be triggered
        # when compiling certain functionals in libxc.
        cmd = ['cmake', '--build', self.build_temp, '-j', '4']
        build_args = os.getenv('CMAKE_BUILD_ARGS')
        if build_args:
            cmd.extend(build_args.split(' '))
        if self.dry_run:
            self.announce(' '.join(cmd))
        else:
            self.spawn(cmd)
        super().run()

settings = {
    'name': metadata.get('NAME', None),
    'version': VERSION,
    'description': metadata.get('DESCRIPTION', None),
    'long_description': metadata.get('DESCRIPTION', None),
    'long_description_content_type': 'text/markdown',
    'author': metadata.get('AUTHOR', None),
    'author_email': metadata.get('AUTHOR_EMAIL', None),
    'install_requires': metadata.get('DEPENDENCIES', []),
    'cmdclass': {'build_py': CMakeBuildPy},
}

if SO_EXTENSIONS:
    settings['ext_modules'] = [make_ext(k, v) for k, v in SO_EXTENSIONS.items()]
else:
    # build_py will produce plat_name = 'any'. Patch the bdist_wheel to change the
    # platform tag because the C extensions are platform dependent.
    # For setuptools<70
    from wheel.bdist_wheel import bdist_wheel
    initialize_options_1 = bdist_wheel.initialize_options
    def initialize_with_default_plat_name(self):
        initialize_options_1(self)
        self.plat_name = get_platform()
        self.plat_name_supplied = True
    bdist_wheel.initialize_options = initialize_with_default_plat_name

    # For setuptools>=70
    try:
        from setuptools.command.bdist_wheel import bdist_wheel
        initialize_options_2 = bdist_wheel.initialize_options
        def initialize_with_default_plat_name(self):
            initialize_options_2(self)
            self.plat_name = get_platform()
            self.plat_name_supplied = True
        bdist_wheel.initialize_options = initialize_with_default_plat_name
    except ImportError:
        pass

setup(
    include_package_data=True,
    packages=modules + ['pyscf.lib'],
    **settings
)
