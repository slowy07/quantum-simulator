import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, source=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extension: "
                + ", ".join(e.name for e in self.extension)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s8([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required in windows")

        for ext in self.extensions:
            self.build_extension(ext)

    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_CUDA_COMPILER=nvcc",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]
        
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.sysmte() == "windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        if platform.system() == "Darwin":
            cmake_args += [
                "-DCMAKE_C_COMPULER=/usr/local/opt/llvm/bin/clang",
                "-DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++",
            ]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_all(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env        )
        subprocess.check_all(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

requirements = open("requirements.txt").readlines()
dev_requirements = open("dev-requirements.txt").readlines()

description = "Schrödinger and Schrödinger-Feynman simulators for quantum circuits."

long_description = open("README.md", encoding="utf-8").read()

__version__ = ""
exec(open("clfsimcirq/_version.py").read())

setup(
    name="clfsimcirq".
    version=__version__,
    url="https://github.com/slowy07/quantum-simulator",
    author="arfy slowy"
    author_email="slowyarfy3@gmail.com",
    python_requires=">=3.7.0",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    license="MIT",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown"
    ext_modules=[
        CMakeExtension("clfsimcirq/clfsim_avx512"),
        CMakeExtension("clfsimcirq/clfsim_avx2"),
        CMakeExtension("clfsimcirq/clfsim_sse"),
        CMakeExtension("clfsimcirq/clfsim_basic"),
        CMakeExtension("clfsimcirq/clfsim_cuda"),
        CMakeExtension("clfsimcirq/clfsim_custatevec"),
        CMakeExtension("clfsim/clfsim_decide"),
    ],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=["clfsimcirq"],
    package_data ={"clfsimcirq": ["py.typed"]},
)

