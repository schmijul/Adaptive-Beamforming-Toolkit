from setuptools import Extension, setup

import pybind11


ext_modules = [
    Extension(
        "core._beamforming_cpp",
        ["core/beamforming_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]


setup(ext_modules=ext_modules)
