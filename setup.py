from setuptools import Extension, find_packages, setup

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


setup(
    name="adaptive-beamforming-toolkit",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["abf_cli"],
    ext_modules=ext_modules,
    install_requires=[
        "dash>=2.18",
        "numpy>=1.26",
        "plotly>=5.24",
        "scipy>=1.13",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "abf=abf_cli:main",
        ]
    },
)
