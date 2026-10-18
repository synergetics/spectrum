#!/usr/bin/env python

import os

from setuptools import setup, find_packages, Extension

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# ---------------------------------------------------------------------------
# Native C core (optional at runtime, built whenever numpy + a compiler exist)
# ---------------------------------------------------------------------------
ext_modules = []
try:
    import numpy  # noqa: W0611

    compile_args = ["-O3", "-funroll-loops"]
    link_args = []
    if os.environ.get("SPECTRUM_ASAN") == "1":
        compile_args = ["-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer"]
        link_args = ["-fsanitize=address"]
    if os.environ.get("SPECTRUM_NATIVE") == "1":
        compile_args.append("-march=native")
    if os.environ.get("SPECTRUM_NO_OPENMP") != "1":
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

    ext_modules.append(
        Extension(
            "spectrum._spectrum_core",
            sources=[
                "spectrum/src/_spectrum_core.c",
                "spectrum/src/fft/fft_iface.c",
                "spectrum/src/fft/kissfft/kiss_fft.c",
                "spectrum/src/estimators_cum.c",
                "spectrum/src/estimators_bispec.c",
                "spectrum/src/cumtrue.c",
                "spectrum/src/wigner.c",
            ],
            include_dirs=["spectrum/src", numpy.get_include()],
            define_macros=[("kiss_fft_scalar", "double")],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    )
except ImportError:  # numpy missing: pure-Python fallback package
    pass

setup(
    name="higher-spectrum",
    version="0.3.0",
    packages=find_packages(),
    install_requires=["numpy>=1.18.0", "scipy>=1.4.0", "matplotlib>=3.1.0"],
    ext_modules=ext_modules,
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.rst", "*.mat"],
    },
    # metadata for upload to PyPI
    author="ixaxaar",
    author_email="root@ixaxaar.in",
    description="Higher Order Spectral Analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="higher order spectrum estimation toolkit signal processing",
    url="https://github.com/synergetics/spectrum",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
)
