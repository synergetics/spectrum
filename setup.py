#!/usr/bin/env python

from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="higher-spectrum",
    version="0.2.0",
    packages=find_packages(),
    install_requires=["numpy>=1.18.0", "scipy>=1.4.0", "matplotlib>=3.1.0"],
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.6",
)
