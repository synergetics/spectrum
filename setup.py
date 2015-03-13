#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
  name = "spectrum",
  version = "0.1",
  packages = find_packages(),
  # scripts = ['say_hello.py'],

  install_requires = ['numpy>=1.8', 'scipy>13.3', 'matplotlib>1.3.1'],

  include_package_data = True,

  package_data = {
    '': ['*.txt', '*.rst', '*.mat'],
  },

  # metadata for upload to PyPI
  # author = "Me",
  # author_email = "me@example.com",
  # description = "This is an Example Package",
  # license = "PSF",
  # keywords = "hello world example examples",
  # url = "http://example.com/HelloWorld/",   # project home page, if any
)
