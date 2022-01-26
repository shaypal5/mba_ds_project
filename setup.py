"""Setup for the mba package."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


README_RST = ''
with open('README.rst', encoding="utf-8") as f:
    README_RST = f.read()

INSTALL_REQUIRES = [
    'pandas>=0.18.0',  # obviously
    'pycaret==2.3.6',
]


setup(
    name='mba',
    description="MBA DS Project",
    long_description=README_RST,
    long_description_content_type='text/x-rst',
    author="Shay Palachy",
    author_email="shaypal5@gmail.com",
    version='0.0.0',
    url='https://github.com/shaypal5/mba_ds_project',
    packages=['mba'],
    install_requires=INSTALL_REQUIRES,
    extras_require={},
)
