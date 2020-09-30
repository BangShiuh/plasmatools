#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

cython_ext = cythonize([Extension('plasmatools.vibstates',
                                  ['plasmatools/vibstates.pyx'])
                       ])

setup(
    author="Bang-Shiuh Chen",
    author_email='bangshiuh.chen@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Useful tools for plasma modelling",
    license="MIT license",
    name='plasmatools',
    packages=find_packages(),
    test_suite='tests',
    url='https://github.com/bangshiuh/plasmatools',
    version='0.1.0',
    ext_modules= cython_ext
)
