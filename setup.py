# coding: utf-8

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

# from Cython.Compiler import Options
# from Cython.Build import cythonize


#long_description = open('README.rst').read()

setup(
    name = 'irsa',
    version = '0.1',
    description = 'Intelligent Raman Spectra Analysis',
    author = 'Zaur Shibzukhov',
    author_email = "szport@gmail.com",
    license = "MIT License",
    # ext_modules = cythonize(ext_modules, nthreads=4,
    #                         compiler_directives=cython_compiler_directives2),
    # ext_modules = ext_modules,
    package_dir = {'': 'lib'},
    # cmdclass = {'build_ext': build_ext},
    packages = ['irsa', 'irsa.io', 
                'irsa.spectra', 
                'irsa.preprocess', 
                'irsa.preprocess.despike', 'irsa.preprocess.smooth',
                'irsa.plots', 'irsa.widgets',
               ],
    # package_data = {'': ['*.pxd']},
    # url = 'https://bitbucket.org/intellimath/mlgrad',
    # download_url = 'https://bitbucket.org/intellimath/mlgrad',
    long_description = "", # long_description,
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
