#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = ['opencv-python', 'matplotlib', 'numpy', 'sklearn']

setup(
    author="Sebastian Kulla",
    author_email='sebastiankulla90@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A small Python tool that allows to convert fotos to paint by number images",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='paint by numbers',
    name='paint_by_numbers',
    packages=find_packages(include=['sudoku_solver', 'sudoku_solver.*']),
    url='https://github.com/sebastiankulla/paint_by_numbers',
    version='0.1.1',
    zip_safe=False,
)
