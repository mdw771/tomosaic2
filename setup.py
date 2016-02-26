#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages, os

setup(
    name='tomosaic',
    packages=find_packages(exclude=['test*']),
    version=open('VERSION').read().strip(),
    include_package_data=True,
    ext_modules=[],
    zip_safe=False,
    author='Rafael Vescovi',
    author_email='ravescovi@aps.anl.gov',
    description='Mosaic Tomographic Reconstruction in Python.',
    keywords=['mosaic tomography', 'panorama', 'imaging'],
    url='http://tomosaic.readthedocs.org',
    download_url='http://github.com/tomosaic/tomosaic.git',
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: C']
)
