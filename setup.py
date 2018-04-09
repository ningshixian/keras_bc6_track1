# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='keras_bc6_track1',
    version='0.1.0',
    description='BioCreative 6 track1 BioId Assignment',
    long_description=readme,
    author='nsx',
    author_email='ningshixian@mail.dlut.edu.cn',
    url='https://github.com/ningshixian/keras_bc6_track1',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

