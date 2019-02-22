#!/usr/bin/env python3.6

from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

meta = {}
exec(read('hdeeprm/__meta__.py'), meta)

setup(
    name=meta['name'],
    version=meta['version'],
    author=meta['author'],
    author_email=meta['author_email'],
    description=meta['description'],
    long_description=read('README.rst'),
    url=meta['url'],
    license='MIT',
    packages=[meta['name']]
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=('deep reinforcement learning workload management job '
              'scheduling resource simulator framework heterogeneous '
              'cluster batsim'),
    install_requires=[
        'defusedxml',
        'gym',
        'lxml',
        'numpy',
        'procset',
        'pybatsim',
        'torch'
    ],
    entry_points={
        'console_scripts': [
            'hdeeprm-launch = hdeeprm.cmd:launch'
        ]
    },
    include_package_data=True)
