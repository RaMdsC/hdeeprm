#!/usr/bin/env python3.6

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='hdeeprm',
    version='0.1.0',
    author='Adrián Herrera',
    author_email='adr.her.arc.95@gmail.com',
    description=('Deep Reinforcement Learning for Workload Management in '
                 'Heterogeneous Clusters'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/RaMdsC/hdeeprm',
    license='MIT',
    packages=setuptools.find_packages(),
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
