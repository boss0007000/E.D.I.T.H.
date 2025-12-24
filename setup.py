#!/usr/bin/env python3
"""
Setup script for Vehicle Identification System
"""
from setuptools import setup, find_packages
import os

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='vehicle-identification',
    version='1.0.0',
    author='E.D.I.T.H. Team',
    description='Complete vehicle identification system using multi-stage deep learning pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/boss0007000/E.D.I.T.H.',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'vehicle-identify=example:main',
        ],
    },
    include_package_data=True,
    package_data={
        'vehicle_identification': [
            'configs/*.yaml',
        ],
    },
    keywords='vehicle identification, computer vision, deep learning, PyTorch, YOLO, OCR',
    project_urls={
        'Bug Reports': 'https://github.com/boss0007000/E.D.I.T.H./issues',
        'Source': 'https://github.com/boss0007000/E.D.I.T.H.',
    },
)
