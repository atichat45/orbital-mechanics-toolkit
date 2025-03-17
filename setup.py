#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="orbital-mechanics-toolkit",
    version="0.1.0",
    description="A comprehensive toolkit for orbital mechanics calculations and visualization",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "spiceypy>=5.0.0",
        "tqdm>=4.60.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
) 