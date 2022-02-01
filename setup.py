

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Polare",
    version="0.1.0",
    author="Jai Willems",
    author_email="jai52h@hotmail.com",
    description="Kernels for continuous data transformations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JaiWillems/polare",
    license="BSD-3-Clause",
    packages=setuptools.find_packages(include=["polare"]),
    install_requires=[
        "numpy",
        "scipy"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ]
)