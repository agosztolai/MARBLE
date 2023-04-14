"""Setup MARBL."""
import numpy
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

setup(
    name="MARBLE",
    version="1.0",
    install_requires=[
        "teaspoon==1.3.1",
        "matplotlib",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "networkx",
        "torch",
        "pympl",
        "tensorboardX",
        "pyyaml",
        "POT",
        "pyEDM",
        "teaspoon",
        "umap-learn",
        "mat73",
        "wget",
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={"MARBLE.lib": ["ptu_dijkstra.pyx", "ptu_dijkstra.c"]},
    ext_modules=cythonize(
        Extension(
            "ptu_dijkstra", ["MARBLE/lib/ptu_dijkstra.pyx"], include_dirs=[numpy.get_include()]
        )
    ),
)
