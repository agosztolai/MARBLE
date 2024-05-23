"""Setup MARBLE."""
import numpy
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

setup(
    name="MARBLE",
    version="1.0",
    author="Adam Gosztolai",
    author_email="a.gosztolai@gmail.com",
    description="""Package for the unsupervised data-driven representation of non-linear dynamics
    over manifolds based on a statistical distribution of local flow fields""",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "matplotlib",
        "pandas",
        "numpy",
        "scipy",
        "networkx",
        "seaborn",
        "torch",
        "pympl",
        "tensorboardX",
        "pyyaml",
        "POT",
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
