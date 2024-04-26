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
    description="""Package for the data-driven representation of non-linear dynamics
    over manifolds based on a statistical distribution of local phase portrait features.
    Includes specific example on dynamical systems, synthetic- and real neural datasets.""",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "POT==0.9.3",
        "umap-learn==0.5.6",
        "wget==3.2",
        "torch_geometric==2.1.0",
        #"torch-scatter==2.1.2",
        #"torch-cluster==1.6.3",
        #"torch_sparse==0.6.18",
        "threadpoolctl==3.1.0"
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
