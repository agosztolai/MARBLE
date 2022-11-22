from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

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
        'cknn @ git+https://github.com/chlorochrule/cknn',
        'DE_library @ git+https://github.com/agosztolai/DE_library',
        "tensorboardX",
        "pyyaml",
        "POT",
        "pyEDM",
        "teaspoon",
        "umap-learn",
        "mat73",
        "wget"
    ],
    packages=find_packages(),
    ext_modules=cythonize(
        Extension(
            "ptu_dijkstra",
            ["MARBLE/lib/ptu_dijkstra.pyx"],
            include_dirs=[numpy.get_include()]
        )
    )
)
