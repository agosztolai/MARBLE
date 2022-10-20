from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="GeoDySys",
    version="1.0",
    install_requires=[
        "matplotlib==3.5.2",
        "numpy==1.22.4",
        "pandas==1.3.2",
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib",
        "networkx",
	    "torch",
        'cknn @ git+https://github.com/chlorochrule/cknn',
        'DE_library @ git+https://github.com/agosztolai/DE_library',
        "tensorboardX",
        "pyyaml"
        "POT",
        "pyEDM",
        "teaspoon"
    ],
    packages=find_packages(),
    ext_modules=cythonize(
        Extension(
            "ptu_dijkstra",
            ["GeoDySys/lib/ptu_dijkstra.pyx"],
            include_dirs=[numpy.get_include()]
        )
    )
)
