from setuptools import setup, find_packages, Extension
import scipy
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
        "torch_geometric",
        "torch_sparse",
        "torch_scatter",
        "torch_cluster",
        "pympl",
        "cknn @ git+https://github.com/chlorochrule/cknn",
        "DE_library @ git+https://github.com/agosztolai/DE_library",
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
    package_data={"MARBLE.lib": ["ptu_dijkstra.pyx"]},
    setup_requires=['torch'],
    ext_modules=cythonize(
        Extension(
            "ptu_dijkstra",
            ["./MARBLE/lib/ptu_dijkstra.pyx", "./MARBLE/lib/ptu_dijkstra.c"],
            include_dirs=[numpy.get_include(), scipy.get_include()],
        )
    ),
)
