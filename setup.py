from setuptools import setup, find_packages

setup(
    name="GeoDySys",
    version="1.0",
    install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib",
        "networkx",
	"torch",
        'cknn @ git+https://github.com/chlorochrule/cknn',
        "tensorboardX",
        "pyyaml",
    ],
    packages=find_packages(),
)
