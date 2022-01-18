from setuptools import setup, find_packages

setup(
    name="GeoDySys",
    version="1.0",
    install_requires=[
        "numpy>=1.19.5",
        "scipy>=1.6.0",
        "scipy",
        "sklearn",
        "matplotlib",
        "matplotlib>=3.3.3",
        "tqdm>=4.56.0",
    ],
    packages=find_packages(),
)
