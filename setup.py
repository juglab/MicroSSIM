from setuptools import find_packages, setup

setup(
    name="microssim",
    version="0.1.0",
    packages=find_packages(include=["ri-ssim", "ri-ssim.*"]),
    install_requires=[
        "numpy",
        "scipy",
    ],
)
