from setuptools import setup, find_packages

setup(
    name="ri-ssim",
    version="0.1.0",
    packages=find_packages(include=["ri-ssim", "ri-ssim.*"]),
    install_requires=[
        "numpy",
        "scipy",
    ],
)
