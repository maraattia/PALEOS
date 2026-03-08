from setuptools import setup, find_packages

setup(
    name="paleos",
    version="1.0.0",
    description="Planetary Assemblage Layers: Equations Of State",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=2.4",
        "scipy>=1.16",
        "sympy>=1.14",
        "RTpress @ git+https://gitlab.com/aswolf/rtpress.git@master",
    ],
)