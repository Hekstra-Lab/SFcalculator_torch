from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("SFC_Torch/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version

__version__ = getVersionNumber()

setup(name="sfcalculator_torch",
    version=__version__,
    author="Minhaun Li",
    license="MIT",
    description="A differentiable pipeline connecting molecule models and crystallpgraphy data", 
    url="https://github.com/Hekstra-Lab/SFcalculator",
    author_email='minhuanli@g.harvard.edu',
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "gemmi>=0.5.6, <=0.6.7",
        "reciprocalspaceship>=0.9.18, <=1.0.3",
        "numpy<2.0.0",
        "tqdm",
        "loguru",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov"]
    },
)