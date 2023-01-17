from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("SFC_Torch/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version

__version__ = getVersionNumber()

setup(name="SFcalculator_torch",
    version=__version__,
    author="Minhaun Li",
    description="A Differentiable pipeline connecting molecule models and crystallpgraphy data", 
    url=" ",
    author_email='minhuanli@g.harvard.edu',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "pytorch>=1.13.0",
        "gemmi>=0.5.6",
        "reciprocalspaceship>=0.9.18",
    ],
)