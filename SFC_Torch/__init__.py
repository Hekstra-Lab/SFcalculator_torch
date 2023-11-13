def getVersionNumber():
    import pkg_resources
    version = pkg_resources.require("SFcalculator_torch")[0].version
    return version

__author__ = "Minhuan Li"
__email__ = "minhuanli@g.harvard.edu"
__version__ = getVersionNumber()

# Top level API
from .Fmodel import SFcalculator
from .io import PDBParser, fetch_pdb

# Suboodules
from . import utils
from . import patterson


