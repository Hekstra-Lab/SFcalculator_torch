import pytest
from os.path import dirname, abspath, join

@pytest.fixture
def data_pdb():
    datapath = "data/4xof.pdb"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def data_mtz_exp():
    datapath = "data/4xof.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def data_mtz_fmodel_ksol0():
    datapath = "data/4xof_vanilla.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def data_mtz_fmodel_ksol1():
    datapath = "data/4xof_ksol1.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename
