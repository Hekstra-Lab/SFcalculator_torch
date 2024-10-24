import pytest
from os.path import dirname, abspath, join

@pytest.fixture
def data_pdb():
    datapath = "data/1dur.pdb"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def data_cif():
    datapath = "data/1dur.cif"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def data_sfcif():
    datapath = "data/1dur-sf.cif"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def data_mtz_exp():
    datapath = "data/1dur.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def data_mtz_fmodel_ksol0():
    datapath = "data/1dur_vanilla.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def data_mtz_fmodel_ksol1():
    datapath = "data/1dur_ksol1.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename
