import pytest
from SFC_Torch.Fmodel import SFcalculator
import numpy as np

@pytest.mark.parametrize("case", [1, 2])
def test_constructor_SFcalculator(data_pdb, data_mtz_exp, case):
    if case == 1:
        sfcalculator = SFcalculator(data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True)
        assert len(sfcalculator.HKL_array) == 22230
        assert len(sfcalculator.Hasu_array) == 22303
    else:
        sfcalculator = SFcalculator(data_pdb, mtzfile_dir=None, dmin=1.5, set_experiment=True) 
        assert sfcalculator.HKL_array is None
        assert len(sfcalculator.Hasu_array) == 10239

def test_inspect_data(data_pdb, data_mtz_exp):
    sfcalculator = SFcalculator(data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True)
    sfcalculator.inspect_data()
    assert sfcalculator.inspected
    assert np.isclose(sfcalculator.solventpct.cpu().numpy(), 0.1111, 1e-3)
    assert sfcalculator.gridsize == [80,120,144]





