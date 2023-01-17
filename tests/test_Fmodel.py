import pytest

import numpy as np
import reciprocalspaceship as rs

from scipy.stats import pearsonr
from SFC_Torch.Fmodel import SFcalculator


@pytest.mark.parametrize("case", [1, 2])
def test_constructor_SFcalculator(data_pdb, data_mtz_exp, case):
    if case == 1:
        sfcalculator = SFcalculator(
            data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True)
        assert len(sfcalculator.HKL_array) == 22230
        assert len(sfcalculator.Hasu_array) == 22303
    else:
        sfcalculator = SFcalculator(
            data_pdb, mtzfile_dir=None, dmin=1.5, set_experiment=True)
        assert sfcalculator.HKL_array is None
        assert len(sfcalculator.Hasu_array) == 10239
    assert len(sfcalculator.atom_name) == 1492


def test_inspect_data(data_pdb, data_mtz_exp):
    sfcalculator = SFcalculator(
        data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True)
    sfcalculator.inspect_data()
    assert sfcalculator.inspected
    assert np.isclose(sfcalculator.solventpct.cpu().numpy(), 0.1111, 1e-3)
    assert sfcalculator.gridsize == [80, 120, 144]


def test_inspect_data_nomtz(data_pdb):
    sfcalculator = SFcalculator(
        data_pdb, mtzfile_dir=None, dmin=1.5, set_experiment=True)
    sfcalculator.inspect_data()
    assert sfcalculator.inspected
    assert np.isclose(sfcalculator.solventpct.cpu().numpy(), 0.1111, 1e-3)
    assert sfcalculator.gridsize == [60, 90, 108]


@pytest.mark.parametrize("Return", [True, False])
def test_calc_Fprotein(data_pdb, data_mtz_exp, data_mtz_fmodel_ksol0, Return):
    sfcalculator = SFcalculator(
        data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True)
    sfcalculator.inspect_data()
    Fprotein = sfcalculator.Calc_Fprotein(Return=Return)
    Fcalc = rs.read_mtz(data_mtz_fmodel_ksol0)
    assert (Fcalc.get_hkls() == sfcalculator.HKL_array).all()
    
    if Return:
        Fprotein_arr = Fprotein.cpu().numpy()
        assert pearsonr(np.abs(Fprotein_arr), Fcalc['FMODEL'].to_numpy()).statistic > 0.99
    else:
        assert Fprotein is None
        Fprotein_arr = sfcalculator.Fprotein_HKL.cpu().numpy()
        assert pearsonr(np.abs(Fprotein_arr), Fcalc['FMODEL'].to_numpy()).statistic > 0.99

@pytest.mark.parametrize("Return", [True, False])
def test_calc_Fsolvent(data_pdb, data_mtz_exp, data_mtz_fmodel_ksol0, data_mtz_fmodel_ksol1, Return):
    sfcalculator = SFcalculator(
        data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True)
    sfcalculator.inspect_data()
    sfcalculator.Calc_Fprotein(Return=False)
    Fsolvent = sfcalculator.Calc_Fsolvent(dmin_mask=6.0, dmin_nonzero=3.0, Return=Return)
    
    Fcalc = rs.read_mtz(data_mtz_fmodel_ksol0)
    Fmodel=rs.read_mtz(data_mtz_fmodel_ksol1)
    assert (Fmodel.get_hkls() == sfcalculator.HKL_array).all()

    calc_mag = Fcalc['FMODEL'].to_numpy()
    calc_ph = np.deg2rad(Fcalc['PHIFMODEL'].to_numpy())
    Fcalc_complex = np.array([complex(mag*np.cos(ph), mag*np.sin(ph)) 
                                for mag, ph in zip(calc_mag,calc_ph)])
    
    model_mag = Fmodel['FMODEL'].to_numpy()
    model_ph = np.deg2rad(Fmodel['PHIFMODEL'].to_numpy())
    Fmodel_complex = np.array([complex(mag*np.cos(ph), mag*np.sin(ph)) 
                                for mag, ph in zip(model_mag,model_ph)])

    Fmask_complex = Fmodel_complex - Fcalc_complex

    if Return:
        Fsolvent_arr = Fsolvent.cpu().numpy()
        assert pearsonr(np.abs(Fsolvent_arr), np.abs(Fmask_complex)).statistic > 0.95
    else:
        assert Fsolvent is None
        Fsolvent_arr = sfcalculator.Fmask_HKL.cpu().numpy()
        assert pearsonr(np.abs(Fsolvent_arr), np.abs(Fmask_complex)).statistic > 0.95 