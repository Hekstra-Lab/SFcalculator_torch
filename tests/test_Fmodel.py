import pytest

import numpy as np
import reciprocalspaceship as rs
import torch

from scipy.stats import pearsonr
from SFC_Torch.Fmodel import SFcalculator
from SFC_Torch.utils import try_gpu


@pytest.mark.parametrize("case", [1, 2])
def test_constructor_SFcalculator(data_pdb, data_mtz_exp, case):
    if case == 1:
        sfcalculator = SFcalculator(
            data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True)
        sfcalculator.inspect_data()
        assert sfcalculator.inspected
        assert np.isclose(sfcalculator.solventpct.cpu().numpy(), 0.1667, 1e-3)
        assert sfcalculator.gridsize == [48, 60, 60]
        assert len(sfcalculator.HKL_array) == 3197
        assert len(sfcalculator.Hasu_array) == 3255
    else:
        sfcalculator = SFcalculator(
            data_pdb, mtzfile_dir=None, dmin=2.5, set_experiment=True)
        sfcalculator.inspect_data()
        assert sfcalculator.inspected
        assert np.isclose(sfcalculator.solventpct.cpu().numpy(), 0.1667, 1e-3)
        assert sfcalculator.gridsize == [40, 48, 48]
        assert sfcalculator.HKL_array is None
        assert len(sfcalculator.Hasu_array) == 1747
    assert len(sfcalculator.atom_name) == 488


@pytest.mark.parametrize("Return", [True, False])
@pytest.mark.parametrize("Anomalous", [True, False])
def test_calc_Fall(data_pdb, data_mtz_exp, data_mtz_fmodel_ksol0, data_mtz_fmodel_ksol1, Return, Anomalous):
    sfcalculator = SFcalculator(
        data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True, anomalous=Anomalous)
    sfcalculator.inspect_data()
    Fprotein = sfcalculator.Calc_Fprotein(Return=Return)
    Fsolvent = sfcalculator.Calc_Fsolvent(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=Return)

    Ftotal = sfcalculator.Calc_Ftotal()
    assert len(Ftotal) == 3197

    Fcalc = rs.read_mtz(data_mtz_fmodel_ksol0)
    Fmodel = rs.read_mtz(data_mtz_fmodel_ksol1)
    assert (Fmodel.get_hkls() == sfcalculator.HKL_array).all()

    calc_mag = Fcalc['FMODEL'].to_numpy()
    calc_ph = np.deg2rad(Fcalc['PHIFMODEL'].to_numpy())
    Fcalc_complex = np.array([complex(mag*np.cos(ph), mag*np.sin(ph))
                              for mag, ph in zip(calc_mag, calc_ph)])
    model_mag = Fmodel['FMODEL'].to_numpy()
    model_ph = np.deg2rad(Fmodel['PHIFMODEL'].to_numpy())
    Fmodel_complex = np.array([complex(mag*np.cos(ph), mag*np.sin(ph))
                               for mag, ph in zip(model_mag, model_ph)])

    Fmask_complex = Fmodel_complex - Fcalc_complex

    if Return:
        Fsolvent_arr = Fsolvent.cpu().numpy()
        Fprotein_arr = Fprotein.cpu().numpy()
        try:
            assert pearsonr(np.abs(Fprotein_arr),
                            Fcalc['FMODEL'].to_numpy()).statistic > 0.99
            assert pearsonr(np.abs(Fsolvent_arr),
                            np.abs(Fmask_complex)).statistic > 0.84
        except:
            assert pearsonr(np.abs(Fprotein_arr),
                            Fcalc['FMODEL'].to_numpy())[0] > 0.99
            assert pearsonr(np.abs(Fsolvent_arr),
                            np.abs(Fmask_complex))[0] > 0.84
    else:
        assert Fsolvent is None
        assert Fprotein is None
        Fprotein_arr = sfcalculator.Fprotein_HKL.cpu().numpy()
        Fsolvent_arr = sfcalculator.Fmask_HKL.cpu().numpy()
        try:
            assert pearsonr(np.abs(Fprotein_arr),
                            Fcalc['FMODEL'].to_numpy()).statistic > 0.99
            assert pearsonr(np.abs(Fsolvent_arr),
                            np.abs(Fmask_complex)).statistic > 0.84
        except:
            assert pearsonr(np.abs(Fprotein_arr),
                            Fcalc['FMODEL'].to_numpy())[0] > 0.99
            assert pearsonr(np.abs(Fsolvent_arr),
                            np.abs(Fmask_complex))[0] > 0.84


def test_calc_Ftotal_nodata(data_pdb):
    sfcalculator = SFcalculator(
        data_pdb, mtzfile_dir=None, dmin=2.5, set_experiment=True)
    sfcalculator.inspect_data()
    sfcalculator.Calc_Fprotein(Return=False)
    sfcalculator.Calc_Fsolvent(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=False)
    Ftotal = sfcalculator.Calc_Ftotal()
    assert len(Ftotal) == 1747


@pytest.mark.parametrize("partition_size", [1, 3, 5])
@pytest.mark.parametrize("Anomalous", [True, False])
def test_calc_Fall_batch(data_pdb, data_mtz_exp, Anomalous, partition_size):
    sfcalculator = SFcalculator(
        data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True, anomalous=Anomalous)
    sfcalculator.inspect_data()
    atoms_pos_batch = torch.tile(sfcalculator.atom_pos_orth, [5, 1, 1])

    Fprotein = sfcalculator.Calc_Fprotein(Return=True)
    Fsolvent = sfcalculator.Calc_Fsolvent(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=True)
    Fprotein_batch = sfcalculator.Calc_Fprotein_batch(atoms_pos_batch, Return=True, PARTITION=partition_size)
    Fsolvent_batch = sfcalculator.Calc_Fsolvent_batch(
         dmin_mask=6.0, dmin_nonzero=3.0, Return=True, PARTITION=partition_size)

    kaniso = torch.tensor(
        [-1.2193, -0.5417, -0.6066,  0.8886,  1.1478, -1.6649], device=try_gpu())
    Ftotal = sfcalculator.Calc_Ftotal(kaniso=kaniso)
    Ftotal_batch = sfcalculator.Calc_Ftotal_batch(kaniso=kaniso)

    assert len(Fprotein_batch) == 5
    assert np.all(np.isclose(Fprotein_batch[3].cpu().numpy(),
                             Fprotein.cpu().numpy(),
                             rtol=1e-5, atol=5e-3))
    assert np.all(np.isclose(sfcalculator.Fprotein_asu_batch[4].cpu().numpy(),
                             sfcalculator.Fprotein_asu.cpu().numpy(),
                             rtol=1e-5, atol=5e-3))
    
    assert len(Fsolvent_batch) == 5
    assert np.all(np.isclose(torch.abs(Fsolvent).cpu().numpy(),
                             torch.abs(Fsolvent_batch[1]).cpu().numpy(),
                             rtol=1e-3, atol=1e-2)) 

    assert np.all(np.isclose(torch.abs(Ftotal_batch[2]).cpu().numpy(),
                             torch.abs(Ftotal).cpu().numpy(),
                             rtol=1e-3, atol=1e1))


def test_prepare_Dataset(data_pdb, data_mtz_exp):
    sfcalculator = SFcalculator(
        data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True)
    sfcalculator.inspect_data()

    sfcalculator.Calc_Fprotein(Return=False)
    sfcalculator.Calc_Fsolvent(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=False)
    sfcalculator.Calc_Ftotal()
    ds = sfcalculator.prepare_DataSet("HKL_array", "Ftotal_HKL")
    assert len(ds) == 3197
