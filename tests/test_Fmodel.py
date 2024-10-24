import pytest

from os.path import exists
import tempfile

import numpy as np
import reciprocalspaceship as rs
import torch

from scipy.stats import pearsonr
from SFC_Torch.io import PDBParser
from SFC_Torch.Fmodel import SFcalculator
from SFC_Torch.utils import assert_numpy, assert_tensor

@pytest.mark.parametrize("case", [1, 2, 3])
def test_constructor_SFcalculator(data_pdb, data_mtz_exp, data_cif, data_sfcif, case):
    if case == 1:
        sfcalculator = SFcalculator(
            data_pdb, mtzdata=data_mtz_exp, set_experiment=True)
        sfcalculator.inspect_data()
        bins_labels = sfcalculator.assign_resolution_bins(return_labels=True)
        assert sfcalculator.inspected
        assert np.isclose(assert_numpy(sfcalculator.solventpct), 0.1667, 1e-3)
        assert sfcalculator.gridsize == [48, 60, 60]
        assert len(sfcalculator.HKL_array) == 3197
        assert len(sfcalculator.Hasu_array) == 3255
        assert len(sfcalculator.bins) == 3197
        assert np.all(np.sort(np.unique(sfcalculator.bins)) == np.arange(0,10))
        assert len(bins_labels) == 10
    elif case == 2:
        sfcalculator = SFcalculator(
            data_cif, mtzdata=data_sfcif, set_experiment=True)
        sfcalculator.inspect_data()
        bins_labels = sfcalculator.assign_resolution_bins(return_labels=True)
        assert sfcalculator.inspected
        assert np.isclose(assert_numpy(sfcalculator.solventpct), 0.1667, 1e-3)
        assert sfcalculator.gridsize == [50, 64, 64]
        assert len(sfcalculator.HKL_array) == 3256
        assert len(sfcalculator.Hasu_array) == 4035
        assert len(sfcalculator.bins) == 3256
        assert np.all(np.sort(np.unique(sfcalculator.bins)) == np.arange(0,10))
        assert len(bins_labels) == 10
    elif case == 3:
        sfcalculator = SFcalculator(
            data_pdb, mtzdata=None, dmin=2.5, set_experiment=True)
        sfcalculator.inspect_data()
        assert sfcalculator.inspected
        assert np.isclose(assert_numpy(sfcalculator.solventpct), 0.1667, 1e-3)
        assert sfcalculator.gridsize == [40, 48, 48]
        assert sfcalculator.HKL_array is None
        assert len(sfcalculator.Hasu_array) == 1747
    assert len(sfcalculator.atom_name) == 488


def test_constructor_SFcalculator_obj(data_pdb, data_mtz_exp):
    pdbmodel = PDBParser(data_pdb)
    mtzdata = rs.read_mtz(data_mtz_exp)
    sfcalculator = SFcalculator(
        pdbmodel, mtzdata=mtzdata, set_experiment=True)
    sfcalculator.inspect_data()
    bins_labels = sfcalculator.assign_resolution_bins(return_labels=True)
    assert sfcalculator.inspected
    assert np.isclose(assert_numpy(sfcalculator.solventpct), 0.1667, 1e-3)
    assert sfcalculator.gridsize == [48, 60, 60]
    assert len(sfcalculator.HKL_array) == 3197
    assert len(sfcalculator.Hasu_array) == 3255
    assert len(sfcalculator.bins) == 3197
    assert np.all(np.sort(np.unique(sfcalculator.bins)) == np.arange(0,10))
    assert len(bins_labels) == 10

@pytest.mark.parametrize("Return", [True, False])
@pytest.mark.parametrize("Anomalous", [True, False])
def test_calc_fall(data_pdb, data_mtz_exp, data_mtz_fmodel_ksol0, data_mtz_fmodel_ksol1, Return, Anomalous):
    sfcalculator = SFcalculator(
        data_pdb, mtzdata=data_mtz_exp, set_experiment=True, anomalous=Anomalous)
    sfcalculator.inspect_data()
    Fprotein = sfcalculator.calc_fprotein(Return=Return)
    Fsolvent = sfcalculator.calc_fsolvent(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=Return)
    sfcalculator.get_scales_lbfgs(ls_steps=2, r_steps=2, ls_lr=0.00001, r_lr=0.00001, verbose=False)
    Ftotal = sfcalculator.calc_ftotal()
    assert len(Ftotal) == 3197
    assert assert_numpy(sfcalculator.r_free) < 0.15
    sfcalculator.get_scales_adam(lr=0.01, n_steps=20, sub_ratio=0.3, initialize=True, verbose=False)
    assert assert_numpy(sfcalculator.r_free) < 0.15

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
        Fsolvent_arr = assert_numpy(Fsolvent)
        Fprotein_arr = assert_numpy(Fprotein)
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
        Fprotein_arr = assert_numpy(sfcalculator.Fprotein_HKL)
        Fsolvent_arr = assert_numpy(sfcalculator.Fmask_HKL)
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

def test_calc_fall_setter(data_pdb, data_mtz_exp, Return=True):
    sfcalculator = SFcalculator(
        data_pdb, mtzdata=data_mtz_exp, set_experiment=True)
    pdbmodel = PDBParser(data_pdb)
    sfcalculator.inspect_data()
    Fprotein = sfcalculator.calc_fprotein(
        atoms_position_tensor=assert_tensor(pdbmodel.atom_pos, arr_type=torch.float32),
        atoms_biso_tensor=assert_tensor(pdbmodel.atom_b_iso, arr_type=torch.float32),
        atoms_aniso_uw_tensor=assert_tensor(pdbmodel.atom_b_aniso, arr_type=torch.float32),
        atoms_occ_tensor=assert_tensor(pdbmodel.atom_occ, arr_type=torch.float32),
        Return=Return
    )
    Fsolvent = sfcalculator.calc_fsolvent(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=Return
    )
    sfcalculator.get_scales_lbfgs(ls_steps=2, r_steps=2, ls_lr=0.00001, r_lr=0.00001, verbose=False)
    Ftotal = sfcalculator.calc_ftotal()
    assert len(Ftotal) == 3197
    assert assert_numpy(sfcalculator.r_free) < 0.15
    sfcalculator.get_scales_adam(lr=0.01, n_steps=20, sub_ratio=0.3, initialize=True, verbose=False)
    assert assert_numpy(sfcalculator.r_free) < 0.15

def test_calc_ftotal_nodata(data_pdb):
    sfcalculator = SFcalculator(
        data_pdb, mtzdata=None, dmin=2.5, set_experiment=False)
    sfcalculator.inspect_data()
    sfcalculator.calc_fprotein(Return=False)
    sfcalculator.calc_fsolvent(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=False)
    sfcalculator.init_scales(requires_grad=False)
    Ftotal = sfcalculator.calc_ftotal()
    assert len(Ftotal) == 1747


@pytest.mark.parametrize("partition_size", [1, 3, 5])
@pytest.mark.parametrize("Anomalous", [True, False])
def test_calc_fall_batch(data_pdb, data_mtz_exp, Anomalous, partition_size):
    sfcalculator = SFcalculator(
        data_pdb, mtzdata=data_mtz_exp, set_experiment=True, anomalous=Anomalous)
    sfcalculator.inspect_data()
    atoms_pos_batch = torch.tile(sfcalculator.atom_pos_orth, [5, 1, 1])

    Fprotein = sfcalculator.calc_fprotein(Return=True)
    Fsolvent = sfcalculator.calc_fsolvent(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=True)
    Fprotein_batch = sfcalculator.calc_fprotein_batch(
        atoms_pos_batch, Return=True, PARTITION=partition_size)
    Fsolvent_batch = sfcalculator.calc_fsolvent_batch(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=True, PARTITION=partition_size)
    
    sfcalculator.init_scales(requires_grad=False)
    Ftotal = sfcalculator.calc_ftotal()
    Ftotal_batch = sfcalculator.calc_ftotal_batch()

    assert len(Fprotein_batch) == 5
    assert np.all(np.isclose(assert_numpy(Fprotein_batch[3]),
                             assert_numpy(Fprotein),
                             rtol=1e-5, atol=5e-3))
    assert np.all(np.isclose(assert_numpy(sfcalculator.Fprotein_asu_batch[4]),
                             assert_numpy(sfcalculator.Fprotein_asu),
                             rtol=1e-5, atol=5e-3))

    assert len(Fsolvent_batch) == 5
    assert np.all(np.isclose(assert_numpy(torch.abs(Fsolvent)),
                             assert_numpy(torch.abs(Fsolvent_batch[1])),
                             rtol=1e-3, atol=1e-2))

    assert np.all(np.isclose(assert_numpy(torch.abs(Ftotal_batch[2])),
                             assert_numpy(torch.abs(Ftotal)),
                             rtol=1e-3, atol=1e1))


def test_prepare_dataset(data_pdb, data_mtz_exp):
    sfcalculator = SFcalculator(
        data_pdb, mtzdata=data_mtz_exp, set_experiment=True)
    sfcalculator.inspect_data()
    sfcalculator.calc_fprotein(Return=False)
    sfcalculator.calc_fsolvent(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=False)
    sfcalculator.init_scales(requires_grad=False)
    sfcalculator.calc_ftotal()
    ds = sfcalculator.prepare_dataset("HKL_array", "Ftotal_HKL")
    assert len(ds) == 3197

def test_savePDB(data_pdb, data_mtz_exp):
    sfcalculator = SFcalculator(
        data_pdb, mtzdata=data_mtz_exp, set_experiment=True)
    sfcalculator.inspect_data()
    sfcalculator.calc_fprotein(Return=False)
    sfcalculator.calc_fsolvent(
        dmin_mask=6.0, dmin_nonzero=3.0, Return=False)
    sfcalculator.init_scales(requires_grad=False)
    sfcalculator.calc_ftotal()
    with tempfile.NamedTemporaryFile(suffix=".pdb") as temp:
        sfcalculator.savePDB(temp.name)
        assert exists(temp.name)
    
