import pytest

import numpy as np
import reciprocalspaceship as rs
import torch

from SFC_Torch.Fmodel import SFcalculator
from SFC_Torch.utils import try_gpu, vdw_rad_tensor, unitcell_grid_center
from SFC_Torch.voxel import voxelvalue_torch_p1, voxelvalue_torch_p1_savememory
from SFC_Torch.voxel import voxel_1dto3d_np, voxel_1dto3d_torch


@pytest.mark.parametrize("binary", [True, False])
def test_voxelvalue_torch_p1_sm(data_pdb, data_mtz_exp, binary):
    sfcalculator = SFcalculator(
        data_pdb, mtzdata=data_mtz_exp, set_experiment=True)
    vdw_rad = vdw_rad_tensor(sfcalculator.atom_name)
    uc_grid_orth_tensor = unitcell_grid_center(sfcalculator.unit_cell,
                                               spacing=4.5, return_tensor=True)
    CUTOFF = 0.0001
    N_grid = len(uc_grid_orth_tensor[:, 0])
    spacing = torch.max(uc_grid_orth_tensor[1] - uc_grid_orth_tensor[0])
    # s ~ log(1/c -1) / (d/2 - r)
    spacing = torch.maximum(spacing, torch.tensor(3.5))
    steepness = np.log(1.0/CUTOFF - 1.0)/(spacing/2.0 - 1.5)

    voxel_map_p1 = voxelvalue_torch_p1(uc_grid_orth_tensor, sfcalculator.atom_pos_orth,
                                       sfcalculator.unit_cell, sfcalculator.space_group, vdw_rad,
                                       s=steepness, binary=binary, cutoff=CUTOFF)
    voxel_map_p1_sm = voxelvalue_torch_p1_savememory(uc_grid_orth_tensor, sfcalculator.atom_pos_orth,
                                                     sfcalculator.unit_cell, sfcalculator.space_group, vdw_rad,
                                                     s=steepness, binary=binary, cutoff=CUTOFF)

    if binary:
        assert np.all(voxel_map_p1.cpu().numpy() ==
                      voxel_map_p1_sm.cpu().numpy())
    else:
        assert np.all(np.isclose(voxel_map_p1.cpu().numpy(),
                      voxel_map_p1_sm.cpu().numpy()))


def test_voxel_1dto3d(data_pdb, data_mtz_exp):
    sfcalculator = SFcalculator(
        data_pdb, mtzdata=data_mtz_exp, set_experiment=True)
    vdw_rad = vdw_rad_tensor(sfcalculator.atom_name)
    uc_grid_orth_tensor = unitcell_grid_center(sfcalculator.unit_cell,
                                               spacing=4.5, return_tensor=True)
    CUTOFF = 0.0001
    N_grid = len(uc_grid_orth_tensor[:, 0])
    spacing = torch.max(uc_grid_orth_tensor[1] - uc_grid_orth_tensor[0])
    # s ~ log(1/c -1) / (d/2 - r)
    spacing = torch.maximum(spacing, torch.tensor(3.5))
    steepness = np.log(1.0/CUTOFF - 1.0)/(spacing/2.0 - 1.5)

    voxel_map_p1 = voxelvalue_torch_p1(uc_grid_orth_tensor, sfcalculator.atom_pos_orth,
                                       sfcalculator.unit_cell, sfcalculator.space_group, vdw_rad,
                                       s=steepness, binary=True, cutoff=CUTOFF)
    a, b, c, _, _, _ = sfcalculator.unit_cell.parameters
    na = int(a/4.5)
    nb = int(b/4.5)
    nc = int(c/4.5)
    map_1 = voxel_1dto3d_np(voxel_map_p1.cpu().numpy(), na, nb, nc)
    map_2 = voxel_1dto3d_torch(voxel_map_p1, na, nb, nc).cpu().numpy()
    assert np.all(map_1 == map_2)
