import torch
import numpy as np
from .voxel import voxelvalue_torch_p1


def packingscore_voxelgrid_torch(
    atom_pos_orth,
    unit_cell,
    space_group,
    vdw_rad,
    unitcell_grid_center_orth_tensor,
    CUTOFF=0.0001,
):
    """
    Calculate the grid occupancy and clash percentage as packing score

    Parameters
    ----------
    atom_pos_orth: torch.tensor, [N_atom, 3]
        asu model in cartesian coordinates, better remove all hydrogen atoms at the beginning

    unit_cell: gemmi.UnitCell

    space_group: gemmi.SpaceGroup

    vdw_rad: torch.tensor, [N_atom, ]
        vdw radius of atoms, use vdw_rad_tensor to calculate

    unitcell_grid_center_orth_tensor: torch.float32 tensor, [N_grid, 3]
        center positions of all grids in carteisian space, use unitcell_grid_center to calculate

    CUTOFF: float, default 0.0001, must < 0.5
        cutoff to convert into binary map; Larger cutoff means slower decay further

    Returns
    -------
    Percentage of the occupancy of all unitcell grids, and percentage
    of clash grids between all symmetrical pairs.
    """
    N_grid = len(unitcell_grid_center_orth_tensor[:, 0])
    spacing = torch.max(
        unitcell_grid_center_orth_tensor[1] - unitcell_grid_center_orth_tensor[0]
    )
    # s ~ log(1/c -1) / (d/2 - r)
    spacing = torch.maximum(spacing, torch.tensor(3.5))
    steepness = np.log(1.0 / CUTOFF - 1.0) / (spacing / 2.0 - 1.5)

    voxel_map_p1 = voxelvalue_torch_p1(
        unitcell_grid_center_orth_tensor,
        atom_pos_orth,
        unit_cell,
        space_group,
        vdw_rad,
        s=steepness,
        binary=True,
        cutoff=CUTOFF,
    )

    occupancy = torch.count_nonzero(voxel_map_p1) / N_grid
    clash = torch.count_nonzero(voxel_map_p1 > 1) / N_grid

    return occupancy, clash
