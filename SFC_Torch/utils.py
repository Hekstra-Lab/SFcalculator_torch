import torch
import numpy as np
import gemmi

__all__ = [
    "r_factor",
    "diff_array",
    "try_gpu",
    "unitcell_grid_center",
    "vdw_distance_matrix",
    "vdw_rad_tensor",
    "nonH_index",
    "assert_numpy",
    "assert_tensor",
    "bin_by_logarithmic",
    "aniso_scaling",
]

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def try_all_gpus():
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]


def is_list_or_tuple(x):
    return isinstance(x, list) or isinstance(x, tuple)


def assert_numpy(x, arr_type=None):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
    if is_list_or_tuple(x):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    if arr_type is not None:
        x = x.astype(arr_type)
    return x


def assert_tensor(x, arr_type=None, device=try_gpu()):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device)
    if is_list_or_tuple(x):
        x = np.array(x)
        x = torch.tensor(x, device=device)
    assert isinstance(x, torch.Tensor)
    if arr_type is not None:
        x = x.to(arr_type)
    return x


def r_factor(Fo, Fmodel, free_flag):
    """
    A function to calculate R_work and R_free

    Parameters
    ----------
    Fo: torch.tensor, [N_hkl,], real
        1D tensor containing Fo magnitude corresponding to HKL list
    Fmodel: torch.tensor, [N_hkl,], real
        1D tensor containing Fmodel magnitude corresponding to HKL list
    free_flag: np.array, [N_hkl,], binary
        1D array, whether this index should be treated as test set

    Returns
    -------
    R_work, R_free: Both are floats
    """
    R_work = torch.sum(torch.abs(Fo[~free_flag] - Fmodel[~free_flag])) / torch.sum(
        Fo[~free_flag]
    )
    R_free = torch.sum(torch.abs(Fo[free_flag] - Fmodel[free_flag])) / torch.sum(
        Fo[free_flag]
    )
    return R_work, R_free


def diff_array(a, b):
    """
    Return the elements in a but not in b, when a and b are array-like object

    Parameters
    ----------
    a: array-like
       Can be N Dimensional

    b: array-like

    return_diff: boolean
       return the set difference or not

    Return
    ------
    Difference Elements
    """
    tuplelist_a = list(map(tuple, a))
    tuplelist_b = list(map(tuple, b))
    set_a = set(tuplelist_a)
    set_b = set(tuplelist_b)
    return set_a - set_b


def asu2HKL(Hasu_array, HKL_array):
    """
    A fast way to find indices convert array Hasu to array HKL
    when both Hasu and HKL are 2D arrays.
    HKL is the subset of Hasu.
    Involves two steps:
    1. an evil string coding along axis1, turn the 2D array into 1D
    2. fancy sortsearch on two 1D arrays
    """

    def tostr(array):
        string = ""
        for i in array:
            string += "_" + str(i)
        return np.asarray(string, dtype="<U20")

    HKL_array_str = np.apply_along_axis(tostr, axis=1, arr=HKL_array)
    Hasu_array_str = np.apply_along_axis(tostr, axis=1, arr=Hasu_array)
    xsorted = np.argsort(Hasu_array_str)
    ypos = np.searchsorted(Hasu_array_str[xsorted], HKL_array_str)
    indices = xsorted[ypos]
    return indices


def DWF_iso(b_iso, dr2_array):
    """
    Calculate Debye_Waller Factor with Isotropic B Factor
    DWF_iso = exp(-B_iso * dr^2/4), Rupp P640, dr is dhkl in reciprocal space

    Parameters:
    -----------
    b_iso: 1D tensor, float32, [N_atoms,]
        Isotropic B factor

    dr2_array: numpy 1D array or 1D tensor, [N_HKLs,]
        Reciprocal d*(hkl)^2 array, corresponding to the HKL_array

    Return:
    -------
    A 2D [N_atoms, N_HKLs] float32 tensor with DWF corresponding to different atoms and different HKLs
    """
    dr2_tensor = torch.tensor(dr2_array).to(b_iso)
    return torch.exp(-b_iso.view([-1, 1]) * dr2_tensor / 4.0).type(torch.float32)


def DWF_aniso(aniso_uw, orth2frac_tensor, HKL_tensor):
    """
    Calculate Debye_Waller Factor with anisotropic B Factor, Rupp P641
    DWF_aniso = exp[-2 * pi^2 * (h^T U^* h))]
    U^* = O^(-1) U_w O(-1).T

    Parameters:
    -----------
    Ustar: 3D tensor float32, [N_atoms, 3, 3]
        Anisotropic B factor Uw matrix
        [[[U11, U12, U13], [U12, U22, U23], [U13, U23, U33]]...], of diffferent particles

    HKL_array: tensor of HKL index, [N_HKLs,3]

    Return:
    -------
    A 2D [N_atoms, N_HKLs] float32 tensor with DWF corresponding to different atoms and different HKLs
    """
    Ustar = torch.einsum("xy,ayz,wz->axw", orth2frac_tensor, aniso_uw, orth2frac_tensor)
    log_arg = (
        -2.0 * np.pi**2 * torch.einsum("rx,axy,ry->ar", HKL_tensor, Ustar, HKL_tensor)
    )
    DWF_aniso_vec = torch.exp(log_arg)
    return DWF_aniso_vec.type(torch.float32)


def aniso_scaling(uaniso, reciprocal_cell_paras, HKL_array):
    """
    This is used for anisotropic scaling for the overall model
    kaniso = exp(-2 * pi**2 * h^T * Uaniso * h)
    Afonine, P. V., et al. Acta Crystallographica Section D (2013): 625-634.

    Parameters
    ----------
    uaniso : torch.tensor, [6,]
        6 unique elements in the anisotropic matrix, [U11,U22,U33,U12,U13,U23]

    reciprocal_cell_paras: list of float or tensor float, [6,]
        Necessary info of Reciprocal unit cell, [ar, br, cr, cos(alpha_r), cos(beta_r), cos(gamma_r)

    HKL_array : numpy.array, [N_HKLs, 3]
        array of HKL index

    Return
    ------
    torch.tensor [N_HKLs,]
    """
    HKL_tensor = torch.tensor(HKL_array, device=uaniso.device)
    U11, U22, U33, U12, U13, U23 = uaniso
    h, k, l = HKL_tensor.T
    ar, br, cr, cos_alphar, cos_betar, cos_gammar = reciprocal_cell_paras
    args = (
        U11 * h**2 * ar**2
        + U22 * k**2 * br**2
        + U33 * l**2 * cr**2
        + 2 * (h * k * U12 * ar * br * cos_gammar 
               + h * l * U13 * ar * cr * cos_betar 
               + k * l * U23 * br * cr * cos_alphar)
    )
    return torch.exp(-2.0 * np.pi**2 * args)


def vdw_rad_tensor(atom_name_list, device=try_gpu()):
    """
    Return the vdw radius tensor of the atom list
    """
    unique_atom = list(set(atom_name_list))
    vdw_rad_dict = {}
    for atom_type in unique_atom:
        element = gemmi.Element(atom_type)
        vdw_rad_dict[atom_type] = torch.tensor(element.vdw_r)
    vdw_rad_tensor = torch.tensor(
        [vdw_rad_dict[atom] for atom in atom_name_list], device=device
    ).type(torch.float32)
    return vdw_rad_tensor


def vdw_distance_matrix(atom_name_list):
    """
    Calculate the minimum distance between atoms by vdw radius
    Use as a criteria of atom clash

    Parameters
    ----------
    atom_name_list: array-like, [N_atom,]
        atom names in order, like ['C', 'N', 'C', ...]

    Returns
    -------
    A matrix with [N_atom, N_atom], value [i,j] means the minimum allowed
    distance between atom i and atom j
    """
    vdw_rad = vdw_rad_tensor(atom_name_list)
    vdw_min_dist = vdw_rad[None, :] + vdw_rad[:, None]
    return vdw_min_dist


def nonH_index(atom_name_list):
    """
    Return the index of non-Hydrogen atoms
    """
    index = np.argwhere(np.array(atom_name_list) != "H").reshape(-1)
    return index


def unitcell_grid_center(unitcell, spacing=4.5, frac=False, return_tensor=True, device=try_gpu()):
    """
    Create a grid in real space given a unitcell and spacing
    output the center positions of all grids

    Parameters
    ----------
    unitcell: gemmi.UnitCell
        A unitcell instance containing size and fractionalization/orthogonalization matrix

    spacing: float, default 4.5
        grid size

    frac: boolean, default False
        If True, positions are in fractional space; Otherwise in cartesian space

    return_tensor: boolean, default True
        If True, convert to tf.tensor and return

    Returns
    -------
    [N_grid, 3] array, containing center positions of all grids
    """
    a, b, c, _, _, _ = unitcell.parameters
    na = int(a / spacing)
    nb = int(b / spacing)
    nc = int(c / spacing)
    u_list = np.linspace(0, 1, na)
    v_list = np.linspace(0, 1, nb)
    w_list = np.linspace(0, 1, nc)
    unitcell_grid_center_frac = np.array(np.meshgrid(u_list, v_list, w_list)).T.reshape(
        -1, 3
    )
    if frac:
        result = unitcell_grid_center_frac
    else:
        result = np.dot(
            unitcell_grid_center_frac, np.array(unitcell.orthogonalization_matrix).T
        )

    if return_tensor:
        return torch.tensor(result, device=device).type(torch.float32)
    else:
        return result

def bin_by_logarithmic(data, bins=10, Nmin=100):
    """Bin data with the logarithmic algorithm
    According to Urzhumtsev, A., et al. Acta Crystallographica Section D: Biological Crystallography 65.12 (2009)

    Parameters
    ----------
    data : np.ndarray, list
        Data to bin with logarithmic algorithm, like dHKL
    bins : int, optional
        Number of bins, by default 10
    Nmin : int, optional
        Minimum number of reflections for the first two low-resolution ranges, by default 100

    Returns
    -------
    assignments : np.ndarray
        Bins to which data were assigned
    bin_edges : np.ndarray
        Values of bin boundaries (1D array with `bins + 1` entries)
    """
    from reciprocalspaceship.utils import assign_with_binedges

    data_sorted = np.sort(data)[::-1]
    dlow = data_sorted[0] + 0.001
    dhigh = data_sorted[-1] - 0.001
    d1 = data_sorted[Nmin]
    d2 = data_sorted[2 * Nmin + 10]
    lnd_list = np.linspace(np.log(d2), np.log(dhigh), bins - 1)
    bin_edges = np.concatenate([[dlow, d1], np.exp(lnd_list)])
    assignment = assign_with_binedges(data, bin_edges, right_inclusive=True)
    return assignment, bin_edges
