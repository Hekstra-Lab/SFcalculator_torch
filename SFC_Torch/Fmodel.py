'''
Calculate Structural Factor from an atomic model: F_model = k_total * (F_calc + k_mask * F_mask)

Note:
1. We use direct summation for the F_calc
2. We support anomalous scattering with cromer_liberman equation
3. We use a bulk solvent masking 

Written in PyTorch
'''

__author__ = "Minhuan Li"
__email__ = "minhuanli@g.harvard.edu"

import gemmi
import time
import numpy as np
import torch
import reciprocalspaceship as rs

from .symmetry import generate_reciprocal_asu, expand_to_p1
from .mask import reciprocal_grid, rsgrid2realmask, realmask2Fmask
from .utils import try_gpu, DWF_aniso, DWF_iso, diff_array, asu2HKL
from .utils import vdw_rad_tensor, unitcell_grid_center, bin_by_logarithmic
from .packingscore import packingscore_voxelgrid_torch
from .utils import r_factor, assert_numpy


class SFcalculator(object):
    '''
    A class to formalize the structural factor calculation.
    '''

    def __init__(self, PDBfile_dir,
                 mtzfile_dir=None,
                 dmin=None,
                 anomalous=False,
                 wavelength=None,
                 set_experiment=True,
                 nansubset=['FP', 'SIGFP'],
                 freeflag='FreeR_flag',
                 testset_value=0):
        '''
        Initialize with necessary reusable information, like spacegroup, unit cell info, HKL_list, et.c.

        Parameters:
        -----------
        model_dir: path, str
            path to the PDB model file, will use its unit cell info, space group info, atom name info,
            atom position info, atoms B-factor info and atoms occupancy info to initialize the instance.

        mtz_file_dir: path, str, default None
            path to the mtz_file_dir, will use the HKL list in the mtz instead, override dmin with an inference

        dmin: float, default None
            highest resolution in the map in Angstrom, to generate Miller indices in recirpocal ASU

        anomalous: Boolean, default False
            Whether or not to include anomalous scattering in the calculation

        wavelength: None or float
            The wavelength of scattering source in A

        set_experiment: Boolean, default True
            Whether or not to set Fo and SigF, r_free, r_work from the experimental mtz file. It only works when
            the mtzfile_dir is not None

        nansubset: list of str, default ['FP', 'SIGFP']
            list of column names to examine the nan values
        '''
        structure = gemmi.read_pdb(PDBfile_dir)  # gemmi.Structure object
        self.unit_cell = structure.cell  # gemmi.UnitCell object
        self.space_group = gemmi.SpaceGroup(
            structure.spacegroup_hm)  # gemmi.SpaceGroup object
        self.operations = self.space_group.operations()  # gemmi.GroupOps object
        self.wavelength = wavelength
        self.anomalous = anomalous

        if anomalous:
            # Try to get the wavelength from PDB remarks
            try:
                line_index = np.argwhere(
                    ["WAVELENGTH OR RANGE" in i for i in structure.raw_remarks])
                pdb_wavelength = eval(
                    structure.raw_remarks[line_index[0, 0]].split()[-1])
                if wavelength is not None:
                    assert np.isclose(pdb_wavelength, wavelength, atol=0.05)
                else:
                    self.wavelength = pdb_wavelength
            except:
                print(
                    "Can't find wavelength record in the PDB file, or it doesn't match your input wavelength!")

        self.R_G_tensor_stack = torch.tensor(np.array([
            np.array(sym_op.rot)/sym_op.DEN for sym_op in self.operations]), device=try_gpu()).type(torch.float32)
        self.T_G_tensor_stack = torch.tensor(np.array([
            np.array(sym_op.tran)/sym_op.DEN for sym_op in self.operations]), device=try_gpu()).type(torch.float32)

        self.reciprocal_cell = self.unit_cell.reciprocal()  # gemmi.UnitCell object
        # [ar, br, cr, cos(alpha_r), cos(beta_r), cos(gamma_r)]
        self.reciprocal_cell_paras = torch.tensor([self.reciprocal_cell.a,
                                                   self.reciprocal_cell.b,
                                                   self.reciprocal_cell.c,
                                                   np.cos(np.deg2rad(
                                                       self.reciprocal_cell.alpha)),
                                                   np.cos(np.deg2rad(
                                                       self.reciprocal_cell.beta)),
                                                   np.cos(np.deg2rad(
                                                       self.reciprocal_cell.gamma))
                                                   ], device=try_gpu()).type(torch.float32)

        # Generate ASU HKL array and Corresponding d*^2 array
        if mtzfile_dir:
            mtz_reference = rs.read_mtz(mtzfile_dir)
            try:
                mtz_reference.dropna(axis=0, subset=nansubset, inplace=True)
            except:
                raise ValueError(
                    f"{nansubset} columns not included in the mtz file!")
            if anomalous:
                # Try to get the wavelength from MTZ file
                try:
                    mtz_wavelength = mtz_reference.dataset(0).wavelength
                    assert mtz_wavelength > 0.05
                    if self.wavelength is not None:
                        assert np.isclose(
                            mtz_wavelength, self.wavelength, atol=0.05)
                    else:
                        self.wavelength = mtz_wavelength
                except:
                    print(
                        "Can't find wavelength record in the MTZ file, or it doesn't match with other sources")
            # HKL array from the reference mtz file, [N,3]
            self.HKL_array = mtz_reference.get_hkls()
            self.dHKL = self.unit_cell.calculate_d_array(
                self.HKL_array).astype("float32")
            self.dmin = self.dHKL.min()
            assert mtz_reference.cell == self.unit_cell, "Unit cell from mtz file does not match that in PDB file!"
            assert mtz_reference.spacegroup.hm == self.space_group.hm, "Space group from mtz file does not match that in PDB file!"  # type: ignore
            self.Hasu_array = generate_reciprocal_asu(
                self.unit_cell, self.space_group, self.dmin, anomalous=anomalous)
            assert diff_array(self.HKL_array, self.Hasu_array) == set(
            ), "HKL_array should be equal or subset of the Hasu_array!"
            self.asu2HKL_index = asu2HKL(self.Hasu_array, self.HKL_array)
            # d*^2 array according to the HKL list, [N]
            self.dr2asu_array = self.unit_cell.calculate_1_d2_array(
                self.Hasu_array)
            self.dr2HKL_array = self.unit_cell.calculate_1_d2_array(
                self.HKL_array)
            if set_experiment:
                self.set_experiment(mtz_reference, freeflag, testset_value)
        else:
            if not dmin:
                raise ValueError(
                    "high_resolution dmin OR a reference mtz file should be provided!")
            else:
                self.dmin = dmin
                self.Hasu_array = generate_reciprocal_asu(
                    self.unit_cell, self.space_group, self.dmin)
                self.dHasu = self.unit_cell.calculate_d_array(
                    self.Hasu_array).astype("float32")
                self.dr2asu_array = self.unit_cell.calculate_1_d2_array(
                    self.Hasu_array)
                self.HKL_array = None

        self.atom_name = []
        self.atom_pos_orth = []
        self.atom_pos_frac = []
        self.atom_b_aniso = []
        self.atom_b_iso = []
        self.atom_occ = []
        model = structure[0]  # gemmi.Model object
        for chain in model:
            for res in chain:
                for atom in res:
                    # A list of atom name like ['O','C','N','C', ...], [Nc]
                    self.atom_name.append(atom.element.name)
                    # A list of atom's Positions in orthogonal space, [Nc,3]
                    self.atom_pos_orth.append(atom.pos.tolist())
                    # A list of atom's Positions in fractional space, [Nc,3]
                    self.atom_pos_frac.append(
                        self.unit_cell.fractionalize(atom.pos).tolist())
                    # A list of anisotropic B Factor [[U11,U22,U33,U12,U13,U23],..], [Nc,6]
                    self.atom_b_aniso.append(atom.aniso.elements_pdb())
                    # A list of isotropic B Factor [B1,B2,...], [Nc]
                    self.atom_b_iso.append(atom.b_iso)
                    # A list of occupancy [P1,P2,....], [Nc]
                    self.atom_occ.append(atom.occ)

        self.atom_pos_orth = torch.tensor(
            self.atom_pos_orth, device=try_gpu()).type(torch.float32)
        self.atom_pos_frac = torch.tensor(
            self.atom_pos_frac, device=try_gpu()).type(torch.float32)
        self.atom_b_aniso = torch.tensor(
            self.atom_b_aniso, device=try_gpu()).type(torch.float32)
        self.atom_b_iso = torch.tensor(
            self.atom_b_iso, device=try_gpu()).type(torch.float32)
        self.atom_occ = torch.tensor(
            self.atom_occ, device=try_gpu()).type(torch.float32)
        self.n_atoms = len(self.atom_name)
        self.unique_atom = list(set(self.atom_name))

        self.orth2frac_tensor = torch.tensor(
            self.unit_cell.fractionalization_matrix.tolist(), device=try_gpu()).type(torch.float32)
        self.frac2orth_tensor = torch.tensor(
            self.unit_cell.orthogonalization_matrix.tolist(), device=try_gpu()).type(torch.float32)

        # A dictionary of atomic structural factor f0_sj of different atom types at different HKL Rupp's Book P280
        # f0_sj = [sum_{i=1}^4 {a_ij*exp(-b_ij* d*^2/4)} ] + c_j
        if anomalous:
            assert self.wavelength is not None, ValueError(
                "If you need anomalous scattering contribution, provide the wavelength info from input, pbd or mtz file!")

        self.full_atomic_sf_asu = {}
        for atom_type in self.unique_atom:
            element = gemmi.Element(atom_type)
            f0 = np.array([element.it92.calculate_sf(dr2/4.)
                          for dr2 in self.dr2asu_array])
            if anomalous:
                fp, fpp = gemmi.cromer_liberman(
                    z=element.atomic_number, energy=gemmi.hc/self.wavelength)
                self.full_atomic_sf_asu[atom_type] = f0 + fp + 1j*fpp
            else:
                self.full_atomic_sf_asu[atom_type] = f0

        if anomalous:
            self.fullsf_tensor = torch.tensor(np.array([
                self.full_atomic_sf_asu[atom] for atom in self.atom_name]), device=try_gpu()).type(torch.complex64)
        else:
            self.fullsf_tensor = torch.tensor(np.array([
                self.full_atomic_sf_asu[atom] for atom in self.atom_name]), device=try_gpu()).type(torch.float32)
        self.inspected = False

    def set_experiment(self, exp_mtz, freeflag='FreeR_flag', testset_value=0):
        '''
        Set experimental data in the refinement

        exp_mtz, rs.Dataset, mtzfile read by reciprocalspaceship
        '''
        try:
            self.Fo = torch.tensor(
                exp_mtz["FP"].to_numpy(), device=try_gpu()).type(torch.float32)
            self.SigF = torch.tensor(
                exp_mtz["SIGFP"].to_numpy(), device=try_gpu()).type(torch.float32)
        except:
            print("MTZ file doesn't contain 'FP' or 'SIGFP'! Check your data!")
        try:
            self.rfree_id = np.argwhere(
                exp_mtz[freeflag].values == testset_value).reshape(-1)
            self.rwork_id = np.argwhere(
                exp_mtz[freeflag].values != testset_value).reshape(-1)
        except:
            print("No Free Flag! Check your data!")

    def assign_resolution_bins(self, bins=10, Nmin=100, return_labels=True, format_str=".2f"):
        """Assign reflections in HKL_array to resolution bins with logarithmic algorithm.
        The labels will be stored in self.bins

        Urzhumtsev, A., et al. Acta Crystallographica Section D: Biological Crystallography 65.12 (2009)

        Parameters
        ----------
        bins : int, optional
            Number of bins, by default 10
        Nmin : int, optional
            Minimum number of reflections for the first two low-resolution ranges, by default 100
        return_labels : bool, optional
            Whether to return a list of labels corresponding to the edges
            of each resolution bin, by default True
        format_str : str, optional
            Format string for constructing bin labels, by default ".2f"

        Returns
        -------
        None or list of labels
        """
        assert hasattr(
            self, "dHKL"), "Must have resolution stored in dHKL attribute!"

        assignments, edges = bin_by_logarithmic(self.dHKL, bins, Nmin)
        self.bins = assignments
        if return_labels:
            labels = [
                f"{e1:{format_str}} - {e2:{format_str}}"
                for e1, e2 in zip(edges[:-1], edges[1:])
            ]
            return labels

    def inspect_data(self, verbose=True):
        '''
        Do an inspection of data, for hints about 
        1. solvent percentage for mask calculation
        2. suitable grid size 
        '''
        # solvent percentage
        vdw_rad = vdw_rad_tensor(self.atom_name)
        uc_grid_orth_tensor = unitcell_grid_center(self.unit_cell,
                                                   spacing=4.5,
                                                   return_tensor=True)
        occupancy, _ = packingscore_voxelgrid_torch(
            self.atom_pos_orth, self.unit_cell, self.space_group, vdw_rad, uc_grid_orth_tensor)
        self.solventpct = 1 - occupancy
        # grid size
        mtz = gemmi.Mtz(with_base=True)
        mtz.cell = self.unit_cell
        mtz.spacegroup = self.space_group
        if not self.HKL_array is None:
            mtz.set_data(self.HKL_array)
        else:
            mtz.set_data(self.Hasu_array)
        self.gridsize = mtz.get_size_for_hkl(sample_rate=3.0)
        if verbose:
            print("Solvent Percentage:", self.solventpct)
            print("Grid size:", self.gridsize)
        self.inspected = True

    def calc_fprotein(self, atoms_position_tensor=None,
                      atoms_biso_tensor=None,
                      atoms_baniso_tensor=None,
                      atoms_occ_tensor=None,
                      NO_Bfactor=False,
                      Return=False):
        '''
        Calculate the structural factor from a single atomic model, without solvent masking

        Parameters
        ----------
        atoms_positions_tensor: 2D [N_atoms, 3] tensor or default None
            Positions of atoms in the model, in unit of angstrom; If not given, the model stored in attribute `atom_pos_frac` will be used

        atoms_biso_tensor: 1D [N_atoms,] tensor or default None
            Isotropic B factors of each atoms in the model; If not given, the info stored in attribute `atoms_b_iso` will be used

        atoms_baniso_tensor: 2D [N_atoms, 6] tensor or default None
            Anisotropic B factors of each atoms in the model; If not given, the info stored in attribute `atoms_b_aniso` will be used

        atoms_occ_tensor: 1D [N_atoms,] tensor or default None
            Occupancy of each atoms in the model; If not given, the info stored in attribute `atom_occ` will be used

        NO_Bfactor: Boolean, default False
            If True, the calculation will not use Bfactor parameterization; Useful when we are parameterizing the ensemble with a true distribution

        Return: Boolean, default False
            If True, it will return the Fprotein as the function output; Or It will just be saved in the `Fprotein_asu` and `Fprotein_HKL` attributes

        Returns
        -------
        None (Return=False) or Fprotein (Return=True)
        '''
        # Read and tensor-fy necessary inforamtion
        if not atoms_position_tensor is None:
            assert len(
                atoms_position_tensor) == self.n_atoms, "Atoms in atoms_positions_tensor should be consistent with atom names in PDB model!"
            self.atom_pos_frac = torch.tensordot(
                atoms_position_tensor, self.orth2frac_tensor.T, 1)

        if not atoms_baniso_tensor is None:
            assert len(atoms_baniso_tensor) == len(
                self.atom_name), "Atoms in atoms_baniso_tensor should be consistent with atom names in PDB model!"
            self.atom_b_aniso = atoms_baniso_tensor

        if not atoms_biso_tensor is None:
            assert len(atoms_biso_tensor) == len(
                self.atom_name), "Atoms in atoms_biso_tensor should be consistent with atom names in PDB model!"
            self.atom_b_iso = atoms_biso_tensor

        if not atoms_occ_tensor is None:
            assert len(atoms_occ_tensor) == len(
                self.atom_name), "Atoms in atoms_occ_tensor should be consistent with atom names in PDB model!"
            self.atom_occ = atoms_occ_tensor

        self.Fprotein_asu = F_protein(self.Hasu_array, self.dr2asu_array,
                                      self.fullsf_tensor,
                                      self.reciprocal_cell_paras,
                                      self.R_G_tensor_stack, self.T_G_tensor_stack,
                                      self.atom_pos_frac,
                                      self.atom_b_iso, self.atom_b_aniso, self.atom_occ,
                                      NO_Bfactor=NO_Bfactor)
        if not self.HKL_array is None:
            self.Fprotein_HKL = self.Fprotein_asu[self.asu2HKL_index]
            if Return:
                return self.Fprotein_HKL
        else:
            if Return:
                return self.Fprotein_asu

    def calc_fsolvent(self, solventpct=None, gridsize=None, dmin_mask=6.0, Return=False, dmin_nonzero=3.0):
        '''
        Calculate the structure factor of solvent mask in a differentiable way

        Parameters
        ----------
        solventpct: 0 - 1 Float, default None
            An approximate value of volume percentage of solvent in the unitcell. 
            run `inspect_data` before to use a suggested value

        gridsize: [Int, Int, Int], default None
            The size of grid to construct mask map.
            run `inspect_data` before to use a suggected value

        dmin_mask: np.float32, Default 6 angstroms.
            Minimum resolution cutoff, in angstroms, for creating the solvent mask

        Return: Boolean, default False
            If True, it will return the Fmask as the function output; Or It will just be saved in the `Fmask_asu` and `Fmask_HKL` attributes
        '''

        if solventpct is None:
            assert self.inspected, "Run inspect_data first or give a valid solvent percentage!"
            solventpct = self.solventpct

        if gridsize is None:
            assert self.inspected, "Run inspect_data first or give a valid grid size!"
            gridsize = self.gridsize

        # Shape [N_HKL_p1, 3], [N_HKL_p1,]
        Hp1_array, Fp1_tensor = expand_to_p1(
            self.space_group, self.Hasu_array, self.Fprotein_asu,
            dmin_mask=dmin_mask, unitcell=self.unit_cell, anomalous=self.anomalous)
        rs_grid = reciprocal_grid(Hp1_array, Fp1_tensor, gridsize)
        self.real_grid_mask = rsgrid2realmask(
            rs_grid, solvent_percent=solventpct)  # type: ignore
        if not self.HKL_array is None:
            self.Fmask_HKL = realmask2Fmask(
                self.real_grid_mask, self.HKL_array)
            zero_hkl_bool = torch.tensor(
                self.dHKL <= dmin_nonzero, device=try_gpu())
            self.Fmask_HKL[zero_hkl_bool] = torch.tensor(
                0., device=try_gpu(), dtype=torch.complex64)
            if Return:
                return self.Fmask_HKL
        else:
            self.Fmask_asu = realmask2Fmask(
                self.real_grid_mask, self.Hasu_array)
            zero_hkl_bool = torch.tensor(
                self.dHasu <= dmin_nonzero, device=try_gpu())
            self.Fmask_asu[zero_hkl_bool] = torch.tensor(
                0., device=try_gpu(), dtype=torch.complex64)
            if Return:
                return self.Fmask_asu

    def init_scales(self, requires_grad=True):
        self.kall = torch.tensor(
            1.0, device=try_gpu(), requires_grad=requires_grad)
        self.kaniso = torch.normal(
            0.01, 0.01, size=[6], device=try_gpu(), requires_grad=requires_grad)
        self.ksol = torch.tensor(
            0.35, device=try_gpu(), requires_grad=requires_grad)
        self.bsol = torch.tensor(
            50.0, device=try_gpu(), requires_grad=requires_grad)

    def set_scales(self, kall=None, kaniso=None, ksol=None, bsol=None):
        if kall is not None:
            self.kall = kall
        if kaniso is not None:
            self.kaniso = kaniso
        if ksol is not None:
            self.ksol = ksol
        if bsol is not None:
            self.bsol = bsol

    def get_scales(self, n_steps=50, lr=0.5, verbose=True, initialize=True, return_loss=False):
        '''
        Use LBFGS to optimize scales
        '''
        if initialize:
            self.init_scales(requires_grad=True)

        def closure():
            Fmodel = self.calc_ftotal()
            Fmodel_mag = torch.abs(Fmodel)
            loss = torch.sum(
                (self.Fo[self.rwork_id] - Fmodel_mag[self.rwork_id])**2)
            self.lbfgs.zero_grad()
            loss.backward()
            return loss

        params = [self.kall, self.kaniso, self.ksol, self.bsol]
        self.lbfgs = torch.optim.LBFGS(params, lr=lr)
        loss_track = []
        for _ in range(n_steps):
            start_time = time.time()
            loss = self.lbfgs.step(closure)
            Fmodel = self.calc_ftotal()
            Fmodel_mag = torch.abs(Fmodel)
            r_work, r_free = r_factor(
                self.Fo, Fmodel_mag, self.rwork_id, self.rfree_id)
            loss_track.append(
                [assert_numpy(loss), assert_numpy(r_work), assert_numpy(r_free)])
            str_ = f"Time: {time.time()-start_time:.3f}"
            if verbose:
                print(
                    f"Scale, {loss_track[-1][0]:.3f}, {loss_track[-1][1]:.3f}, {loss_track[-1][2]:.3f}", str_, flush=True)
        self.r_work, self.r_free = r_work, r_free
        if return_loss:
            return loss_track

    def calc_ftotal(self):
        if not self.HKL_array is None:
            dr2_tensor = torch.tensor(self.dr2HKL_array, device=try_gpu())
            scaled_Fmask = self.ksol * self.Fmask_HKL * \
                torch.exp(-self.bsol * dr2_tensor/4.0)
            self.Ftotal_HKL = self.kall * \
                DWF_aniso(self.kaniso[None, ...], self.reciprocal_cell_paras, self.HKL_array)[
                    0] * (self.Fprotein_HKL+scaled_Fmask)
            return self.Ftotal_HKL
        else:
            dr2_tensor = torch.tensor(self.dr2asu_array, device=try_gpu())
            scaled_Fmask = self.ksol * self.Fmask_asu * \
                torch.exp(-self.bsol * dr2_tensor/4.0)
            self.Ftotal_asu = self.kall * \
                DWF_aniso(self.kaniso[None, ...], self.reciprocal_cell_paras, self.Hasu_array)[
                    0] * (self.Fprotein_asu+scaled_Fmask)
            return self.Ftotal_asu

    def calc_fprotein_batch(self, atoms_position_batch, NO_Bfactor=False, Return=False, PARTITION=20):
        '''
        Calculate the Fprotein with batched models. Most parameters are similar to `Calc_Fprotein`

        atoms_positions_batch: torch.float32 tensor, [N_batch, N_atoms, 3]

        PARTITION: Int, default 20
            To reduce the memory cost during the computation, we divide the batch into several partitions and loops through them.
            Larger PARTITION will require larger GPU memory. Default 20 will take around 4GB, if N_atoms~1600 and N_HKLs~13000.
            But larger PARTITION will give a smaller wall time, so this is a trade-off.
        '''
        # Read and tensor-fy necessary information
        # TODO Test the following line with non-orthogonal unit cell, check if we need a transpose at the transform matrix
        atom_pos_frac_batch = torch.tensordot(
            atoms_position_batch, self.orth2frac_tensor.T, 1)  # [N_batch, N_atoms, N_dim=3]

        self.Fprotein_asu_batch = F_protein_batch(self.Hasu_array, self.dr2asu_array,
                                                  self.fullsf_tensor,
                                                  self.reciprocal_cell_paras,
                                                  self.R_G_tensor_stack, self.T_G_tensor_stack,
                                                  atom_pos_frac_batch,
                                                  self.atom_b_iso, self.atom_b_aniso, self.atom_occ,
                                                  NO_Bfactor=NO_Bfactor,
                                                  PARTITION=PARTITION)  # [N_batch, N_Hasus]

        if not self.HKL_array is None:
            # type: ignore
            self.Fprotein_HKL_batch = self.Fprotein_asu_batch[:,
                                                              self.asu2HKL_index]
            if Return:
                return self.Fprotein_HKL_batch
        else:
            if Return:
                return self.Fprotein_asu_batch

    def calc_fsolvent_batch(self, solventpct=None, gridsize=None, dmin_mask=6, Return=False, PARTITION=100, dmin_nonzero=3.0):
        '''
        Should run after Calc_Fprotein_batch, calculate the solvent mask structure factors in batched manner
        most parameters are similar to `Calc_Fmask`

        PARTITION: Int, default 100
            To reduce the memory cost during the computation, we divide the batch into several partitions and loops through them.
            Larger PARTITION will require larger GPU memory. Default 100 will take around 15GB, if gridsize=[160,160,160].
            But larger PARTITION will give a smaller wall time, so this is a trade-off. 
        '''

        if solventpct is None:
            assert self.inspected, "Run inspect_data first or give a valid solvent percentage!"
            solventpct = self.solventpct

        if gridsize is None:
            assert self.inspected, "Run inspect_data first or give a valid grid size!"
            gridsize = self.gridsize

        Hp1_array, Fp1_tensor_batch = expand_to_p1(
            self.space_group, self.Hasu_array, self.Fprotein_asu_batch,
            dmin_mask=dmin_mask, Batch=True, unitcell=self.unit_cell)

        batchsize = self.Fprotein_asu_batch.shape[0]  # type: ignore
        N_partition = batchsize // PARTITION + 1
        Fmask_batch = 0.

        if not self.HKL_array is None:
            HKL_array = self.HKL_array
        else:
            HKL_array = self.Hasu_array

        for j in range(N_partition):
            if j*PARTITION >= batchsize:
                continue
            start = j*PARTITION
            end = min((j+1)*PARTITION, batchsize)
            # Shape [N_batch, *gridsize]
            rs_grid = reciprocal_grid(
                Hp1_array, Fp1_tensor_batch[start:end], gridsize, end-start)
            real_grid_mask = rsgrid2realmask(
                rs_grid, solvent_percent=solventpct, Batch=True)  # type: ignore
            Fmask_batch_j = realmask2Fmask(
                real_grid_mask, HKL_array, end-start)
            if j == 0:
                Fmask_batch = Fmask_batch_j
            else:
                # Shape [N_batches, N_HKLs]
                Fmask_batch = torch.concat(
                    (Fmask_batch, Fmask_batch_j), dim=0)  # type: ignore

        if not self.HKL_array is None:
            zero_hkl_bool = torch.tensor(
                self.dHKL <= dmin_nonzero, device=try_gpu())
            Fmask_batch[:, zero_hkl_bool] = torch.tensor(
                0., device=try_gpu(), dtype=torch.complex64)  # type: ignore
            self.Fmask_HKL_batch = Fmask_batch
            if Return:
                return self.Fmask_HKL_batch
        else:
            zero_hkl_bool = torch.tensor(
                self.dHasu <= dmin_nonzero, device=try_gpu())
            Fmask_batch[:, zero_hkl_bool] = torch.tensor(
                0., device=try_gpu(), dtype=torch.complex64)  # type: ignore
            self.Fmask_asu_batch = Fmask_batch
            if Return:
                return self.Fmask_asu_batch

    def calc_ftotal_batch(self):
        if not self.HKL_array is None:
            dr2_tensor = torch.tensor(self.dr2HKL_array, device=try_gpu())
            scaled_Fmask = self.ksol * self.Fmask_HKL_batch * \
                torch.exp(-self.bsol * dr2_tensor/4.0)
            self.Ftotal_HKL_batch = self.kall * \
                DWF_aniso(self.kaniso[None, ...], self.reciprocal_cell_paras, self.HKL_array)[
                    0] * (self.Fprotein_HKL_batch+scaled_Fmask)
            return self.Ftotal_HKL_batch
        else:
            dr2_tensor = torch.tensor(self.dr2asu_array, device=try_gpu())
            scaled_Fmask = self.ksol * self.Fmask_asu_batch * \
                torch.exp(-self.bsol * dr2_tensor/4.0)
            self.Ftotal_asu_batch = self.kall * \
                DWF_aniso(self.kaniso[None, ...], self.reciprocal_cell_paras, self.Hasu_array)[
                    0] * (self.Fprotein_asu_batch+scaled_Fmask)
            return self.Ftotal_asu_batch

    def prepare_dataset(self, HKL_attr, F_attr):
        F_out = getattr(self, F_attr)
        HKL_out = getattr(self, HKL_attr)
        assert len(F_out) == len(
            HKL_out), "HKL and structural factor should have same length!"
        F_out_mag = torch.abs(F_out)
        PI_on_180 = 0.017453292519943295
        F_out_phase = torch.angle(F_out) / PI_on_180
        dataset = rs.DataSet(spacegroup=self.space_group,
                             cell=self.unit_cell)  # type: ignore
        dataset["H"] = HKL_out[:, 0]
        dataset["K"] = HKL_out[:, 1]
        dataset["L"] = HKL_out[:, 2]
        dataset["FMODEL"] = F_out_mag.cpu().numpy()
        dataset["PHIFMODEL"] = F_out_phase.cpu().numpy()
        dataset["FMODEL_COMPLEX"] = F_out.cpu().numpy()
        dataset.set_index(["H", "K", "L"], inplace=True)
        return dataset


def F_protein(HKL_array, dr2_array, fullsf_tensor, reciprocal_cell_paras,
              R_G_tensor_stack,
              T_G_tensor_stack,
              atom_pos_frac,
              atom_b_iso,
              atom_b_aniso,
              atom_occ,
              NO_Bfactor=False):
    '''
    Calculate Protein Structural Factor from an atomic model

    atom_pos_frac: 2D tensor, [N_atom, N_dim=3]
    '''
    # F_calc = sum_Gsum_j{ [f0_sj*DWF*exp(2*pi*i*(h,k,l)*(R_G*(x1,x2,x3)+T_G))]} fractional postion, Rupp's Book P279
    # G is symmetry operations of the spacegroup and j is the atoms
    # DWF is the Debye-Waller Factor, has isotropic and anisotropic version, based on the PDB file input, Rupp's Book P641
    HKL_tensor = torch.tensor(HKL_array, dtype=torch.float32, device=try_gpu())

    if NO_Bfactor:
        magnitude = fullsf_tensor * atom_occ[..., None]  # [N_atom, N_HKLs]
    else:
        # DWF calculator
        dwf_iso = DWF_iso(atom_b_iso, dr2_array)
        dwf_aniso = DWF_aniso(atom_b_aniso, reciprocal_cell_paras, HKL_array)
        # Some atoms do not have Anisotropic U
        mask_vec = torch.all(atom_b_aniso == torch.tensor(
            [0.]*6, device=try_gpu()), dim=-1)
        dwf_all = dwf_aniso
        dwf_all[mask_vec] = dwf_iso[mask_vec]

        # Apply Atomic Structure Factor and Occupancy for magnitude
        magnitude = dwf_all * fullsf_tensor * \
            atom_occ[..., None]  # [N_atoms, N_HKLs]

    # Vectorized phase calculation
    sym_oped_pos_frac = torch.permute(torch.tensordot(R_G_tensor_stack,
                                                      atom_pos_frac.T, 1), [2, 0, 1]) + T_G_tensor_stack  # Shape [N_atom, N_op, N_dim=3]
    exp_phase = 0.
    # Loop through symmetry operations instead of fully vectorization, to reduce the memory cost
    for i in range(sym_oped_pos_frac.size(dim=1)):
        phase_G = 2*np.pi * \
            torch.tensordot(HKL_tensor, sym_oped_pos_frac[:, i, :].T, 1)
        exp_phase = exp_phase + torch.exp(1j*phase_G)
    F_calc = torch.sum(exp_phase*magnitude.T, dim=-1)
    return F_calc


def F_protein_batch(HKL_array, dr2_array, fullsf_tensor, reciprocal_cell_paras,
                    R_G_tensor_stack,
                    T_G_tensor_stack,
                    atom_pos_frac_batch,
                    atom_b_iso,
                    atom_b_aniso,
                    atom_occ,
                    NO_Bfactor=False,
                    PARTITION=20):
    '''
    Calculate Protein Structural Factor from a batch of atomic models

    atom_pos_frac_batch: 3D tensor, [N_batch, N_atoms, N_dim=3]

    TODO: Support batched B factors
    '''
    # F_calc = sum_Gsum_j{ [f0_sj*DWF*exp(2*pi*i*(h,k,l)*(R_G*(x1,x2,x3)+T_G))]} fractional postion, Rupp's Book P279
    # G is symmetry operations of the spacegroup and j is the atoms
    # DWF is the Debye-Waller Factor, has isotropic and anisotropic version, based on the PDB file input, Rupp's Book P641
    HKL_tensor = torch.tensor(HKL_array, device=try_gpu()).type(torch.float32)
    batchsize = atom_pos_frac_batch.shape[0]

    if NO_Bfactor:
        magnitude = fullsf_tensor * atom_occ[..., None]  # [N_atom, N_HKLs]
    else:
        # DWF calculator
        dwf_iso = DWF_iso(atom_b_iso, dr2_array)
        dwf_aniso = DWF_aniso(atom_b_aniso, reciprocal_cell_paras, HKL_array)
        # Some atoms do not have Anisotropic U
        mask_vec = torch.all(atom_b_aniso == torch.tensor(
            [0.]*6, device=try_gpu()), dim=-1)
        dwf_all = dwf_aniso
        dwf_all[mask_vec] = dwf_iso[mask_vec]
        # Apply Atomic Structure Factor and Occupancy for magnitude
        magnitude = dwf_all * fullsf_tensor * \
            atom_occ[..., None]  # [N_atoms, N_HKLs]

    # Vectorized phase calculation
    sym_oped_pos_frac = torch.tensordot(atom_pos_frac_batch, torch.permute(R_G_tensor_stack, [
        2, 1, 0]), 1) + T_G_tensor_stack.T  # Shape [N_batch, N_atom, N_dim=3, N_ops]
    N_ops = R_G_tensor_stack.shape[0]
    N_partition = batchsize // PARTITION + 1
    F_calc = 0.
    for j in range(N_partition):
        Fcalc_j = 0.
        if j*PARTITION >= batchsize:
            continue
        start = j*PARTITION
        end = min((j+1)*PARTITION, batchsize)
        for i in range(N_ops):  # Loop through symmetry operations to reduce memory cost
            # Shape [PARTITION, N_atoms, N_HKLs]
            phase_ij = 2 * torch.pi * \
                torch.tensordot(
                    sym_oped_pos_frac[start:end, :, :, i], HKL_tensor.T, 1)
            exp_phase_ij = torch.exp(1j*phase_ij)
            # Shape [PARTITION, N_HKLs], sum over atoms
            Fcalc_ij = torch.sum(exp_phase_ij*magnitude, dim=1)
            # Shape [PARTITION, N_HKLs], sum over symmetry operations
            Fcalc_j = Fcalc_j + Fcalc_ij
        if j == 0:
            F_calc = Fcalc_j
        else:
            # Shape [N_batches, N_HKLs]
            F_calc = torch.concat((F_calc, Fcalc_j), dim=0)  # type: ignore
    return F_calc
