
"""
Calculate Structural Factor from an atomic model: F_model = k_total * (F_calc + k_mask * F_mask)

Note:
1. We use direct summation for the F_calc
2. We support anomalous scattering with cromer_liberman equation
3. We use a bulk solvent masking 

Written in PyTorch
"""
from __future__ import annotations

from typing import Optional, List
import gemmi
import time
import numpy as np
import torch
import reciprocalspaceship as rs
from loguru import logger

from .symmetry import generate_reciprocal_asu, expand_to_p1
from .mask import reciprocal_grid, rsgrid2realmask, realmask2Fmask
from .utils import try_gpu, DWF_aniso, DWF_iso, diff_array, asu2HKL, aniso_scaling
from .utils import vdw_rad_tensor, unitcell_grid_center, bin_by_logarithmic
from .utils import r_factor, assert_numpy, assert_tensor
from .packingscore import packingscore_voxelgrid_torch
from .io import PDBParser

class SFcalculator(object):
    """
    A class to formalize the structural factor calculation.
    """

    def __init__(
        self,
        pdbmodel:       str | PDBParser,
        mtzdata:        str | rs.DataSet = None,
        n_bins:         int = 10, 
        dmin:           Optional[float] = None,
        anomalous:      bool = False,
        wavelength:     Optional[float] = None,
        set_experiment: bool = True,
        expcolumns:     List[str] = ["FP", "SIGFP"],
        freeflag:       str = "FreeR_flag",
        testset_value:  int = 0,
        device:         torch.device = try_gpu(),
        mode:           str = "xray",
    ) -> None:
        """
        Initialize with necessary reusable information, like spacegroup, unit cell info, HKL_list, et.c.

        Parameters:
        -----------
        pdbmodel: str | PDBParser
            path to the PDB file or an instance of PDBparser. Will use its unit cell info, space group info, atom name info,
            atom position info, atoms B-factor info and atoms occupancy info to initialize the instance.

        mtzdata: str | rs.Dataset 
            path to the mtz data or instance of rs.Dataset, will use the HKL list in the mtz instead, override dmin with an inference
            also support reflection data in cif file
    
        n_bins: str, default 10
            Number of resolution bins used in the reciprocal space
        
        dmin: float | None
            highest resolution in the map in Angstrom, to generate Miller indices in recirpocal ASU

        anomalous: bool, default False
            Whether or not to include anomalous scattering in the calculation

        wavelength: float | None
            The wavelength of scattering source in A

        set_experiment: bool, default True
            Whether or not to set Fo, SigF, free_flag and Outlier from the experimental mtz file. It only works when
            the mtzfile_dir is not None

        expcolumns: List[str], default ['FP', 'SIGFP']
            list of column names used as expeimental data

        freeflag: str, default "FreeR_flag"
            column names used as free flag, default "FreeR_flag". Also could be "FREE" and "R-free-flags" 
            in Phenix(CNS/XPOLAR) convention
        
        testset_value: int, default 0
            Test set values in the freeflag column

        device: torch.device

        mode: str, default xray
            xray or cryoem, use xray atomic factors or elelctron atomic factors
        """
        
        self.wavelength = wavelength
        self.anomalous = anomalous
        self.device = device
        assert mode in ["xray", "cryoem"], ValueError(
                "mode has to be xray or cryoem!"
            )
        self.mode = mode
        self.init_pdb(pdbmodel)
        if mtzdata is not None:
            self.init_mtz(mtzdata, n_bins, expcolumns, set_experiment, freeflag, testset_value, dmin)
        else:
            self.init_withoutmtz(dmin, n_bins)
        self._init_spacegroup()
        self._init_cell()
        self.init_atomic_scattering()
        self.inspected = False

    def init_pdb(self, pdbmodel: str | PDBParser):
        """
        set pdb topology, symmetry operations, unit_cell properties, and initialize model coordinates
        """
        if isinstance(pdbmodel, str):
            self._pdb = PDBParser(pdbmodel)  # sfc.PDBparser object
        elif isinstance(pdbmodel, PDBParser):
            self._pdb = pdbmodel
        else:
            raise TypeError("pdbmodel should be PDBparser instance or path str to a pdb file!")
        
        # set molecule related property
        # Tensor atom's Positions in orthogonal space, [Nc,3]
        self._atom_pos_orth = assert_tensor(self._pdb.atom_pos, device=self.device, arr_type=torch.float32)
        # Tensor of anisotropic B Factor in matrix form, [Nc,3,3]
        self._atom_aniso_uw = assert_tensor(self._pdb.atom_b_aniso, device=self.device, arr_type=torch.float32)
        # Tensor of isotropic B Factor [B1,B2,...], [Nc]
        self._atom_b_iso = assert_tensor(self._pdb.atom_b_iso, device=self.device, arr_type=torch.float32)
        # Tensor of occupancy [P1,P2,....], [Nc]
        self._atom_occ = assert_tensor(self._pdb.atom_occ, device=self.device, arr_type=torch.float32)

        if self.anomalous:
            # Try to get the wavelength from PDB remarks
            try:
                line_index = np.argwhere(
                    ["WAVELENGTH OR RANGE" in i for i in self._pdb.pdb_header]
                )
                pdb_wavelength = eval(
                    self._pdb.pdb_header[line_index[0, 0]].split()[-1]
                )
                if self.wavelength is not None:
                    assert np.isclose(pdb_wavelength, self.wavelength, atol=0.05)
                else:
                    self.wavelength = pdb_wavelength
            except:
                logger.warning(
                    "Can't find wavelength record in the PDB file, or it doesn't match your input wavelength!"
                )

    @property
    def space_group(self):
        return self._pdb.spacegroup
    
    @space_group.setter
    def space_group(self, spacegroup):
        self._pdb.set_spacegroup(spacegroup)
        self._init_spacegroup()
    
    @property
    def unit_cell(self):
        return self._pdb.cell
    
    @unit_cell.setter
    def unit_cell(self, cell):
        self._pdb.set_unitcell(cell)
        self._init_cell()

    def _init_spacegroup(self):
        # Set up spacegroup related property
        self.operations = self.space_group.operations()  # gemmi.GroupOps object
        self.R_G_tensor_stack = assert_tensor(
            np.array([np.array(sym_op.rot) / sym_op.DEN for sym_op in self.operations]),
            arr_type=torch.float32,
            device=self.device,
        )
        self.T_G_tensor_stack = assert_tensor(
            np.array(
                [np.array(sym_op.tran) / sym_op.DEN for sym_op in self.operations]
            ),
            arr_type=torch.float32,
            device=self.device,
        )
    
    def _init_cell(self):
        # Set up unit cell related property
        self.orth2frac_tensor = torch.tensor(
            self.unit_cell.fractionalization_matrix.tolist(), device=self.device
        ).type(torch.float32)
        self.frac2orth_tensor = torch.tensor(
            self.unit_cell.orthogonalization_matrix.tolist(), device=self.device
        ).type(torch.float32)
        self.reciprocal_cell = self.unit_cell.reciprocal()  # gemmi.UnitCell object
        # [ar, br, cr, cos(alpha_r), cos(beta_r), cos(gamma_r)]
        self.reciprocal_cell_paras = torch.tensor(
            [
                self.reciprocal_cell.a,
                self.reciprocal_cell.b,
                self.reciprocal_cell.c,
                np.cos(np.deg2rad(self.reciprocal_cell.alpha)),
                np.cos(np.deg2rad(self.reciprocal_cell.beta)),
                np.cos(np.deg2rad(self.reciprocal_cell.gamma)),
            ],
            device=self.device,
        ).type(torch.float32)
    
    @property
    def atom_pos_orth(self):
        return self._atom_pos_orth
    
    @atom_pos_orth.setter
    def atom_pos_orth(self, atoms_positions_tensor: torch.Tensor):
        assert (
            atoms_positions_tensor.shape == torch.Size([self.n_atoms, 3])
        ), f"atoms_postions_tensor should be in shape [{self.n_atoms}, 3]!"
        self._atom_pos_orth = atoms_positions_tensor
    
    @property
    def atom_b_iso(self):
        return self._atom_b_iso
    
    @atom_b_iso.setter
    def atom_b_iso(self, atoms_biso_tensor: torch.Tensor):
        assert atoms_biso_tensor.shape == torch.Size([self.n_atoms]), f"atoms_biso_tensor should be in shape [{self.n_atoms}]"
        self._atom_b_iso = atoms_biso_tensor
    
    @property
    def atom_aniso_uw(self):
        return self._atom_aniso_uw
    
    @atom_aniso_uw.setter
    def atom_aniso_uw(self, atoms_aniso_uw_tensor: torch.Tensor):
        assert (
            atoms_aniso_uw_tensor.shape == torch.Size([self.n_atoms, 3, 3]) 
        ), f"atoms_aniso_uw_tensor should be in shape [{self.n_atoms}, 3, 3]"
        self._atom_aniso_uw = atoms_aniso_uw_tensor

    @property
    def atom_occ(self):
        return self._atom_occ
    
    @atom_occ.setter
    def atom_occ(self, atoms_occ_tensor: torch.Tensor):
        assert atoms_occ_tensor.shape == torch.Size([self.n_atoms]), f"atoms_occ_tensor should be in shape [{self.n_atoms}]"
        self._atom_occ = atoms_occ_tensor        

    @property
    def atom_pos_frac(self):
        """
        Tensor of atom's Positions in fractional space, [Nc,3]
        """
        return self.orth2frac(self.atom_pos_orth)

    @property
    def cra_name(self):
        """
        A list of Chain-Residue-Atom name, ['A-0-ALA-CA', ...]
        """
        return self._pdb.cra_name
    
    @property
    def atom_name(self):
        """
        A list of element name, ['C', 'N', 'H', ...]
        """
        return self._pdb.atom_name
    
    @property
    def n_atoms(self):
        return len(self.atom_name)
    
    @property
    def unique_atom(self):
        return list(set(self.atom_name))
    
    def frac2orth(self, frac_pos: torch.Tensor) -> torch.Tensor:
        """
        Convert fractional coordinates to orthogonal coordinates

        Args:
            frac_pos: torch.Tensor, [n_points, ..., 3]
        
        Returns:
            orthogonal coordinates, torch.Tensor
        """
        orth_pos = torch.einsum("n...x,yx->n...y", frac_pos, self.frac2orth_tensor)
        return orth_pos
    
    def orth2frac(self, orth_pos: torch.Tensor) -> torch.Tensor:
        """
        Convert orthogonal coordinates to fractional coordinates

        Args:
            orth_pos: torch.Tensor, [n_points, ..., 3]
        
        Returns:
            fractional coordinates, torch.Tensor
        """
        frac_pos = torch.einsum("n...x,yx->n...y", orth_pos, self.orth2frac_tensor)
        return frac_pos
    
    def exp_sym(self, frac_pos: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply all symmetry operations to the fractional coordinates

        Args:
            frac_pos, torch.Tensor, [n_points, 3]
                fractional coordinates of model in single ASU. 
                If not given, will use self.atom_pos_frac
        Returns:
            torch.Tensor, [n_points, n_ops, 3]
                fractional coordinates of symmetry operated models        
        """
        if frac_pos is None:
            frac_pos = self.atom_pos_frac
        sym_oped_frac_pos = torch.einsum("oxy,ay->aox", self.R_G_tensor_stack, frac_pos) + self.T_G_tensor_stack
        return sym_oped_frac_pos
    
    def init_mtz(self, mtzdata, N_bins, expcolumns, set_experiment, freeflag, testset_value, dmin):
        """
        set mtz file for HKL list, resolution and experimental related properties
        """

        if type(mtzdata) == str:
            if mtzdata.endswith(".mtz"):
                mtz_reference = rs.read_mtz(mtzdata)
            elif mtzdata.endswith(".cif"):
                mtz_reference = rs.read_cif(mtzdata)
            else:
                raise ValueError("mtzdata should be rs.Dataset object or path str to a mtz/cif file!")
        elif type(mtzdata) == rs.DataSet:
            mtz_reference = mtzdata
        else:
            raise TypeError("mtzdata should be rs.Dataset object or path str to a mtz/cif file!")

        try:
            mtz_reference.dropna(axis=0, subset=expcolumns, inplace=True)
        except:
            raise ValueError(f"{expcolumns} columns not included in the mtz file!")
        
        # Make sure no systematic absent indices are included, and map to asu
        if rs.utils.is_absent(mtz_reference.get_hkls(), mtz_reference.spacegroup).sum() > 0:
            mtz_reference = mtz_reference[~rs.utils.is_absent(mtz_reference.get_hkls(), mtz_reference.spacegroup)]
            logger.info("Found systematic absent indices in the mtz, already dropped!")
        mtz_reference.hkl_to_asu(inplace=True, anomalous=self.anomalous)
        
        if self.anomalous:
            # Try to get the wavelength from MTZ file
            try:
                mtz_wavelength = mtz_reference.dataset(0).wavelength
                assert mtz_wavelength > 0.05
                if self.wavelength is not None:
                    assert np.isclose(mtz_wavelength, self.wavelength, atol=0.05)
                else:
                    self.wavelength = mtz_wavelength
            except:
                logger.warning(
                    "Can't find wavelength record in the MTZ file, or it doesn't match with other sources"
                )

        if (mtz_reference.cell == self._pdb.cell):
            pass
        else:
            logger.warning("Unit cell from mtz file does not match that in PDB file! Using the cell info from MTZ file!")
            self._pdb.set_unitcell(mtz_reference.cell)

        if (mtz_reference.spacegroup.hm == self._pdb.spacegroup.hm):
            pass
        else:
            logger.warning("Space group from mtz file does not match that in PDB file! Using the spacegroup from MTZ file!")  # type: ignore
            self._pdb.set_spacegroup(mtz_reference.spacegroup)

        # HKL array from the reference mtz file, [N,3]
        self.HKL_array = mtz_reference.get_hkls()
        self.dHKL = self.unit_cell.calculate_d_array(self.HKL_array).astype(
            "float32"
        )
        if dmin is None:
            self.dmin = self.dHKL.min()
            resol_bool = None
        else:
            resol_bool = self.dHKL >= dmin
            self.dmin = dmin
            self.dHKL = self.dHKL[resol_bool]
            self.HKL_array = self.HKL_array[resol_bool] 
            
        if self.mode == 'cryoem':
            # override the asu settings for cryoem cases, issue #10
            self.Hasu_array = self.HKL_array.copy()
        else:
            self.Hasu_array = generate_reciprocal_asu(
                self.unit_cell, self.space_group, self.dmin, anomalous=self.anomalous
            )
        assert (
            diff_array(self.HKL_array, self.Hasu_array) == set()
        ), "HKL_array should be equal or subset of the Hasu_array!"
        self.asu2HKL_index = asu2HKL(self.Hasu_array, self.HKL_array)
        # d*^2 array according to the HKL list, [N]
        self.dr2asu_array = self.unit_cell.calculate_1_d2_array(self.Hasu_array)
        self.dr2HKL_array = self.unit_cell.calculate_1_d2_array(self.HKL_array)
        # assign reslution bins
        self.assign_resolution_bins(bins=N_bins)
        if set_experiment:
            self.set_experiment(mtz_reference, expcolumns, freeflag, testset_value, resol_bool)

    def set_experiment(self, exp_mtz, expcolumns=["FP", "SIGFP"], freeflag="FreeR_flag", testset_value=0, resol_bool=None):
        """
        Set experimental data for refinement,
        including Fo, SigF, free_flag, Outlier

        exp_mtz, rs.Dataset, mtzfile read by reciprocalspaceship
        """
        if resol_bool is None:
            resol_bool = np.ones_like(self.dHKL, dtype=bool)
        exp_mtz = exp_mtz[resol_bool].copy()
        
        if isinstance(exp_mtz[expcolumns[0]].dtype, rs.IntensityDtype):
            raise ValueError(
                f"Experimental data {expcolumns[0]} should be Structure Factor Amplitude, not Intensity! Convert it to Structure Factor Amplitude first!"
            )
        
        if isinstance(exp_mtz[expcolumns[0]].dtype, rs.FriedelIntensityDtype):
            raise ValueError(
                f"Experimental data {expcolumns[0]} should be Structure Factor Amplitude, not Friedel Intensity! Convert it to Friedel Structure Factor Amplitude and run stack_anomalous first!"
            )

        if isinstance(exp_mtz[expcolumns[0]].dtype, rs.FriedelStructureFactorAmplitudeDtype):
            raise ValueError(
                f"Experimental data {expcolumns[0]} should be Structure Factor Amplitude, not Friedel Structure Factor Amplitude! Try to run stack_anomalous first!"
            )
        
        try:
            self.Fo = torch.tensor(exp_mtz[expcolumns[0]].to_numpy(), device=self.device).type(
                torch.float32
            )
            self.SigF = torch.tensor(
                exp_mtz[expcolumns[1]].to_numpy(), device=self.device
            ).type(torch.float32)
            nonpositive_Fo = assert_numpy(self.Fo <= 0.0)
            if nonpositive_Fo.sum() > 0:
                logger.info(f"With {nonpositive_Fo.sum()} non-positive experimental magnitudes! Will be treated as outliers.")
        except:
            logger.warning(f"MTZ file doesn't contain {expcolumns[0]} or {expcolumns[1]}! Check your data!")

        try:
            self.free_flag = np.where(
                exp_mtz[freeflag].values == testset_value, True, False
            )
        except:
            self.free_flag = np.zeros(len(self.Fo)).astype(bool) 
            logger.warning("No Free Flag Column, will treat all elements as working set! Check your data!")

        # label outliers
        # Detecting outliers in non-redundant diffraction data, Read, 1999
        try:
            exp_mtz.label_centrics(inplace=True)
            exp_mtz.compute_multiplicity(inplace=True)
            self.centric = exp_mtz["CENTRIC"].to_numpy()
            self.epsilon = torch.tensor(exp_mtz["EPSILON"].to_numpy(), device=self.device).type(
                torch.float32
            )
            exp_mtz["SIGN"] = 0.0
            for i in range(self.n_bins):
                index_i = self.bins == i
                exp_mtz.loc[index_i, "SIGN"] = np.mean(
                    exp_mtz[index_i][expcolumns[0]].to_numpy() ** 2
                    / exp_mtz[index_i]["EPSILON"].to_numpy()
                )
            exp_mtz["EP"] = exp_mtz[expcolumns[0]] / np.sqrt(exp_mtz["EPSILON"].astype(float) * exp_mtz["SIGN"])
            self._SigN = torch.tensor(exp_mtz["SIGN"].to_numpy(), device=self.device).type(
                torch.float32
            ) 
            self.Eo = torch.tensor(exp_mtz["EP"].to_numpy(), device=self.device).type(
                torch.float32
            )
            self.SigEo = self.SigF / (self._SigN * self.epsilon).sqrt()
            self.Outlier = (
                (exp_mtz["CENTRIC"] & (exp_mtz["EP"] > 4.89))
                | ((~exp_mtz["CENTRIC"]) & (exp_mtz["EP"] > 3.72))
            ).to_numpy(dtype=bool)
        except:
            self.Outlier = np.zeros(len(self.Fo)).astype(bool)
            logger.info("No outlier detection, will use all reflections!")
        self.Outlier = self.Outlier | nonpositive_Fo 
    
    def init_withoutmtz(self, dmin, n_bins):
        if not dmin:
            raise ValueError(
                "high_resolution dmin OR a reference mtz file should be provided!"
            )
        else:
            self.dmin = dmin
            # Generate ASU HKL array and Corresponding d*^2 array
            self.Hasu_array = generate_reciprocal_asu(
                self.unit_cell, self.space_group, self.dmin
            )
            self.dHasu = self.unit_cell.calculate_d_array(self.Hasu_array).astype(
                "float32"
            )
            self.dr2asu_array = self.unit_cell.calculate_1_d2_array(self.Hasu_array)
            self.HKL_array = None
            self.assign_resolution_bins(n_bins)

    def init_atomic_scattering(self):
        """
        mode xray: xray atomic scatterring factor
            A dictionary of atomic structural factor f0_sj of different atom types at different HKL Rupp's Book P280
            f0_sj = [sum_{i=1}^4 {a_ij*exp(-b_ij* d*^2/4)} ] + c_j
        
        mode cryoem: electron atomic scattering factor
            The electron form factors are parametrized as 5 Gaussians, using coefficients from International Tables 
            for Crystallography Volume C, edition 2011, table 4.3.2.2 (pp. 282-283), “Elastic atomic scattering 
            factors of electrons for neutral atoms and s up to 2.0 A–1”. The same parametrization is used in cctbx and in CCP4.
        """
        
        if self.anomalous:
            assert self.mode == "xray", ValueError(
                "Only support anomalous scattering for xray mode!"
            )
            assert self.wavelength is not None, ValueError(
                "If you need anomalous scattering contribution, provide the wavelength info from input, pdb or mtz file!"
            )

        self.full_atomic_sf_asu = {}
        for atom_type in self.unique_atom:
            element = gemmi.Element(atom_type)
            if self.mode == "xray":
                f0 = np.array(
                    [element.it92.calculate_sf(dr2 / 4.0) for dr2 in self.dr2asu_array]
                )
                if self.anomalous:
                    fp, fpp = gemmi.cromer_liberman(
                        z=element.atomic_number, energy=gemmi.hc / self.wavelength
                    )
                    self.full_atomic_sf_asu[atom_type] = f0 + fp + 1j * fpp
                else:
                    self.full_atomic_sf_asu[atom_type] = f0
            elif self.mode == "cryoem":
                f0 = np.array(
                    [element.c4322.calculate_sf(dr2 / 4.0) for dr2 in self.dr2asu_array]
                )
                self.full_atomic_sf_asu[atom_type] = f0

        if self.anomalous:
            self.fullsf_tensor = torch.tensor(
                np.array([self.full_atomic_sf_asu[atom] for atom in self.atom_name]),
                device=self.device,
            ).type(torch.complex64)
        else:
            self.fullsf_tensor = torch.tensor(
                np.array([self.full_atomic_sf_asu[atom] for atom in self.atom_name]),
                device=self.device,
            ).type(torch.float32)


    def assign_resolution_bins(
        self, bins=10, Nmin=100, return_labels=False, format_str=".2f"
    ):
        """Assign reflections in HKL_array to resolution bins with logarithmic algorithm.
        The labels will be stored in self.bins

        Urzhumtsev, A., et al. Acta Crystallographica Section D: Biological Crystallography 65.12 (2009)
        https://doi.org/10.1107/S0907444909039638 

        Parameters
        ----------
        bins : int, optional
            Number of bins, by default 10
        Nmin : int, optional
            Minimum number of reflections for the first two low-resolution ranges, by default 100
        return_labels : bool, optional
            Whether to return a list of labels corresponding to the edges
            of each resolution bin, by default False
        format_str : str, optional
            Format string for constructing bin labels, by default ".2f"

        Returns
        -------
        None or list of labels
        """
        if not self.HKL_array is None:
            assert hasattr(
                self, "dHKL"
            ), "Must have resolution stored in dHKL attribute!"
            d = self.dHKL
        else:
            assert hasattr(
                self, "dHasu"
            ), "Must have resolution stored in dHasu attribute!"
            d = self.dHasu
        assignments, edges = bin_by_logarithmic(d, bins, Nmin)
        self.n_bins = bins
        self.bins = assignments
        self.bin_labels = [
            f"{e1:{format_str}} - {e2:{format_str}}"
            for e1, e2 in zip(edges[:-1], edges[1:])
        ]
        if return_labels:
            return self.bin_labels

    def inspect_data(self, verbose=False, spacing=4.5):
        """
        Do an inspection of data, for hints about
        1. solvent percentage for mask calculation
        2. suitable grid size
        """
        # solvent percentage
        vdw_rad = vdw_rad_tensor(self.atom_name, device=self.device)
        uc_grid_orth_tensor = unitcell_grid_center(
            self.unit_cell, spacing=spacing, return_tensor=True, device=self.device
        )
        occupancy, _ = packingscore_voxelgrid_torch(
            self.atom_pos_orth,
            self.unit_cell,
            self.space_group,
            vdw_rad,
            uc_grid_orth_tensor,
        )
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

    def calc_fprotein(
        self,
        atoms_position_tensor=None,
        atoms_biso_tensor=None,
        atoms_aniso_uw_tensor=None,
        atoms_occ_tensor=None,
        Return=False,
    ):
        """
        Calculate the structural factor from a single atomic model, without solvent masking

        Parameters
        ----------
        atoms_positions_tensor: 2D [N_atoms, 3] tensor or default None
            Positions of atoms in the model, in unit of angstrom; If not given, the model stored in attribute `atom_pos_orth` will be used

        atoms_biso_tensor: 1D [N_atoms,] tensor or default None
            Isotropic B factors of each atoms in the model; If not given, the info stored in attribute `atoms_b_iso` will be used

        atoms_aniso_uw_tensor: 3D [N_atoms, 3, 3] tensor or default None
            Anisotropic B factors of each atoms in the model, in matrix form; If not given, the info stored in attribute `atoms_aniso_uw` will be used

        atoms_occ_tensor: 1D [N_atoms,] tensor or default None
            Occupancy of each atoms in the model; If not given, the info stored in attribute `atom_occ` will be used

        NO_Bfactor: Boolean, default False
            If True, the calculation will not use Bfactor parameterization; Useful when we are parameterizing the ensemble with a true distribution

        Return: Boolean, default False
            If True, it will return the Fprotein as the function output; Or It will just be saved in the `Fprotein_asu` and `Fprotein_HKL` attributes

        Returns
        -------
        None (Return=False) or Fprotein (Return=True)
        """
        # Read and tensor-fy necessary inforamtion
        if not atoms_position_tensor is None:
            self.atom_pos_orth = atoms_position_tensor

        if not atoms_aniso_uw_tensor is None:
            self.atom_aniso_uw = atoms_aniso_uw_tensor

        if not atoms_biso_tensor is None:
            self.atom_b_iso = atoms_biso_tensor

        if not atoms_occ_tensor is None:
            self.atom_occ = atoms_occ_tensor

        self.Fprotein_asu = F_protein(
            self.Hasu_array,
            self.dr2asu_array,
            self.fullsf_tensor,
            self.R_G_tensor_stack,
            self.T_G_tensor_stack,
            self.orth2frac_tensor,
            self.atom_pos_frac,
            self.atom_b_iso,
            self.atom_aniso_uw,
            self.atom_occ,
        )
        if not self.HKL_array is None:
            self.Fprotein_HKL = self.Fprotein_asu[self.asu2HKL_index]
            self.Fmask_HKL = torch.zeros_like(self.Fprotein_HKL)
            if Return:
                return self.Fprotein_HKL
        else:
            self.Fmask_asu = torch.zeros_like(self.Fprotein_asu)
            if Return:
                return self.Fprotein_asu

    def calc_fsolvent(
        self,
        solventpct=None,
        gridsize=None,
        dmin_mask=5.0,
        dmin_nonzero=3.0,
        exponent=10.0,
        Return=False,
    ):
        """
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
        """

        if solventpct is None:
            assert (
                self.inspected
            ), "Run inspect_data first or give a valid solvent percentage!"
            solventpct = self.solventpct

        if gridsize is None:
            assert self.inspected, "Run inspect_data first or give a valid grid size!"
            gridsize = self.gridsize

        # Shape [N_HKL_p1, 3], [N_HKL_p1,]
        Hp1_array, Fp1_tensor = expand_to_p1(
            self.space_group,
            self.Hasu_array,
            self.Fprotein_asu,
            dmin_mask=dmin_mask,
            unitcell=self.unit_cell,
            anomalous=self.anomalous,
        )
        rs_grid = reciprocal_grid(Hp1_array, Fp1_tensor, gridsize)
        self.real_grid_mask = rsgrid2realmask(
            rs_grid, solvent_percent=solventpct, exponent=exponent, 
        )  # type: ignore
        if not self.HKL_array is None:
            self.Fmask_HKL = realmask2Fmask(self.real_grid_mask, self.HKL_array)
            zero_hkl_bool = torch.tensor(self.dHKL <= dmin_nonzero, device=self.device)
            self.Fmask_HKL[zero_hkl_bool] = torch.tensor(
                0.0, device=self.device, dtype=torch.complex64
            )
            if Return:
                return self.Fmask_HKL
        else:
            self.Fmask_asu = realmask2Fmask(self.real_grid_mask, self.Hasu_array)
            zero_hkl_bool = torch.tensor(self.dHasu <= dmin_nonzero, device=self.device)
            self.Fmask_asu[zero_hkl_bool] = torch.tensor(
                0.0, device=self.device, dtype=torch.complex64
            )
            if Return:
                return self.Fmask_asu

    def _init_kmask_kiso(self, requires_grad=True):
        """
        Use the root finding approach discussed to initialize kmask and kiso per resolution bin
        Afonine, P. V., et al. Acta Crystallographica Section D: Biological Crystallography 69.4 (2013): 625-634.

        Note: Only work when you have mtz data, self.Fo
        """

        kmasks = []
        kisos = []
        Fmask_HKL = self.Fmask_HKL.detach().clone()
        Fprotein_HKL = self.Fprotein_HKL.detach().clone()
        ws = torch.abs(Fmask_HKL).pow(2)
        vs = 0.5 * torch.real(
            Fprotein_HKL.conj() * Fmask_HKL
            + Fprotein_HKL * Fmask_HKL.conj()
        )
        us = torch.abs(Fprotein_HKL).pow(2)
        Is = self.Fo.pow(2)

        for bin_i in np.sort(np.unique(self.bins)):
            index_i = (~self.free_flag) & (self.bins == bin_i) & (~self.Outlier)

            C2 = torch.sum(ws[index_i] * Is[index_i])
            B2 = 2.0 * torch.sum(vs[index_i] * Is[index_i])
            A2 = torch.sum(us[index_i] * Is[index_i])
            Y2 = torch.sum(Is[index_i].pow(2))
            # They made this wrong in their original paper
            D3 = torch.sum(ws[index_i].pow(2))
            C3 = 3.0 * torch.sum(ws[index_i] * vs[index_i])
            B3 = torch.sum(2 * vs[index_i].pow(2) + us[index_i] * ws[index_i])
            A3 = torch.sum(us[index_i] * vs[index_i])
            Y3 = torch.sum(Is[index_i] * vs[index_i])

            a = assert_numpy((C3 * Y2 - C2 * B2 - C2 * Y3) / (D3 * Y2 - C2**2))
            b = assert_numpy((B3 * Y2 - C2 * A2 - Y3 * B2) / (D3 * Y2 - C2**2))
            c = assert_numpy((A3 * Y2 - Y3 * A2) / (D3 * Y2 - C2**2))

            try:
                roots = np.roots([1.0, a, b, c])
            except:
                # In case no nonzero Fmask observations for this resolution bin
                kmask = torch.tensor(1e-4).to(C2)
                K_temp = (kmask.pow(2) * C2 + kmask * B2 + A2) / Y2
                kiso = torch.sqrt(K_temp).reciprocal()
                kmasks.append(kmask.requires_grad_())
                kisos.append(kiso.requires_grad_())
                continue

            kmask_candidates = torch.zeros(3).to(C2)
            LSpp_candidates = torch.zeros(3).to(C2)
            K_candidates = torch.zeros(3).to(C2)
            LS_candidates = torch.zeros(3).to(C2)
            for i, root in enumerate(roots):
                if np.iscomplex(root) or (root < 0):
                    kmask_temp = torch.tensor(0.0001).to(C2)
                else:
                    kmask_temp = torch.tensor(np.real_if_close(root, tol=1000)).to(C2)
                K_temp = (kmask_temp.pow(2) * C2 + kmask_temp * B2 + A2) / Y2
                if K_temp < 0.0:
                    kmask_temp = torch.tensor(1.0).to(C2)
                LS_temp = torch.sum(
                    torch.abs(
                        self.Fprotein_HKL[index_i]
                        + kmask_temp * self.Fmask_HKL[index_i]
                    ).pow(2)
                    - K_temp * Is[index_i]
                ).pow(2)
                LSpp_temp = (
                    3 * kmask_temp.pow(2) * D3 + 2 * kmask_temp * C3 + B3 - C2 * K_temp
                )
                kmask_candidates[i] = kmask_temp
                # Note, K should be positive too
                K_candidates[i] = K_temp
                LSpp_candidates[i] = LSpp_temp
                LS_candidates[i] = LS_temp

            legitimacy = (
                (roots > 0) & np.isreal(roots) & (assert_numpy(LSpp_candidates) > 0)
            )
            if (np.sum(legitimacy) > 0) and (np.sum(legitimacy) < 3):
                kmask_candidates = kmask_candidates[legitimacy]
                LS_candidates = LS_candidates[legitimacy]
                K_candidates = K_candidates[legitimacy]

            min_index = torch.argmin(LS_candidates)
            kmask = kmask_candidates[min_index]
            # K = k_total^(-2)
            kiso = torch.sqrt(K_candidates[min_index]).reciprocal()

            kmasks.append(kmask.requires_grad_(requires_grad))
            kisos.append(kiso.requires_grad_(requires_grad))
        return kmasks, kisos

    def _init_uaniso(self, requires_grad=True):
        """
        Use the analytical solutuon discussed to initialize Uaniso per resolution bin
        Afonine, P. V., et al. Acta Crystallographica Section D: Biological Crystallography 69.4 (2013): 625-634.

        Note: Only work when you have mtz data, self.Fo
        """
        uanisos = []
        for bin_i in np.sort(np.unique(self.bins)):
            index_i = (self.bins == bin_i) & (~self.free_flag) & (~self.Outlier)
            s = self.HKL_array[index_i]
            V = np.concatenate([s**2, 2 * s[:, [0, 2, 1]] * s[:, [1, 0, 2]]], axis=-1)
            Z = assert_numpy(
                torch.log(
                    self.Fo[index_i]
                    / (
                        self.kisos[bin_i]
                        * torch.abs(
                            self.Fprotein_HKL[index_i]
                            + self.kmasks[bin_i] * self.Fmask_HKL[index_i]
                        )
                    )
                )
                / (2.0 * np.pi**2)
            )
            M = V.T @ V  # M = np.einsum("ki,kj->ij", V, V)
            b = -np.sum(Z * V.T, axis=-1)
            U = np.linalg.inv(M) @ b
            uanisos.append(torch.tensor(U).to(self.Fo).requires_grad_(requires_grad))
        return uanisos

    def _set_scales(
        self,
        requires_grad,
        kiso=1.0,
        kmask=0.35,
        uaniso=[0.01, 0.01, 0.01, 1e-4, 1e-4, 1e-4],
    ):
        """Only used for case you don't have data"""
        self.kmasks = [
            torch.tensor(kmask).to(self.atom_pos_frac).requires_grad_(requires_grad)
            for i in range(self.n_bins)
        ]
        self.kisos = [
            torch.tensor(kiso).to(self.atom_pos_frac).requires_grad_(requires_grad)
            for i in range(self.n_bins)
        ]
        self.uanisos = [
            torch.tensor(uaniso).to(self.atom_pos_frac).requires_grad_(requires_grad)
            for i in range(self.n_bins)
        ]

    def init_scales(self, requires_grad=True):
        if hasattr(self, "Fo"):
            self.kmasks, self.kisos = self._init_kmask_kiso(requires_grad=requires_grad)
            self.uanisos = self._init_uaniso(requires_grad=requires_grad)
            Fmodel = self.calc_ftotal()
            self.r_work, self.r_free = self.get_rfactors(ftotal=Fmodel)
        else:
            self._set_scales(requires_grad)

    def _get_scales_lbfgs_LS(
        self, n_steps=3, lr=0.1, verbose=True, initialize=True, return_loss=False
    ):
        """
        Use LBFGS to optimize scales with least square error
        """
        assert hasattr(self, "Fo"), "No experimental data Fo!"

        if initialize:
            self.init_scales(requires_grad=True)
        else:
            self.unfreeze_scales()

        def closure():
            Fmodel = self.calc_ftotal(scale_mode=True)
            Fmodel_mag = torch.abs(Fmodel)
            # LS loss
            loss = torch.sum(
                (
                    self.Fo[(~self.free_flag) & (~self.Outlier)]
                    - Fmodel_mag[(~self.free_flag) & (~self.Outlier)]
                )
                ** 2
            )
            self.lbfgs.zero_grad()
            loss.backward()
            return loss

        params = self.kmasks + self.kisos + self.uanisos
        self.lbfgs = torch.optim.LBFGS(params, lr=lr)
        loss_track = []
        for _ in range(n_steps):
            start_time = time.time()
            loss = self.lbfgs.step(closure)
            Fmodel = self.calc_ftotal()
            r_work, r_free = self.get_rfactors(ftotal=Fmodel)
            loss_track.append(
                [assert_numpy(loss), assert_numpy(r_work), assert_numpy(r_free)]
            )
            str_ = f"Time: {time.time()-start_time:.3f}"
            if verbose:
                logger.info(
                    f"Scale, {loss_track[-1][0]:.3f}, {loss_track[-1][1]:.3f}, {loss_track[-1][2]:.3f}",
                    str_,
                )
        self.r_work, self.r_free = r_work, r_free
        if return_loss:
            return loss_track

    def _get_scales_lbfgs_r(
        self, n_steps=5, lr=0.1, verbose=True, initialize=False, return_loss=False
    ):
        """
        Use LBFGS to optimize scales directly with r factor error
        """
        assert hasattr(self, "Fo"), "No experimental data Fo!"

        if initialize:
            self.init_scales(requires_grad=True)
        else:
            self.unfreeze_scales()

        def closure():
            Fmodel = self.calc_ftotal(scale_mode=True)
            Fmodel_mag = torch.abs(Fmodel)
            # R factor
            loss = torch.sum(
                torch.abs(
                    self.Fo[(~self.free_flag) & (~self.Outlier)]
                    - Fmodel_mag[(~self.free_flag) & (~self.Outlier)]
                )
            ) / torch.sum(self.Fo[(~self.free_flag) & (~self.Outlier)])
            self.lbfgs.zero_grad()
            loss.backward()
            return loss

        params = self.kmasks + self.kisos + self.uanisos
        self.lbfgs = torch.optim.LBFGS(params, lr=lr)
        loss_track = []
        for _ in range(n_steps):
            start_time = time.time()
            loss = self.lbfgs.step(closure)
            Fmodel = self.calc_ftotal()
            r_work, r_free = self.get_rfactors(ftotal=Fmodel)
            loss_track.append(
                [assert_numpy(loss), assert_numpy(r_work), assert_numpy(r_free)]
            )
            str_ = f"Time: {time.time()-start_time:.3f}"
            if verbose:
                logger.info(
                    f"Scale, {loss_track[-1][0]:.3f}, {loss_track[-1][1]:.3f}, {loss_track[-1][2]:.3f}",
                    str_,
                )
        self.r_work, self.r_free = r_work, r_free
        if return_loss:
            return loss_track

    def get_scales_lbfgs(
        self,
        ls_steps=3,
        r_steps=3,
        ls_lr=0.1,
        r_lr=0.1,
        initialize=True,
        verbose=False,
    ):
        self._get_scales_lbfgs_LS(ls_steps, ls_lr, verbose, initialize)
        self._get_scales_lbfgs_r(r_steps, r_lr, verbose, initialize=False)
        self.freeze_scales()

    def freeze_scales(self):
        """
        Do not require grad on scales
        """
        self.kmasks = [kmask.requires_grad_(False) for kmask in self.kmasks]
        self.kisos = [kiso.requires_grad_(False) for kiso in self.kisos]
        self.uanisos = [uaniso.requires_grad_(False) for uaniso in self.uanisos]
    
    def unfreeze_scales(self):
        """
        Do not require grad on scales
        """
        self.kmasks = [kmask.requires_grad_(True) for kmask in self.kmasks]
        self.kisos = [kiso.requires_grad_(True) for kiso in self.kisos]
        self.uanisos = [uaniso.requires_grad_(True) for uaniso in self.uanisos]

    def get_scales_adam(
        self, 
        lr=0.01, 
        n_steps=100, 
        sub_ratio=0.7, 
        initialize=True, 
        verbose=False
    ):
        def adam_opt_i(
            bin_i, index_i, sub_ratio=0.3, lr=0.001, n_steps=100, verbose=False
        ):
            def adam_stepopt(sub_boolean_mask):
                Fmodel_i = self._calc_ftotal_bini(
                    bin_i, index_i, self.HKL_array, self.Fprotein_HKL, self.Fmask_HKL, scale_mode=True
                )
                Fmodel_mag = torch.abs(Fmodel_i)
                # LS loss with subsampling
                fo_i_sub = fo_i[sub_boolean_mask]
                Fmodel_i_sub = Fmodel_mag[sub_boolean_mask]
                free_flagi_sub = free_flagi[sub_boolean_mask]
                loss = torch.sum(
                    (fo_i_sub[~free_flagi_sub] - Fmodel_i_sub[~free_flagi_sub]) ** 2
                )
                # loss = torch.sum(torch.abs(fo_i_sub[~free_flagi_sub] - Fmodel_i_sub[~free_flagi_sub]))/torch.sum(Fmodel_i_sub[~free_flagi_sub])
                r_work, r_free = r_factor(fo_i, Fmodel_mag, free_flagi)
                adam.zero_grad()
                loss.backward()
                adam.step()
                return loss, r_work, r_free

            params = [self.kmasks[bin_i], self.kisos[bin_i], self.uanisos[bin_i]]
            adam = torch.optim.Adam(params, lr=lr)
            for _ in range(n_steps):
                start_time = time.time()
                sub_boolean_mask = (
                    np.random.rand(
                        np.sum(index_i),
                    )
                    < sub_ratio
                )
                temp = adam_stepopt(sub_boolean_mask)
                time_this_round = round(time.time() - start_time, 3)
                str_ = "Time: " + str(time_this_round)
                if verbose:
                    logger.info("Scale", *[assert_numpy(i) for i in temp], str_)

        if initialize:
            self.init_scales(requires_grad=True)
        else:
            self.unfreeze_scales()

        for bin_i in range(self.n_bins):
            index_i = (self.bins == bin_i) & (~self.Outlier)
            fo_i = self.Fo[index_i]
            free_flagi = self.free_flag[index_i]
            adam_opt_i(
                bin_i,
                index_i,
                lr=lr,
                n_steps=n_steps,
                sub_ratio=sub_ratio,
                verbose=verbose,
            )
        self.freeze_scales()
        Fmodel = self.calc_ftotal()
        self.r_work, self.r_free = self.get_rfactors(ftotal=Fmodel)

    def _calc_ftotal_bini(self, bin_i, index_i, HKL_array, Fprotein, Fmask, scale_mode=False):
        """
        calculate ftotal for bin i
        """
        if scale_mode:
            Fmask = Fmask.detach()
            Fprotein = Fprotein.detach()
        scaled_fmask_i = Fmask[index_i] * self.kmasks[bin_i]
        fmodel_i = (
            self.kisos[bin_i]
            * aniso_scaling(
                self.uanisos[bin_i],
                self.reciprocal_cell_paras,
                HKL_array[index_i],
            )
            * (Fprotein[index_i] + scaled_fmask_i)
        )
        return fmodel_i

    def calc_ftotal(self, bins=None, Return=True, scale_mode=False):
        """
        Calculate Ftotal = kiso * exp(-2*pi^2*s^T*Uaniso*s) * (Fprotein + kmask * Fmask)

        kiso, uaniso and kmask are stored for each resolution bin

        Parameters
        ----------
        bins: None or List[int], default None
            Specify which resolution bins to calculate the ftotal, if None, calculate for all

        Return: Boolean, default True
            Whether to return the results

        Returns
        -------
        torch.tensor, complex
        """
        if bins is None:
            bins = range(self.n_bins)

        if not self.HKL_array is None:
            ftotal_hkl = torch.zeros_like(self.Fprotein_HKL)
            for bin_i in bins:
                index_i = self.bins == bin_i
                ftotal_hkl[index_i] = self._calc_ftotal_bini(
                    bin_i, index_i, self.HKL_array, self.Fprotein_HKL, self.Fmask_HKL, scale_mode=scale_mode
                )
            self.Ftotal_HKL = ftotal_hkl
            if Return:
                return ftotal_hkl
        else:
            ftotal_asu = torch.zeros_like(self.Fprotein_asu)
            for bin_i in bins:
                index_i = self.bins == bin_i
                ftotal_asu[index_i] = self._calc_ftotal_bini(
                    bin_i, index_i, self.Hasu_array, self.Fprotein_asu, self.Fmask_asu, scale_mode=scale_mode
                )
            self.Ftotal_asu = ftotal_asu
            if Return:
                return ftotal_asu
    
    def calc_Ec(self, Ftotal_HKL=None):
        """
        compute normalized magnitudes from Fc
        
        Args:
            Ftotal_HKL (torch.Tensor) : [N_HKL] or [N_batch, N_HKL]
        
        Returns:
            Ec_HKL : [N_HKL] or [N_batch, N_HKL]
        """ 
        if Ftotal_HKL is None:
            Ftotal_HKL = self.Ftotal_HKL

        SigmaP = torch.zeros_like(Ftotal_HKL, dtype=torch.float32)
        for i in range(self.n_bins):
            index_i = self.bins == i
            SigmaP[..., index_i] = torch.mean(
                Ftotal_HKL[..., index_i].abs().square()
                / self.epsilon[index_i], dim=-1, keepdims=True
            )
        Ec_HKL = Ftotal_HKL / torch.sqrt(self.epsilon * SigmaP)
        return Ec_HKL
    
    def get_sigmaA(self, Ec_HKL, subset="working"):
        """
        Compute sigmaA using corr(Eo, Ec)

        Args:
            Ec_HKL (torch.tensor) : [N_HKL] or [N_batch, N_HKL]
        
        Returns:
            sigmaAs [N_bins] or [N_batch, N_bins]
        """
        if subset == "working":
            subset_boolean = (~self.free_flag) & (~self.Outlier)
        else:
            subset_boolean = (~self.Outlier)
        Ecalc = Ec_HKL.abs().detach().clone()
        sigmaAs = torch.zeros(*Ecalc.shape[:-1], self.n_bins, dtype=torch.float32, device=self.device)
        for i in range(self.n_bins):
            index_i = self.bins[subset_boolean] == i
            Eobs_i = self.Eo[subset_boolean][index_i].square() # [N_subset]
            Ecalc_i = Ecalc[..., subset_boolean][..., index_i].square() # [N_batch, N_subset] or [N_subset]
            # Compute correlation coefficient
            Eoi_centered = Eobs_i - Eobs_i.mean()
            Eci_centered = Ecalc_i - torch.mean(Ecalc_i, dim=-1, keepdims=True)
            Covi = (Eci_centered @ Eoi_centered) / (Eoi_centered.shape[0] -1)
            Eoi_std = torch.std(Eoi_centered, correction=0)
            Eci_std = torch.std(Eci_centered, dim=-1, correction=0)
            sigmaAs[..., i] = (Covi / (Eoi_std * Eci_std)).clamp(min=0.001, max=0.999).sqrt()
        return sigmaAs

    def summarize(self, Ftotal_HKL=None):
        """
        Print model quality log like phenix.model_vs_data
        """
        if Ftotal_HKL is None:
            ftotal = self.calc_ftotal(Return=True)
        else:
            ftotal = Ftotal_HKL
        _, counts = np.unique(self.bins, return_counts=True)
        print(
            f"{'Resolution':15} {'N_work':>7} {'N_free':>7} {'<Fobs>':>7} {'<|Fmodel|>':>9} {'R_work':>7} {'R_free':>7} {'k_mask':>7} {'k_iso':>7}"
        )
        for i in range(self.n_bins):
            index_i = (self.bins == i) & (~self.Outlier)
            r_worki, r_freei = r_factor(
                self.Fo[index_i], torch.abs(ftotal[index_i]), self.free_flag[index_i]
            )
            N_work = counts[i] - np.sum(self.free_flag[index_i])
            N_free = np.sum(self.free_flag[index_i])
            print(
                f"{self.bin_labels[i]:<15} {N_work:7d} {N_free:7d} {assert_numpy(torch.mean(self.Fo[index_i])):7.1f} {assert_numpy(torch.mean(torch.abs(ftotal[index_i]))):9.1f} {assert_numpy(r_worki):7.3f} {assert_numpy(r_freei):7.3f} {assert_numpy(self.kmasks[i]):7.3f} {assert_numpy(self.kisos[i]):7.3f}"
            )
        self.r_work, self.r_free = self.get_rfactors(ftotal=ftotal)
        print(f"r_work: {assert_numpy(self.r_work):<7.3f}")
        print(f"r_free: {assert_numpy(self.r_free):<7.3f}")
        print(f"Number of outliers: {np.sum(self.Outlier):<7d}")
    
    def get_rfactors(self, ftotal=None, mask=None):
        if ftotal is None:
            ftotal = self.Ftotal_HKL.detach().clone()
        if mask is None:
            mask = np.ones(len(self.Fo)).astype(bool)
        bool_array = (~self.Outlier) & mask
        r_work, r_free = r_factor(
            self.Fo[bool_array],
            torch.abs(ftotal)[bool_array],
            self.free_flag[bool_array],
        )
        return r_work, r_free

    def calc_fprotein_batch(self, atoms_position_batch, Return=False, PARTITION=20):
        """
        Calculate the Fprotein with batched models. Most parameters are similar to `Calc_Fprotein`

        atoms_positions_batch: torch.float32 tensor, [N_batch, N_atoms, 3]

        PARTITION: Int, default 20
            To reduce the memory cost during the computation, we divide the batch into several partitions and loops through them.
            Larger PARTITION will require larger GPU memory. Default 20 will require around 21GB, if N_atoms~2400 and N_HKLs~24000. 
            Memory scales linearly with PARITION, N_atoms, and N_HKLs.
            But larger PARTITION will give a smaller wall time, so this is a trade-off.
        """
        # Read and tensor-fy necessary information
        atom_pos_frac_batch = torch.tensordot(
            atoms_position_batch, self.orth2frac_tensor.T, 1
        )  # [N_batch, N_atoms, N_dim=3]

        self.Fprotein_asu_batch = F_protein_batch(
            self.Hasu_array,
            self.dr2asu_array,
            self.fullsf_tensor,
            self.R_G_tensor_stack,
            self.T_G_tensor_stack,
            self.orth2frac_tensor,
            atom_pos_frac_batch,
            self.atom_b_iso,
            self.atom_aniso_uw,
            self.atom_occ,
            PARTITION=PARTITION,
        )  # [N_batch, N_Hasus]

        if not self.HKL_array is None:
            # type: ignore
            self.Fprotein_HKL_batch = self.Fprotein_asu_batch[:, self.asu2HKL_index]
            self.Fmask_HKL_batch = torch.zeros_like(self.Fprotein_HKL_batch)
            if Return:
                return self.Fprotein_HKL_batch
        else:
            self.Fmask_asu_batch = torch.zeros_like(self.Fprotein_asu_batch)
            if Return:
                return self.Fprotein_asu_batch

    def calc_fsolvent_batch(
        self,
        solventpct=None,
        gridsize=None,
        dmin_mask=5,
        dmin_nonzero=3.0,
        exponent=10.0,
        Return=False,
        PARTITION=10,
    ):
        """
        Should run after Calc_Fprotein_batch, calculate the solvent mask structure factors in batched manner
        most parameters are similar to `Calc_Fmask`

        PARTITION: Int, default 10
            To reduce the memory cost during the computation, we divide the batch into several partitions and loops through them.
            Larger PARTITION will require larger GPU memory. Default 10 will take around 2GB, if gridsize=[160,160,160].
            Larger PARTITION does not mean smaller wall time, the optimal size is around 10
        """

        if solventpct is None:
            assert (
                self.inspected
            ), "Run inspect_data first or give a valid solvent percentage!"
            solventpct = self.solventpct

        if gridsize is None:
            assert self.inspected, "Run inspect_data first or give a valid grid size!"
            gridsize = self.gridsize

        Hp1_array, Fp1_tensor_batch = expand_to_p1(
            self.space_group,
            self.Hasu_array,
            self.Fprotein_asu_batch,
            dmin_mask=dmin_mask,
            Batch=True,
            unitcell=self.unit_cell,
        )

        batchsize = self.Fprotein_asu_batch.shape[0]  # type: ignore
        N_partition = batchsize // PARTITION + 1
        Fmask_batch = 0.0

        if not self.HKL_array is None:
            HKL_array = self.HKL_array
        else:
            HKL_array = self.Hasu_array

        for j in range(N_partition):
            if j * PARTITION >= batchsize:
                continue
            start = j * PARTITION
            end = min((j + 1) * PARTITION, batchsize)
            # Shape [N_batch, *gridsize]
            rs_grid = reciprocal_grid(
                Hp1_array, Fp1_tensor_batch[start:end], gridsize, end - start
            )
            real_grid_mask = rsgrid2realmask(
                rs_grid, solvent_percent=solventpct, exponent=exponent, Batch=True
            )  # type: ignore
            Fmask_batch_j = realmask2Fmask(real_grid_mask, HKL_array, end - start)
            if j == 0:
                Fmask_batch = Fmask_batch_j
            else:
                # Shape [N_batches, N_HKLs]
                Fmask_batch = torch.concat(
                    (Fmask_batch, Fmask_batch_j), dim=0
                )  # type: ignore

        if not self.HKL_array is None:
            zero_hkl_bool = torch.tensor(self.dHKL <= dmin_nonzero, device=self.device)
            Fmask_batch[:, zero_hkl_bool] = torch.tensor(
                0.0, device=self.device, dtype=torch.complex64
            )  # type: ignore
            self.Fmask_HKL_batch = Fmask_batch
            if Return:
                return self.Fmask_HKL_batch
        else:
            zero_hkl_bool = torch.tensor(self.dHasu <= dmin_nonzero, device=self.device)
            Fmask_batch[:, zero_hkl_bool] = torch.tensor(
                0.0, device=self.device, dtype=torch.complex64
            )  # type: ignore
            self.Fmask_asu_batch = Fmask_batch
            if Return:
                return self.Fmask_asu_batch

    def _calc_ftotal_batch_bini(self, bin_i, index_i, HKL_array, Fprotein, Fmask):
        """
        calculate ftotal for bin i
        """
        scaled_fmask_i = Fmask[:, index_i] * self.kmasks[bin_i]
        fmodel_i = (
            self.kisos[bin_i]
            * aniso_scaling(
                self.uanisos[bin_i],
                self.reciprocal_cell_paras,
                HKL_array[index_i],
            )
            * (Fprotein[:, index_i] + scaled_fmask_i)
        )
        return fmodel_i

    def calc_ftotal_batch(self, bins=None, Return=True):
        """
        Calculate Ftotal = kiso * exp(-2*pi^2*s^T*Uaniso*s) * (Fprotein + kmask * Fmask)

        Parameters
        ----------
        bins: None or List[int], default None
            Specify which resolution bins to calculate the ftotal, if None, calculate for all

        Return: Boolean, default True
            Whether to return the results

        Returns
        -------
        torch.tensor, complex

        """
        if bins is None:
            bins = range(self.n_bins)
        if not self.HKL_array is None:
            ftotal_hkl_batch = torch.zeros_like(self.Fprotein_HKL_batch)
            for bin_i in bins:
                index_i = self.bins == bin_i
                ftotal_hkl_batch[:, index_i] = self._calc_ftotal_batch_bini(
                    bin_i,
                    index_i,
                    self.HKL_array,
                    self.Fprotein_HKL_batch,
                    self.Fmask_HKL_batch,
                )
            self.Ftotal_HKL_batch = ftotal_hkl_batch
            if Return:
                return ftotal_hkl_batch
        else:
            ftotal_asu_batch = torch.zeros_like(self.Fprotein_asu_batch)
            for bin_i in bins:
                index_i = self.bins == bin_i
                ftotal_asu_batch[index_i] = self._calc_ftotal_batch_bini(
                    bin_i, index_i, self.Hasu_array, self.Fprotein_asu_batch, self.Fmask_asu_batch
                )
            self.Ftotal_asu_batch = ftotal_asu_batch
            if Return:
                return ftotal_asu_batch

    def prepare_dataset(self, HKL_attr: str, F_attr: str):
        """
        create a rs.Dataset with HKL list and complex structure factor
        """
        F_out = getattr(self, F_attr)
        HKL_out = getattr(self, HKL_attr)
        assert len(F_out) == len(
            HKL_out
        ), "HKL and structural factor should have same length!"
        F_out_mag = torch.abs(F_out)
        PI_on_180 = 0.017453292519943295
        F_out_phase = torch.angle(F_out) / PI_on_180
        dataset = rs.DataSet(
            spacegroup=self.space_group, cell=self.unit_cell
        )  # type: ignore
        dataset["H"] = HKL_out[:, 0]
        dataset["K"] = HKL_out[:, 1]
        dataset["L"] = HKL_out[:, 2]
        dataset["FMODEL"] = assert_numpy(F_out_mag)
        dataset["PHIFMODEL"] = assert_numpy(F_out_phase)
        dataset.set_index(["H", "K", "L"], inplace=True)
        dataset.infer_mtz_dtypes(inplace=True)
        return dataset
    
    def savePDB(self, outname: str):
        """
        Write out a PDB with current model coordinates
        """
        
        self._pdb.set_positions(assert_numpy(self.atom_pos_orth))
        self._pdb.set_biso(assert_numpy(self.atom_b_iso))
        self._pdb.set_baniso(assert_numpy(self.atom_aniso_uw))
        self._pdb.set_occ(assert_numpy(self.atom_occ))
        
        from datetime import date
        from . import __version__, __author__
        pdb_header = [
            f'REMARK   1 CREATED WITH SFcalculator {__version__}, {date.today()}',
            f'REMARK   1 AUTHOR {__author__}',
        ]
        self._pdb.pdb_header = pdb_header
        self._pdb.savePDB(outname)

def F_protein(
    HKL_array,
    dr2_array,
    fullsf_tensor,
    R_G_tensor_stack,
    T_G_tensor_stack,
    orth2frac_tensor,
    atom_pos_frac,
    atom_b_iso,
    atom_aniso_uw,
    atom_occ,
):
    """
    Calculate Protein Structural Factor from an atomic model

    atom_pos_frac: 2D tensor, [N_atom, N_dim=3]
    """
    # F_calc = sum_Gsum_j{ [f0_sj*DWF*exp(2*pi*i*(h,k,l)*(R_G*(x1,x2,x3)+T_G))]} fractional postion, Rupp's Book P279
    # G is symmetry operations of the spacegroup and j is the atoms
    # DWF is the Debye-Waller Factor, has isotropic and anisotropic version, based on the PDB file input, Rupp's Book P641
    HKL_tensor = torch.tensor(HKL_array, dtype=torch.float32, device=fullsf_tensor.device)

    oc_sf = fullsf_tensor * atom_occ[..., None]  # [N_atom, N_HKLs]
    # DWF calculator
    dwf_iso = DWF_iso(atom_b_iso, dr2_array)  # [N_atoms, N_HKLs]
    mask_vec = torch.all(torch.all(atom_aniso_uw == 0.0, dim=-1), dim=-1)
    # Vectorized phase calculation
    # sym_oped_pos_frac = (
    #     torch.permute(torch.tensordot(R_G_tensor_stack, atom_pos_frac.T, 1), [2, 0, 1])
    #     + T_G_tensor_stack
    # )
    # Shape [N_atom, N_op, N_dim=3]
    sym_oped_pos_frac = (
        torch.einsum("oxy,ay->aox", R_G_tensor_stack, atom_pos_frac) + T_G_tensor_stack
    )
    sym_oped_hkl = torch.einsum("rx,oxy->roy", HKL_tensor, R_G_tensor_stack)
    exp_phase = 0.0
    # Loop through symmetry operations instead of fully vectorization, to reduce the memory cost
    for i in range(sym_oped_pos_frac.size(dim=1)):
        phase_G = (
            2
            * np.pi
            * torch.einsum("ax,rx->ar", sym_oped_pos_frac[:, i, :], HKL_tensor)
        )  # [N_atom, N_HKLs]
        dwf_aniso = DWF_aniso(
            atom_aniso_uw, orth2frac_tensor, sym_oped_hkl[:, i, :]
        )  # [N_atom, N_HKLs]
        dwf_all = torch.where(mask_vec[:, None], dwf_iso, dwf_aniso)
        exp_phase = exp_phase + dwf_all * torch.exp(1j * phase_G)
    F_calc = torch.sum(exp_phase * oc_sf, dim=0)
    return F_calc


def F_protein_batch(
    HKL_array,
    dr2_array,
    fullsf_tensor,
    R_G_tensor_stack,
    T_G_tensor_stack,
    orth2frac_tensor,
    atom_pos_frac_batch,
    atom_b_iso,
    atom_aniso_uw,
    atom_occ,
    PARTITION=20,
):
    """
    Calculate Protein Structural Factor from a batch of atomic models

    atom_pos_frac_batch: 3D tensor, [N_batch, N_atoms, N_dim=3]

    TODO: Support batched B factors
    """
    # F_calc = sum_Gsum_j{ [f0_sj*DWF*exp(2*pi*i*(h,k,l)*(R_G*(x1,x2,x3)+T_G))]} fractional postion, Rupp's Book P279
    # G is symmetry operations of the spacegroup and j is the atoms
    # DWF is the Debye-Waller Factor, has isotropic and anisotropic version, based on the PDB file input, Rupp's Book P641
    HKL_tensor = torch.tensor(HKL_array).to(R_G_tensor_stack)
    batchsize = atom_pos_frac_batch.shape[0]

    oc_sf = fullsf_tensor * atom_occ[..., None]  # [N_atom, N_HKLs]
    dwf_iso = DWF_iso(atom_b_iso, dr2_array)
    mask_vec = torch.all(torch.all(atom_aniso_uw == 0.0, dim=-1), dim=-1)
    # Vectorized phase calculation
    # sym_oped_pos_frac = (
    #     torch.tensordot(
    #         atom_pos_frac_batch, torch.permute(R_G_tensor_stack, [2, 1, 0]), 1
    #     )
    #     + T_G_tensor_stack.T
    # )
    # Shape [N_batch, N_atom, N_dim=3, N_ops]
    sym_oped_pos_frac = (
        torch.einsum("bay,oxy->baxo", atom_pos_frac_batch, R_G_tensor_stack)
        + T_G_tensor_stack.T
    )
    sym_oped_hkl = torch.einsum("rx,oxy->roy", HKL_tensor, R_G_tensor_stack)
    N_ops = R_G_tensor_stack.shape[0]
    N_partition = batchsize // PARTITION + 1
    F_calc = 0.0
    for j in range(N_partition):
        Fcalc_j = 0.0
        if j * PARTITION >= batchsize:
            continue
        start = j * PARTITION
        end = min((j + 1) * PARTITION, batchsize)
        for i in range(N_ops):  # Loop through symmetry operations to reduce memory cost
            # [N_atoms, N_HKLs]
            dwf_aniso = DWF_aniso(
                atom_aniso_uw, orth2frac_tensor, sym_oped_hkl[:, i, :]
            )  
            # [N_atoms, N_HKLs]
            dwf_all = torch.where(
                mask_vec[:, None], dwf_iso, dwf_aniso
            ) 
            # Shape [PARTITION, N_atoms, N_HKLs]
            exp_phase_ij = dwf_all * torch.exp(1j * (
                2
                * torch.pi
                * torch.tensordot(
                    sym_oped_pos_frac[start:end, :, :, i], HKL_tensor.T, 1
                )
            ))
            # Shape [PARTITION, N_HKLs], sum over atoms
            Fcalc_ij = torch.sum(exp_phase_ij * oc_sf, dim=1)
            del exp_phase_ij # release the memory
            # Shape [PARTITION, N_HKLs], sum over symmetry operations
            Fcalc_j = Fcalc_j + Fcalc_ij
            del Fcalc_ij
        if j == 0:
            F_calc = Fcalc_j
        else:
            # Shape [N_batches, N_HKLs]
            F_calc = torch.concat((F_calc, Fcalc_j), dim=0)  # type: ignore
    return F_calc
