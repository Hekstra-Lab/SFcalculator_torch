from __future__ import annotations

import gemmi
import urllib.request, os

import numpy as np
from tqdm import tqdm
import pandas as pd
from loguru import logger

from .utils import assert_numpy


def hier2array(structure):
    """
    Convert the hierachical gemmi.structure into arrays of info

    structure: gemmi.Structure
    """
    atom_name = []
    cra_name = []
    atom_pos = []
    atom_b_aniso = []
    atom_b_iso = []
    atom_occ = []
    atom_altloc = []
    res_id = []
    i = 0
    model = structure[i]  # gemmi.Model object
    while len(model) == 0:  # Some PDB file has the empty first model
        i += 1
        try:
            model = structure[i]
        except:
            raise ValueError("Can't read valid model from the input PDB file!")
    j = 0
    for chain in model:
        for res in chain:
            for atom in res:
                # A list of atom name like ['O','C','N','C', ...], [Nc]
                atom_name.append(atom.element.name)
                # A list of atom long name like ['A-GLU-5-OD', ...], [Nc]
                cra_name.append(
                    chain.name + "-" + str(j) + "-" + res.name + "-" + atom.name
                )
                # A list of atom's Positions in orthogonal space, [Nc,3]
                atom_pos.append(atom.pos.tolist())
                # A list of anisotropic B Factor matrix, [Nc,3,3]
                atom_b_aniso.append(atom.aniso.as_mat33().tolist())
                # A list of isotropic B Factor [B1,B2,...], [Nc]
                atom_b_iso.append(atom.b_iso)
                # A list of occupancy [P1,P2,....], [Nc]
                atom_occ.append(atom.occ)
                # A list of altloc label
                atom_altloc.append(atom.altloc)
                # A list of residue id
                res_id.append(res.seqid.num)
            j += 1
    atom_pos = np.array(atom_pos)
    atom_b_aniso = np.array(atom_b_aniso)
    atom_b_iso = np.array(atom_b_iso)
    atom_occ = np.array(atom_occ)
    return atom_pos, atom_b_aniso, atom_b_iso, atom_occ, atom_name, cra_name, atom_altloc, res_id


def array2hier(
    atom_pos, atom_b_aniso, atom_b_iso, atom_occ, atom_name, cra_name, atom_altloc, res_id
):
    new_model = gemmi.Model("SFC")
    for i in range(len(cra_name)):
        Chain_i, resnum_i, resname_i, atomname_i = cra_name[i].split("-")
        if i == 0:
            current_chain = gemmi.Chain(Chain_i)
            current_res = gemmi.Residue()
            current_res.name = resname_i
            current_res.seqid = gemmi.SeqId(str(res_id[i]))
        else:
            if res_id[i] != current_res.seqid.num:
                current_chain.add_residue(current_res.clone())
                current_res = gemmi.Residue()
                current_res.name = resname_i
                current_res.seqid = gemmi.SeqId(str(res_id[i]))
            if Chain_i != current_chain.name:
                new_model.add_chain(current_chain.clone())
                current_chain = gemmi.Chain(Chain_i)

        current_atom = gemmi.Atom()
        current_atom.name = atomname_i
        current_atom.element = gemmi.Element(atom_name[i])
        Ui = atom_b_aniso[i]
        current_atom.aniso = gemmi.SMat33f(*[Ui[0,0], Ui[1,1], Ui[2,2], Ui[0,1], Ui[0,2], Ui[1,2]])
        current_atom.b_iso = atom_b_iso[i]
        current_atom.pos = gemmi.Position(*atom_pos[i])
        current_atom.occ = atom_occ[i]
        current_atom.altloc = atom_altloc[i]

        current_res.add_atom(current_atom)

    current_chain.add_residue(current_res.clone())
    new_model.add_chain(current_chain.clone())

    new_structure = gemmi.Structure()
    new_structure.add_model(new_model)
    new_structure.setup_entities()
    return new_structure


class PDBParser(object):
    """
    Read in the pdb file, and save atom name, atom positions, atom Biso, atom Baniso, atom occupancy in array manner
    Suppport indexing and gemmi-syntax structure selection
    """

    def __init__(self, data):
        """
        Create an PDBparser object from pbdfile

        data: pdb file path, or gemmi.Structure
        """
        if isinstance(data, str):
            if data.endswith("pdb"):
                structure = gemmi.read_pdb(data)
            elif data.endswith("cif"):
                structure = gemmi.make_structure_from_block(gemmi.cif.read(data)[0])
        elif isinstance(data, gemmi.Structure):
            structure = data
        else:
            raise KeyError(
                "data should be path str to a pdb file or a gemmi.Structure object"
            )
        (
            self.atom_pos,
            self.atom_b_aniso,
            self.atom_b_iso,
            self.atom_occ,
            self.atom_name,
            self.cra_name,
            self.atom_altloc,
            self.res_id,
        ) = hier2array(structure)
        try:
            self.spacegroup = gemmi.SpaceGroup(structure.spacegroup_hm)
        except:
            self.spacegroup = gemmi.SpaceGroup("P 1")
            logger.info("No valid spacegroup in the file, set as P 1", flush=True)
        self.cell = structure.cell

        # Save the pdb headers, exclude the CRYST1 line
        header = structure.make_pdb_headers().split("\n")
        not_cryst = ["CRYST1" not in i for i in header]
        self.pdb_header = [header[i] for i in range(len(header)) if not_cryst[i]]

    @property
    def sequence(self):
        """
        Get one code squence
        """
        seen = set()
        unique_cras = [x for x in self.cra_name if x not in seen and not seen.add(x)]
        CA_cras = [i for i in unique_cras if "CA" in i]
        sequence = list(map(lambda x: x.split('-')[2], CA_cras))
        sequence = "".join([gemmi.find_tabulated_residue(r).one_letter_code for r in sequence])
        return sequence

    def to_gemmi(self, include_header=True):
        """
        Convert the array data to gemmi.Structure
        """
        st = array2hier(
            self.atom_pos,
            self.atom_b_aniso,
            self.atom_b_iso,
            self.atom_occ,
            self.atom_name,
            self.cra_name,
            self.atom_altloc,
            self.res_id,
        )
        st.spacegroup_hm = self.spacegroup.hm
        st.cell = self.cell
        if include_header:
            # I cheat here
            # In gemmi there is no method to set the whole header
            # because gemmi will parse all info in pdb headers except REMARKS
            # and save in corresponding attributes
            # Here a new gemmi.Structure object will be empty in all the attributes
            # So I set the raw_remarks to be the whole header
            # Next time user parse the new pdb with gemmi, will have all the info again
            st.raw_remarks = self.pdb_header
        return st
    
    @property
    def atom_pos_frac(self):
        return self.orth2frac(self.atom_pos)
    
    @property
    def operations(self):
        return self.spacegroup.operations() 

    @property
    def R_G_stack(self):
        """
        [n_ops, 3, 3], np.ndarray
        """
        return np.array([np.array(sym_op.rot) / sym_op.DEN for sym_op in self.operations])
    
    @property
    def T_G_stack(self):
        """
        [n_ops, 3], np.ndarray
        """
        return np.array([np.array(sym_op.tran) / sym_op.DEN for sym_op in self.operations])
    
    def exp_sym(self, frac_pos: np.ndarray | None = None) -> np.ndarray:
        """
        Apply all symmetry operations to the fractional coordinates

        Args:
            frac_pos, np.ndarray, [n_points, 3]
                fractional coordinates of model in single ASU. 
                If not given, will use self.atom_pos_frac
        Returns:
            np.ndarray, [n_points, n_ops, 3]
                fractional coordinates of symmetry operated models
        """
        if frac_pos is None:
            frac_pos = self.atom_pos_frac
        sym_oped_frac_pos = np.einsum("oxy,ay->aox", self.R_G_stack, frac_pos) + self.T_G_stack
        return sym_oped_frac_pos

    def set_spacegroup(self, spacegroup):
        """
        spacegroup: H-M notation of spacegroup or gemmi.SpaceGroup
        """
        if isinstance(spacegroup, str):
            self.spacegroup = gemmi.SpaceGroup(spacegroup)
        elif isinstance(spacegroup, gemmi.SpaceGroup):
            self.spacegroup = spacegroup

    def set_unitcell(self, unitcell):
        """
        unitcell: gemmi.UnitCell
        """
        self.cell = unitcell

    def set_positions(self, positions):
        """
        Set the atom positions with an array

        positions: array-like, [Nc, 3]
        """
        assert len(positions) == len(self.atom_pos), "Different atom number!"
        assert len(positions[0]) == 3, "Provide 3D coordinates!"
        self.atom_pos = positions

    def set_biso(self, biso):
        """
        Set the B-factors with an array

        biso: array-like, [Nc,]
        """
        assert len(biso) == len(self.atom_b_iso), "Different atom number!"
        assert biso[0].size == 1, "Provide one biso per atom!"
        self.atom_b_iso = biso

    def set_baniso(self, baniso):
        """
        Set the Anisotropic B-factors with an array

        baniso: array-like, [Nc,3,3]
        """
        assert len(baniso) == len(self.atom_b_aniso), "Different atom number!"
        assert baniso[0].shape == (3,3), "Provide a 3*3 matrix per atom!"
        self.atom_b_aniso = baniso

    def set_occ(self, occ):
        """
        Set the occupancy with an array

        occ: array-like, [Nc,]
        """
        assert len(occ) == len(self.atom_occ), "Different atom number!"
        assert occ[0].size == 1, "Provide one occupancy per atom!"
        self.atom_occ = occ

    def selection(self, condition, inplace=False):
        """
        Do structure selection with gemmi syntax

        condition: str, gemmi selection syntax, see https://gemmi.readthedocs.io/en/latest/analysis.html#selections
        """
        selection = gemmi.Selection(condition)
        structure = self.to_gemmi(include_header=False)
        sele_st = selection.copy_structure_selection(structure)

        if inplace:
            (
                self.atom_pos,
                self.atom_b_aniso,
                self.atom_b_iso,
                self.atom_occ,
                self.atom_name,
                self.cra_name,
                self.atom_altloc,
                self.res_id,
            ) = hier2array(sele_st)
        else:
            new_parser = PDBParser(sele_st)
            new_parser.pdb_header = self.pdb_header
            return new_parser

    def from_atom_slices(self, atom_slices, inplace=False):
        """
        Do selection or order change with atom slices

        atom_slices: array_like, index of the atoms you are interested in
        """
        if inplace:
            self.atom_pos = self.atom_pos[atom_slices]
            self.atom_b_aniso = self.atom_b_aniso[atom_slices]
            self.atom_b_iso = self.atom_b_iso[atom_slices]
            self.atom_occ = self.atom_occ[atom_slices]
            self.atom_name = [self.atom_name[i] for i in atom_slices]
            self.cra_name = [self.cra_name[i] for i in atom_slices]
            self.atom_altloc = [self.atom_altloc[i] for i in atom_slices]
            self.res_id = [self.res_id[i] for i in atom_slices]
        else:
            st = array2hier(
                self.atom_pos[atom_slices],
                self.atom_b_aniso[atom_slices],
                self.atom_b_iso[atom_slices],
                self.atom_occ[atom_slices],
                [self.atom_name[i] for i in atom_slices],
                [self.cra_name[i] for i in atom_slices],
                [self.atom_altloc[i] for i in atom_slices],
                [self.res_id[i] for i in atom_slices],
            )
            st.spacegroup_hm = self.spacegroup.hm
            st.cell = self.cell
            new_parser = PDBParser(st)
            new_parser.pdb_header = self.pdb_header
            return new_parser
        
    def move2cell(self):
        """
        move the current model into the cell by shifting
        """
        frac_mat = np.array(self.cell.fractionalization_matrix.tolist())
        mean_positions_frac = np.dot(frac_mat, np.mean(assert_numpy(self.atom_pos), axis=0))
        shift_vec = np.dot(np.linalg.inv(frac_mat), mean_positions_frac % 1.0 - mean_positions_frac)
        self.set_positions(assert_numpy(self.atom_pos) + shift_vec)

    def orth2frac(self, orth_pos: np.ndarray) -> np.ndarray:
        """
        Convert orthogonal coordinates to fractional coordinates

        Args:
            orth_pos: np.ndarray, [n_points, ..., 3]
        
        Returns:
            frational coordinates, np.ndarray, [n_points, ..., 3]
        """
        orth2frac_mat = np.array(self.cell.fractionalization_matrix.tolist())
        frac_pos = np.einsum("n...x,yx->n...y", orth_pos, orth2frac_mat)
        return frac_pos
    
    def frac2orth(self, frac_pos: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates to orthogonal coordinates

        Args:
            frac_pos: np.ndarray, [n_points, ..., 3]
        
        Returns:
            orthogonal coordinates, np.ndarray, [n_points, ..., 3]
        """
        frac2orth_mat = np.array(self.cell.orthogonalization_matrix.tolist())
        orth_pos = np.einsum("n...x,yx->n...y", frac_pos, frac2orth_mat)
        return orth_pos
    
    def savePDB(self, savefilename, include_header=True):
        structure = self.to_gemmi(include_header=include_header)
        structure.write_pdb(savefilename)
    
    def saveCIF(self, savefilename, include_header=True):
        structure = self.to_gemmi(include_header=include_header)
        structure.make_mmcif_block().write_file(savefilename)


def fetch_pdb(idlist, outpath):
    '''
    Fetch pdb and mtz files from Protein Data Bank, with static urllib

    Parameters
    ----------
    idlist : [str]
        List of PDB ids
    
    outpath : str

    Returns
    -------
    DataFrame of fetch stats

    pdb files will be saved at outpath/models/
    mtz files will be saved at outpath/reflections/
    Record csv file will be saved at outpath/fetchpdb.csv
    '''

    if len(idlist) > 1:
        sequence_path = os.path.join(outpath, 'sequences/')
        model_path = os.path.join(outpath, 'model_pdbs/')
        mmcif_path = os.path.join(outpath, 'model_mmcifs/')
        sfcif_path = os.path.join(outpath, 'model_sfcifs/')
        reflection_path = os.path.join(outpath, 'reflections/')
        for folder in [sequence_path, model_path, mmcif_path, sfcif_path, reflection_path]:
            if os.path.exists(folder):
                logger.info(f"{folder:<80}" + f"{'already exists': >20}")
            else:
                os.makedirs(folder)
                logger.info(f"{folder:<80}" + f"{'created': >20}")
    else:
        sequence_path = outpath
        model_path = outpath
        mmcif_path = outpath
        sfcif_path = outpath
        reflection_path = outpath
    
    codes = []
    with_sequence = []
    with_pdb = []
    with_mmcif = []
    with_sfcif = []
    with_mtz = []
    for pdb_code in tqdm(idlist):
        valid_code = pdb_code.lower()
        seqlink = "https://www.rcsb.org/fasta/entry/" + valid_code.upper()
        pdblink = "https://files.rcsb.org/download/" + valid_code.upper() + ".pdb"
        mmciflink = "https://files.rcsb.org/download/" + valid_code.upper() + ".cif"
        sfciflink = "https://files.rcsb.org/download/" + valid_code.upper() + "-sf.cif"
        mtzlink = "https://edmaps.rcsb.org/coefficients/" + valid_code + ".mtz"
        codes.append(valid_code)

        try:
            urllib.request.urlretrieve(seqlink, os.path.join(sequence_path, valid_code+".fasta"))
            with_sequence.append(1)
        except:
            with_sequence.append(0)

        try:
            urllib.request.urlretrieve(pdblink, os.path.join(model_path, valid_code+".pdb"))
            with_pdb.append(1)
        except:
            with_pdb.append(0)

        try:
            urllib.request.urlretrieve(mmciflink, os.path.join(mmcif_path, valid_code+".cif"))
            with_mmcif.append(1)
        except:
            with_mmcif.append(0)

        try:
            urllib.request.urlretrieve(sfciflink, os.path.join(sfcif_path, valid_code+"-sf.cif"))
            with_sfcif.append(1)
        except:
            with_sfcif.append(0)

        try:
            urllib.request.urlretrieve(mtzlink, os.path.join(reflection_path, valid_code+".mtz"))
            with_mtz.append(1)
        except:
            with_mtz.append(0)
    
    stat_df = pd.DataFrame({
        "code" : codes,
        "with_sequence" : with_sequence,
        "with_pdb" : with_pdb,
        "with_mmcif" : with_mmcif,
        "with_sfcif" : with_sfcif,
        "with_mtz" : with_mtz
    })
    stat_df.to_csv(os.path.join(outpath, "fetchpdb.csv"))
    return stat_df


def fetch_pdbredo(idlist, outpath):
    '''
    Fetch re-refined and rebuilt moldes PDB-REDO, with static urllib
    see https://pdb-redo.eu/download for a full list of entry description for future development

    Parameters
    ----------
    idlist : [str]
        List of PDB ids
    
    outpath : str

    Returns
    -------
    DataFrame of fetch stats

    files will be saved at outpath/pdbredo_db/<pdb-id>/xxx
    '''

    codes = []
    with_pdb = []
    with_cif = []
    with_mtz = []
    with_version = []
    for pdb_code in tqdm(idlist):
        
        valid_code = pdb_code.lower()
        parent_link = f"https://pdb-redo.eu/db/{valid_code}/"
        pdblink = parent_link + f"{valid_code}_final.pdb"
        ciflink = parent_link + f"{valid_code}_final.cif"
        mtzlink = parent_link + f"{valid_code}_final.mtz"
        versionlink = parent_link + "versions.json"
        codes.append(valid_code)
        
        temp_path = os.path.join(outpath, "pdbredo_db", valid_code)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        try:
            urllib.request.urlretrieve(pdblink, os.path.join(temp_path, f"{valid_code}_final.pdb"))
            with_pdb.append(1)
        except:
            with_pdb.append(0)

        try:
            urllib.request.urlretrieve(ciflink, os.path.join(temp_path, f"{valid_code}_final.cif"))
            with_cif.append(1)
        except:
            with_cif.append(0)

        try:
            urllib.request.urlretrieve(mtzlink, os.path.join(temp_path, f"{valid_code}_final.mtz"))
            with_mtz.append(1)
        except:
            with_mtz.append(0)

        try:
            urllib.request.urlretrieve(versionlink, os.path.join(temp_path, f"{valid_code}_versions.json"))
            with_version.append(1)
        except:
            with_version.append(0)
    
    stat_df = pd.DataFrame({
        "code" : codes,
        "with_pdb" : with_pdb,
        "with_cif" : with_cif,
        "with_mtz" : with_mtz,
        "with_version" : with_version,
    })
    stat_df.to_csv(os.path.join(outpath, "fetchpdbredo.csv"))
    return stat_df






