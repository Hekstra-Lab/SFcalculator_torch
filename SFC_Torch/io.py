import gemmi
import torch
import numpy as np
import urllib.request, os
from tqdm import tqdm
import pandas as pd

from .utils import try_gpu


def hier2array(structure, as_tensor=False):
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
                # A list of anisotropic B Factor [[U11,U22,U33,U12,U13,U23],..], [Nc,6]
                atom_b_aniso.append(atom.aniso.elements_pdb())
                # A list of isotropic B Factor [B1,B2,...], [Nc]
                atom_b_iso.append(atom.b_iso)
                # A list of occupancy [P1,P2,....], [Nc]
                atom_occ.append(atom.occ)
                # A list of residue id
                res_id.append(res.seqid.num)
            j += 1
    if as_tensor:
        atom_pos = torch.tensor(atom_pos, dtype=torch.float32, device=try_gpu())
        atom_b_aniso = torch.tensor(atom_b_aniso, dtype=torch.float32, device=try_gpu())
        atom_b_iso = torch.tensor(atom_b_iso, dtype=torch.float32, device=try_gpu())
        atom_occ = torch.tensor(atom_occ, dtype=torch.float32, device=try_gpu())
    else:
        atom_pos = np.array(atom_pos)
        atom_b_aniso = np.array(atom_b_aniso)
        atom_b_iso = np.array(atom_b_iso)
        atom_occ = np.array(atom_occ)
    return atom_pos, atom_b_aniso, atom_b_iso, atom_occ, atom_name, cra_name, res_id


def array2hier(
    atom_pos, atom_b_aniso, atom_b_iso, atom_occ, atom_name, cra_name, res_id
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
        current_atom.aniso = gemmi.SMat33f(*atom_b_aniso[i])
        current_atom.b_iso = atom_b_iso[i]
        current_atom.pos = gemmi.Position(*atom_pos[i])
        current_atom.occ = atom_occ[i]

        current_res.add_atom(current_atom)

    current_chain.add_residue(current_res.clone())
    new_model.add_chain(current_chain.clone())

    new_structure = gemmi.Structure()
    new_structure.add_model(new_model)
    return new_structure


class PDBParser(object):
    """
    Read in the pdb file, and save atom name, atom positions, atom Biso, atom Baniso, atom occupancy in array manner
    Suppport indexing and gemmi-syntax structure selection
    """

    def __init__(self, data, use_tensor=False):
        """
        Create an PDBparser object from pbdfile

        data: pdb file path, or gemmi.Structure
        """
        if isinstance(data, str):
            structure = gemmi.read_pdb(data)
        elif isinstance(data, gemmi.Structure):
            structure = data
        else:
            raise KeyError(
                "data should be path str to a pdb file or a gemmi.Structure object"
            )
        self.use_tensor = use_tensor
        (
            self.atom_pos,
            self.atom_b_aniso,
            self.atom_b_iso,
            self.atom_occ,
            self.atom_name,
            self.cra_name,
            self.res_id,
        ) = hier2array(structure, self.use_tensor)
        self.spacegroup = gemmi.SpaceGroup(structure.spacegroup_hm)
        self.cell = structure.cell

        # Save the pdb headers, exclude the CRYST1 line
        header = structure.make_pdb_headers().split("\n")
        not_cryst = ["CRYST1" not in i for i in header]
        self.pdb_header = [header[i] for i in range(len(header)) if not_cryst[i]]

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

        baniso: array-like, [Nc,6]
        """
        assert len(baniso) == len(self.atom_b_aniso), "Different atom number!"
        assert len(baniso[0]) == 6, "Provide 6 baniso parameters per atom!"
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
                self.res_id,
            ) = hier2array(sele_st, self.use_tensor)
        else:
            new_parser = PDBParser(sele_st, use_tensor=self.use_tensor)
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
            self.res_id = [self.res_id[i] for i in atom_slices]
        else:
            st = array2hier(
                self.atom_pos[atom_slices],
                self.atom_b_aniso[atom_slices],
                self.atom_b_iso[atom_slices],
                self.atom_occ[atom_slices],
                [self.atom_name[i] for i in atom_slices],
                [self.cra_name[i] for i in atom_slices],
                [self.res_id[i] for i in atom_slices],
            )
            st.spacegroup_hm = self.spacegroup.hm
            st.cell = self.cell
            new_parser = PDBParser(st, use_tensor=self.use_tensor)
            new_parser.pdb_header = self.pdb_header
            return new_parser

    def savePDB(self, savefilename, include_header=True):
        structure = self.to_gemmi(include_header=include_header)
        structure.write_pdb(savefilename)

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
        model_path = os.path.join(outpath, 'models/')
        reflection_path = os.path.join(outpath, 'reflections/')
        for folder in [sequence_path, model_path, reflection_path]:
            if os.path.exists(folder):
                print(f"{folder:<80}" + f"{'already exists': >20}")
            else:
                os.makedirs(folder)
                print(f"{folder:<80}" + f"{'created': >20}")
    else:
        sequence_path = outpath
        model_path = outpath
        reflection_path = outpath
    
    codes = []
    with_sequence = []
    with_pdb = []
    with_mtz = []
    for pdb_code in tqdm(idlist):
        valid_code = pdb_code.lower()
        seqlink = "https://www.rcsb.org/fasta/entry/" + valid_code.upper()
        pdblink = "https://files.rcsb.org/download/" + valid_code.upper() + ".pdb"
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
            urllib.request.urlretrieve(mtzlink, os.path.join(reflection_path, valid_code+".mtz"))
            with_mtz.append(1)
        except:
            with_mtz.append(0)
    
    stat_df = pd.DataFrame({
        "code" : codes,
        "with_sequence" : with_sequence,
        "with_pdb" : with_pdb,
        "with_mtz" : with_mtz
    })
    stat_df.to_csv(os.path.join(outpath, "fetchpdb.csv"))
    return stat_df







