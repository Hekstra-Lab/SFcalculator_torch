import pytest
import tempfile
from os.path import exists

import gemmi
import numpy as np

from SFC_Torch.io import PDBParser, fetch_pdb


def test_setdata(data_pdb):
    a = PDBParser(data_pdb)

    # test set_position
    new_pos = a.atom_pos.copy()
    new_pos[10, 2] = 7.88
    a.set_positions(new_pos)
    assert a.atom_pos[10, 2] == 7.88

    # test set_biso
    new_biso = a.atom_b_iso.copy()
    new_biso[10] = 7.88
    a.set_biso(new_biso)
    assert a.atom_b_iso[10] == 7.88

    # test set_baniso
    new_baniso = a.atom_b_aniso.copy()
    new_baniso[10, 1, 2] = 7.88
    a.set_baniso(new_baniso)
    assert a.atom_b_aniso[10, 1, 2] == 7.88
    
    # test set_occ
    new_occ = a.atom_occ.copy()
    new_occ[10] = 7.88
    a.set_occ(new_occ)
    assert a.atom_occ[10] == 7.88

    # test set_spacegroup
    a.set_spacegroup('I 4')
    assert a.spacegroup.hm == 'I 4'
    a.set_spacegroup(gemmi.SpaceGroup('P 41 3 2'))
    assert a.spacegroup.hm == 'P 41 3 2'

    # test set_unitcell
    a.set_unitcell(gemmi.UnitCell(50, 50, 50, 90, 90, 90))
    assert a.cell.parameters == (50, 50, 50, 90, 90, 90)


def test_savePDB(data_pdb):
    a = PDBParser(data_pdb)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as temp:
        a.savePDB(temp.name, include_header=True)
        assert exists(temp.name)


def test_readcif(data_pdb, data_cif):
    a = PDBParser(data_pdb)
    b = PDBParser(data_cif)
    assert a.cra_name == b.cra_name
    assert (a.atom_pos == b.atom_pos).all()
    assert a.spacegroup.hm == b.spacegroup.hm

def test_saveCIF(data_pdb):
    a = PDBParser(data_pdb)
    with tempfile.NamedTemporaryFile(suffix=".cif") as temp:
        a.saveCIF(temp.name, include_header=True)
        assert exists(temp.name)

@pytest.mark.parametrize("inplace", [True, False])
def test_selection(data_pdb, inplace):
    a = PDBParser(data_pdb)
    b = a.selection('CA:*', inplace=inplace)
    if inplace:
        assert b is None
        assert len(a.atom_pos) == 55
    else:
        assert b.cell == a.cell
        assert b.spacegroup.hm == a.spacegroup.hm
        assert len(b.atom_pos) == 55


@pytest.mark.parametrize("inplace", [True, False])
def test_fromatomslices(data_pdb, inplace):
    a = PDBParser(data_pdb)
    is_CA = ["CA" in i for i in a.cra_name]
    CA_index = np.where(is_CA)[0]

    b = a.from_atom_slices(CA_index, inplace=inplace)
    if inplace:
        assert b is None
        assert len(a.atom_pos) == 55
    else:
        assert b.cell == a.cell
        assert b.spacegroup.hm == a.spacegroup.hm
        assert len(b.atom_pos) == 55

def test_fetchpdb():
    df = fetch_pdb(['4lZt', '1cTS', "1dUr"], outpath='../dev/')
    assert df['code'].tolist() == ['4lzt', '1cts', '1dur']
    assert df['with_pdb'].tolist() == [1, 1, 1]
    assert df['with_mtz'].tolist() == [0, 0, 0]
    assert df['with_sfcif'].tolist() == [1, 0, 1]
    assert df['with_mmcif'].tolist() == [1, 1, 1]
    assert exists("../dev/model_pdbs/4lzt.pdb")
    assert exists("../dev/model_pdbs/1dur.pdb")
    assert exists("../dev/model_pdbs/1cts.pdb")    
    assert exists("../dev/model_mmcifs/1dur.cif")
    assert exists("../dev/model_sfcifs/1dur-sf.cif")