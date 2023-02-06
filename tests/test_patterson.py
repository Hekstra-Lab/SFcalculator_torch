import numpy as np
import torch

import reciprocalspaceship as rs

import SFC_Torch.patterson as pat
from SFC_Torch.utils import try_gpu

import pytest


def test_Puvw(data_mtz_exp):
    mtz = rs.read_mtz(data_mtz_exp)
    mtz.dropna(axis=0, subset=['FP'], inplace=True)
    
    Fh = mtz['FP'].to_numpy()
    Fh2 = Fh**2
    HKL_array = mtz.get_hkls()
    cell = mtz.cell
    volume = cell.volume

    uvw_frac = pat.uvw_array_frac(cell, 3.0, 5.0, step=0.3)
    Puvw_np = pat.P_uvw_np(uvw_frac, Fh2, HKL_array, volume)

    uvw_tensor = torch.tensor(uvw_frac, dtype=torch.float32, device=try_gpu())
    HKL_tensor = torch.tensor(HKL_array, device=try_gpu())
    Fh_tensor = torch.tensor(Fh, device=try_gpu())
    Fh2_tensor = torch.tensor(Fh2, device=try_gpu())
    Puvw_torch = pat.P_uvw_torch(uvw_tensor, Fh2_tensor, HKL_tensor, volume)

    Pu = pat.Patterson_torch(uvw_frac, Fh_tensor, HKL_array, volume, PARTITION=1000)

    assert np.all(np.isclose(Puvw_np, Puvw_torch.cpu().numpy(), rtol=1e-03))
    assert np.all(np.isclose(Puvw_np, Pu.cpu().numpy(), rtol=1e-03))

@pytest.mark.parametrize("sharpen", [True, False])
@pytest.mark.parametrize("remove_origin", [True, False])
def test_Puvw_batch(data_mtz_exp, sharpen, remove_origin):
    
    mtz = rs.read_mtz(data_mtz_exp)
    mtz.dropna(axis=0, subset=['FP'], inplace=True)
    
    Fh = mtz['FP'].to_numpy()
    Fh2 = Fh**2
    HKL_array = mtz.get_hkls()
    cell = mtz.cell
    volume = cell.volume

    uvw_frac = pat.uvw_array_frac(cell, 3.0, 5.0, step=0.3)
    uvw_tensor = torch.tensor(uvw_frac, dtype=torch.float32, device=try_gpu())
    HKL_tensor = torch.tensor(HKL_array, device=try_gpu())
    Fh_tensor = torch.tensor(Fh, device=try_gpu())
    Fh2_tensor = torch.tensor(Fh2, device=try_gpu())
    
    Fh_batch = Fh_tensor.unsqueeze(0).repeat(10,1)
    Fh2_batch = Fh2_tensor.unsqueeze(0).repeat(10,1)
    
    Puvw_batch = pat.P_uvw_torch_batch(uvw_tensor, Fh2_batch, HKL_tensor, volume)

    Pu = pat.Patterson_torch(uvw_frac, Fh_tensor, HKL_array, volume, PARTITION=1000, sharpen=sharpen, remove_origin=remove_origin)
    Pu_batch = pat.Patterson_torch_batch(uvw_frac, Fh_batch, HKL_array, volume, PARTITION_uvw=1000, PARTITION_batch=5, sharpen=sharpen, remove_origin=remove_origin)

    if not sharpen and not remove_origin:
        assert torch.all(torch.isclose(Pu_batch, Puvw_batch, rtol=1e-03))
    
    assert torch.all(torch.isclose(Pu, Pu_batch[1], rtol=1e-03))










