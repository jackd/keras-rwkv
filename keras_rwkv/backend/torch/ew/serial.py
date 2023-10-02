"""Provides serial cumsum implementation."""

import typing as tp
import numpy as np
import triton
import triton.language as tl
import torch
from . import triton_utils

# pylint:disable=no-member


@triton.jit
def _cumsum_kernel(
    v_in_ptr,
    t_in_ptr,
    v_out_ptr,
    t_out_ptr,
    sequence_length: tl.constexpr,
    num_channels: tl.constexpr,
):
    b = tl.program_id(axis=0)
    c = tl.program_id(axis=1)
    start_id = b * sequence_length * num_channels + c
    v_in_ptr += start_id
    t_in_ptr += start_id
    v_out_ptr += start_id
    t_out_ptr += start_id

    v = tl.load(v_in_ptr)
    t = tl.load(t_in_ptr)
    tl.store(v_out_ptr, v)
    tl.store(t_out_ptr, t)
    # for _ in tl.static_range(1, sequence_length):
    for _ in range(1, sequence_length):
        v_in_ptr += num_channels
        t_in_ptr += num_channels
        v_out_ptr += num_channels
        t_out_ptr += num_channels
        v, t = triton_utils.add(v, t, tl.load(v_in_ptr), tl.load(t_in_ptr))
        tl.store(v_out_ptr, v)
        tl.store(t_out_ptr, t)


def cumsum(
    v: torch.Tensor,
    t: torch.Tensor,
    dim: int,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """cumsum implementation via triton without gradient support."""
    assert v.shape == t.shape, (v.shape, t.shape)
    if dim < 0:
        dim += v.ndim
    shape = v.shape
    # reshape to [B, T, C], even if B or C become 1.
    leading = 1
    for d in shape[:dim]:
        leading *= d
    sequence_length = shape[dim]
    trailing = 1
    for d in shape[dim + 1 :]:
        trailing *= d
    v = v.reshape(leading, sequence_length, trailing)
    t = t.reshape(leading, sequence_length, trailing)
    v_out = torch.empty_like(v)
    t_out = torch.empty_like(t)

    _cumsum_kernel[(leading, trailing)](
        v,
        t,
        v_out,
        t_out,
        sequence_length=sequence_length,
        num_channels=trailing,
    )
    v_out = v_out.reshape(shape)
    t_out = t_out.reshape(shape)
    return v_out, t_out
