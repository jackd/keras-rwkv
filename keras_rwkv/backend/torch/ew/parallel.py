"""Provides parallel cumsum implementation."""

import typing as tp
import numpy as np
import triton
import triton.language as tl
import torch
from . import triton_utils

# pylint:disable=no-member


def _bitcast_unmerge_torch(merged):
    assert merged.dtype == torch.int64
    b = (merged & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    a = (merged >> 32).to(torch.int32).view(torch.float32)  # shifted by 32 bits
    return a, b


def _bitcast_merge_torch(a, b):
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32
    mask = 0xFFFFFFFF
    a = a.view(dtype=torch.int32).to(torch.int64) & mask  # directly converted to int32
    a = a << 32  # shifted by 32 bits
    b = b.view(dtype=torch.int32).to(torch.int64) & mask  # directly converted to int32
    return a + b


@triton.jit
def _bitcast_unmerge_triton(merged):
    tl.static_assert(merged.dtype == tl.int64)
    b = (merged & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)
    a = (merged >> 32).to(tl.int32).to(tl.float32, bitcast=True)  # shifted by 32 bits
    return a, b


@triton.jit
def _bitcast_merge_triton(a, b):
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    mask = 0xFFFFFFFF
    a = a.to(dtype=tl.int32, bitcast=True).to(tl.int64) & mask
    a = a << 32  # shifted by 32 bits
    b = b.to(dtype=tl.int32, bitcast=True).to(tl.int64) & mask
    return a | b


@triton.jit
def _add_merged(x1: tl.tensor, x2: tl.tensor) -> tl.tensor:
    """unmerge/merge wrapper around wkv_add (v, t)."""
    v1, t1 = _bitcast_unmerge_triton(x1)
    v2, t2 = _bitcast_unmerge_triton(x2)
    v_out, t_out = triton_utils.add(v1, t1, v2, t2)
    x_out = _bitcast_merge_triton(v_out, t_out)
    return x_out


@triton.jit
def _cumsum_kernel(
    merged_ptr,
    result_ptr,
    sequence_length: tl.constexpr,
    valid_sequence_length: tl.constexpr,
    num_channels: tl.constexpr,
):
    b = tl.program_id(axis=0)
    c = tl.program_id(axis=1)
    start_id = b * sequence_length * num_channels + c
    offsets = tl.arange(0, sequence_length)
    mask = offsets < valid_sequence_length
    offsets = offsets * num_channels + start_id
    merged = tl.load(merged_ptr + offsets)
    result = tl.associative_scan(merged, axis=0, combine_fn=_add_merged)
    tl.store(result_ptr + offsets, result, mask=mask)


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
    # leading = np.prod(shape[:dim], initial=1, dtype=int).item()
    leading = 1
    for d in shape[:dim]:
        leading *= d
    sequence_length = shape[dim]
    # trailing = np.prod(shape[dim + 1 :], initial=1, dtype=int).item()
    trailing = 1
    for d in shape[dim + 1 :]:
        trailing *= d
    v = v.reshape(leading, sequence_length, trailing)
    t = t.reshape(leading, sequence_length, trailing)

    # pad dim to power of 2
    padded_sequence_length = triton.next_power_of_2(sequence_length)
    if padded_sequence_length != sequence_length:
        padding = padded_sequence_length - sequence_length
        v = torch.nn.functional.pad(v, (0, 0, 0, padding, 0, 0))
        t = torch.nn.functional.pad(t, (0, 0, 0, padding, 0, 0), value=1)

    merged = _bitcast_merge_torch(v.contiguous(), t.contiguous())
    result = torch.empty_like(merged)
    _cumsum_kernel[(leading, trailing)](
        merged,
        result,
        sequence_length=padded_sequence_length,
        valid_sequence_length=sequence_length,
        num_channels=trailing,
    )
    # remove power of 2 padding
    if padded_sequence_length != sequence_length:
        result = result.contiguous()[:, :sequence_length]
    v, t = _bitcast_unmerge_torch(result)
    # revert to original shape
    v = v.reshape(shape)
    t = t.reshape(shape)
    return v, t
