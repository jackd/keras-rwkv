import functools
import typing as tp
import os

import jax
import jax2torch
import torch

from ..jax import wkv as jax_wkv

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@functools.cache
def _get_wkv(parallel: bool):
    def fn(k, v, w, u, current_index):
        result, state = jax_wkv.wkv(k, v, w, u, current_index, parallel=parallel)
        return result, *state

    return jax2torch.jax2torch(jax.jit(fn))


def wkv(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    current_index: tp.Optional[tp.Union[int, torch.Tensor]] = None,
    parallel: bool = True,
) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
    # return _wkv(k, v, w, u, current_index)
    result, v, t = _get_wkv(parallel)(k, v, w, u, current_index)
    return result, (v, t)


wkv.__doc__ = jax_wkv.wkv.__doc__


def _jax_wkv_update(k, v, w, u, state_v, state_t):
    result, state = jax_wkv.wkv_update(k, v, w, u, (state_v, state_t))
    return result, *state


_wkv_update = jax2torch.jax2torch(jax.jit(_jax_wkv_update))


def wkv_update(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    state: tp.Tuple[torch.Tensor, torch.Tensor],
) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
    result, v, t = _wkv_update(k, v, w, u, *state)
    return result, (v, t)
