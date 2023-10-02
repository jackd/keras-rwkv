import typing as tp
import os
import warnings

import torch

# pylint:disable=no-member

if not os.environ.get("RWKV_CUDA_ON", None):
    os.environ["RWKV_CUDA_ON"] = "1"

if not os.environ.get("RWKV_JIT_ON", None):
    os.environ["RWKV_JIT_ON"] = "1"


def _wkv(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    current_index: tp.Optional[int] = None,
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        import rwkv.model  # pylint:disable=import-outside-toplevel,unused-import
    except ImportError as e:
        raise ImportError(
            "Failed to import `rwkv.model`, required for wkv_cuda."
        ) from e

    B, T, C = k.shape
    aa = torch.zeros(size=(B, 1, C), dtype=k.dtype, device=k.device)
    bb = torch.zeros(size=(B, 1, C), dtype=k.dtype, device=k.device)
    pp = torch.zeros(size=(B, 1, C), dtype=k.dtype, device=k.device)
    y = torch.empty_like(k, memory_format=torch.contiguous_format)
    torch.ops.rwkv.wkv_forward(
        B,
        T,
        C,
        -w.contiguous(),
        u.contiguous(),
        k.contiguous(),
        v.contiguous(),
        y,
        aa,
        bb,
        pp,
    )
    if current_index is None:
        state = (aa, bb, pp)
    else:
        # hacky, but we just re-run from the start
        _, *state = _wkv(k[:, : current_index + 1], v[:, : current_index + 1], w, u)
    return y, *state


class WkvCuda(torch.autograd.Function):  # pylint:disable=abstract-method
    @staticmethod
    def forward(  # pylint:disable=arguments-differ
        ctx,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        current_index: tp.Optional[tp.Union[int, torch.Tensor]],
    ):
        return _wkv(k, v, w, u, current_index)

    @staticmethod
    def backward(ctx, grad_v_out, *grad_state):  # pylint:disable=arguments-differ
        if os.environ.get("USE_RANDOM_WKV_CUDA_GRADIENTS", "0") != "0":
            warnings.warn("Using random wkv_cuda gradients")
            return (
                torch.rand_like(grad_v_out),
                torch.rand_like(grad_v_out),
                None,
                None,
                None,
            )
        raise NotImplementedError("WkvCuda gradients not supported")


def wkv(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    current_index: tp.Optional[int] = None,
) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    y, *state = WkvCuda.apply(k, v, w, u, current_index)
    return y, state


def wkv_update(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    state: tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    aa, bb, pp = state

    ww = u + k
    p = torch.maximum(pp, ww)
    e1 = torch.exp(pp - p)
    e2 = torch.exp(ww - p)
    a = e1 * aa + e2 * v
    b = e1 * bb + e2
    result = a / b

    # update state
    ww = pp - w
    p = torch.maximum(ww, k)
    e1 = torch.exp(ww - p)
    e2 = torch.exp(k - p)
    aa = e1 * aa + e2 * v
    bb = e1 * bb + e2
    pp = p

    return result, (aa, bb, pp)
