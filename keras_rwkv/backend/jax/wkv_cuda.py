import typing as tp
import jax.numpy as jnp


def wkv(
    k: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    u: jnp.ndarray,
    current_index: tp.Optional[tp.Union[int, jnp.ndarray]] = None,
) -> tp.Tuple[jnp.ndarray, tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    raise NotImplementedError("wkv_cuda currently only implemented with torch backend")
    # B, T, C = k.shape
    # aa = torch.zeros(size=(B, C), dtype=k.dtype, device=k.device)
    # bb = torch.zeros(size=(B, C), dtype=k.dtype, device=k.device)
    # pp = torch.zeros(size=(B, C), dtype=k.dtype, device=k.device)
    # y = torch.empty_like(k, memory_format=torch.contiguous_format)
    # torch.ops.rwkv.wkv_forward(
    #     B,
    #     T,
    #     C,
    #     -w.contiguous(),
    #     u.contiguous(),
    #     k.contiguous(),
    #     v.contiguous(),
    #     y,
    #     aa,
    #     bb,
    #     pp,
    # )
    # return y, (aa, bb, pp)


def wkv_update(
    k: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    u: jnp.ndarray,
    state: tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tp.Tuple[jnp.ndarray, tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    raise NotImplementedError("wkv_cuda currently only implemented with torch backend")
    # k = k.squeeze(1)  # [B, C]
    # v = v.squeeze(1)  # [B, C]
    # aa, bb, pp = state  # [B, C]

    # ww = u + k
    # p = torch.maximum(pp, ww)
    # e1 = torch.exp(pp - p)
    # e2 = torch.exp(ww - p)
    # a = e1 * aa + e2 * v
    # b = e1 * bb + e2
    # result = a / b

    # # update state
    # ww = pp - w
    # p = torch.maximum(ww, k)
    # e1 = torch.exp(ww - p)
    # e2 = torch.exp(k - p)
    # aa = e1 * aa + e2 * v
    # bb = e1 * bb + e2
    # pp = p

    # return result.unsqueeze(1), (aa, bb, pp)
