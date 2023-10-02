import typing as tp

import jax.numpy as jnp

from . import ew


def wkv(
    k: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    u: jnp.ndarray,
    current_index: tp.Optional[tp.Union[int, jnp.ndarray]] = None,
    *,
    parallel: bool = True
) -> tp.Tuple[jnp.ndarray, tp.Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Args:
        k: [B, T, C]
        v: [B, T, C]
        w: [C]
        u: [C]

    Returns:
        [B, T, C]
    """
    sequence_length = k.shape[1]
    k = k + jnp.arange(-sequence_length + 1, 1)[:, None] * w
    v_acc, t_acc = ew.cumsum(v, k, axis=1, parallel=parallel)

    if current_index is None:
        v_acc, v_next = jnp.split(v_acc, (sequence_length - 1,), axis=1)
        t_acc, t_next = jnp.split(t_acc, (sequence_length - 1,), axis=1)
    else:
        current_index = jnp.where(
            current_index < 0, current_index + sequence_length, current_index
        )
        v_next = v_acc[:, current_index]
        t_next = t_acc[:, current_index]
        t_next += (sequence_length - 1 - current_index) * w  # undo effect of k offset
        v_next = jnp.expand_dims(v_next, 1)
        t_next = jnp.expand_dims(t_next, 1)
        v_acc = v_acc[:, :-1]
        t_acc = t_acc[:, :-1]
    state_next = (v_next, t_next)
    # split values. First goes straight to output, rest are used with lagged acc.
    v_start, v_rest = jnp.split(v, (1,), axis=1)
    # -w below is necessary to accommodate for k offset + lag
    v_out, t_out = ew.add(v_acc, t_acc, v_rest, k[:, 1:] + u - w)
    del t_out  # exp(t_out) is the denominator
    v_out = jnp.concatenate((v_start, v_out), axis=1)
    return v_out, state_next


def wkv_update(
    k: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    u: jnp.ndarray,
    state: tp.Tuple[jnp.ndarray, jnp.ndarray],
) -> tp.Tuple[jnp.ndarray, tp.Tuple[jnp.ndarray, jnp.ndarray]]:
    v_acc, t_acc = state
    v_out, t_out = ew.add(v_acc, t_acc, v, k + u)
    del t_out  # exp(t_out) is the denominator
    state_next = ew.add(v_acc, t_acc - w, v, k)
    return v_out, state_next
