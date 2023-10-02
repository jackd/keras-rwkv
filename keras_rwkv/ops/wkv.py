import typing as tp
from ..backend import Array, ops
from . import ew


def wkv(
    k: Array,
    v: Array,
    w: Array,
    u: Array,
    current_index: tp.Optional[tp.Union[int, Array]] = None,
    parallel: bool = True,
) -> tp.Tuple[Array, ew.ExponentiallyWeighted]:
    """
    Args:
        k: [B, T, C]
        v: [B, T, C]
        w: [C]
        u: [C]

    Returns:
        [B, T, C]
    """
    sequence_length = ops.shape(k)[1]
    k = k + ops.arange(-sequence_length + 1, 1, dtype=k.dtype)[:, None] * w

    v_acc, t_acc = ew.cumsum((v, k), axis=1, parallel=parallel)

    if current_index is None:
        current_index = sequence_length - 1
    v_next = ops.expand_dims(v_acc[:, current_index], 1)
    t_next = ops.expand_dims(t_acc[:, current_index], 1)
    t_next = t_next + w * ops.convert_to_tensor(
        sequence_length - 1 - current_index, dtype=w.dtype
    )
    state_next = ew.ExponentiallyWeighted(v_next, t_next)
    v_acc = v_acc[:, :-1]
    t_acc = t_acc[:, :-1]
    # split values. First goes straight to output, rest are used with lagged acc.
    v_start = v[:, :1]
    v_rest = v[:, 1:]
    v_out, t_out = ew.add((v_acc, t_acc), (v_rest, k[:, 1:] + u - w))
    del t_out  # exp(t_out) is the denominator term
    v_out = ops.concatenate((v_start, v_out), axis=1)
    return v_out, state_next


def wkv_update(
    k: Array,
    v: Array,
    w: Array,
    u: Array,
    state: ew.ExponentiallyWeighted,
) -> tp.Tuple[Array, ew.ExponentiallyWeighted]:
    if not isinstance(state, ew.ExponentiallyWeighted):
        state = ew.ExponentiallyWeighted(*state)
    v_out, t_out = state + ew.ExponentiallyWeighted(v, k + u)
    state = ew.ExponentiallyWeighted(state.v, state.t - w)  # decay current state
    state_next = state + ew.ExponentiallyWeighted(v, k)  #
    del t_out
    return v_out, state_next
