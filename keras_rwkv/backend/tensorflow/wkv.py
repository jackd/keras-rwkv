import typing as tp
import tensorflow as tf
from . import ew


def wkv(
    k: tf.Tensor,
    v: tf.Tensor,
    w: tf.Tensor,
    u: tf.Tensor,
    current_index: tp.Optional[tp.Union[int, tf.Tensor]] = None,
    parallel: bool = True,
) -> tp.Tuple[tf.Tensor, tp.Tuple[tf.Tensor, tf.Tensor]]:
    """
    Args:
        k: [B, T, C]
        v: [B, T, C]
        w: [C]
        u: [C]

    Returns:
        [B, T, C]
    """
    sequence_length = tf.shape(k)[1]
    k = k + tf.range(-sequence_length + 1, 1, dtype=k.dtype)[:, None] * w

    v_acc, t_acc = ew.cumsum(v, k, axis=1, parallel=parallel)

    if current_index is None:
        current_index = sequence_length - 1
    v_next = tf.expand_dims(v_acc[:, current_index], 1)
    t_next = tf.expand_dims(t_acc[:, current_index], 1)
    t_next = t_next + w * tf.cast(sequence_length - 1 - current_index, w.dtype)
    state_next = (v_next, t_next)
    v_acc = v_acc[:, :-1]
    t_acc = t_acc[:, :-1]
    # split values. First goes straight to output, rest are used with lagged acc.
    v_start = v[:, :1]
    v_rest = v[:, 1:]
    v_out, t_out = ew.add(v_acc, t_acc, v_rest, k[:, 1:] + u - w)
    del t_out  # exp(t_out) is the denominator term
    v_out = tf.concat((v_start, v_out), axis=1)
    return v_out, state_next


def wkv_update(
    k: tf.Tensor,
    v: tf.Tensor,
    w: tf.Tensor,
    u: tf.Tensor,
    state: tp.Tuple[tf.Tensor, tf.Tensor],
) -> tp.Tuple[tf.Tensor, tp.Tuple[tf.Tensor, tf.Tensor]]:
    v_acc, t_acc = state
    v_out, t_out = ew.add(v_acc, t_acc, v, k + u)
    del t_out  # exp(t_out) is just the denominator
    state_next = ew.add(v_acc, t_acc - w, v, k)
    return v_out, state_next
