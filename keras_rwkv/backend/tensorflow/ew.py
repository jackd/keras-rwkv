import typing as tp
import tensorflow as tf
from tensorflow.experimental import (  # pylint:disable=no-name-in-module,import-error  # type:ignore
    numpy as tfnp,
)
import tensorflow_probability as tfp


def evaluate(v: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """Get the value corresponding to the exponentially weighted representation."""
    return tf.exp(t) * v


def add(
    v1: tf.Tensor,
    t1: tf.Tensor,
    v2: tf.Tensor,
    t2: tf.Tensor,
) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """
    Add two exponentially weighted v.

    Args:
        v1: v of the first ew value
        t1: t of the first ew value
        v2: v of the second ew value
        t2: t of the second ew value
    """
    t_out = tfnp.logaddexp(t1, t2)
    v1 = evaluate(v1, t1 - t_out)
    v2 = evaluate(v2, t2 - t_out)
    return v1 + v2, t_out


def cumsum_parallel(
    v: tf.Tensor, t: tf.Tensor, axis: int = 0, reverse: bool = False
) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute cumulative summation of exponentially weighted `(v, t)`.

    This implementation is based on `jax.lax.associative_scan`. It is work efficient
    (`O(n)`) though requires `log(n)` steps on `n` threads.

    See `cumsum_serial` for a serial implementation which may be faster if there is
    significant parallelism on other axes which can saturate cores.

    Args:
        v: same shape as t
        t: same shape as v

    Returns:
        (v_out, t_out) structure, same shape/dtype as inputs.
    """
    if reverse:
        v = tfnp.flip(v, axis=axis)
        t = tfnp.flip(t, axis=axis)
    v, t = tfp.math.scan_associative(
        lambda a, b: add(*a, *b),
        (v, t),
        axis=axis,
    )
    if reverse:
        v = tfnp.flip(v, axis=axis)
        t = tfnp.flip(t, axis=axis)
    return v, t


def cumsum_serial(v: tf.Tensor, t: tf.Tensor, axis: int = 0, reverse: bool = False):
    """
    Compute cumulative summation of exponentially weighted `(v, t)`.

    This implementation is based on `jax.lax.scan`. This requires less operations than
    `cumsum_parallel` but is not parallelized over the scan axis. This may be faster if
    threads are saturated (e.g. if there is significant parallelism on other axes).

    See `cumsum_parallel` for a work-efficient parallel implementation.

    Args:
        v: same shape as t
        t: same shape as v

    Returns:
        (v_out, t_out) structure, same shape/dtype as inputs.
    """
    if axis != 0:
        v = tfnp.swapaxes(v, 0, axis)
        t = tfnp.swapaxes(t, 0, axis)

    def f(carry, vt):
        result = add(*carry, *vt)
        return result

    (v_out, t_out) = tf.scan(
        f,
        elems=(v, t),
        reverse=reverse,
        parallel_iterations=1,
    )

    if axis != 0:
        v_out = tfnp.swapaxes(v_out, 0, axis)
        t_out = tfnp.swapaxes(t_out, 0, axis)

    return v_out, t_out


def cumsum(
    v: tf.Tensor,
    t: tf.Tensor,
    axis: int = 0,
    reverse: bool = False,
    parallel: bool = True,
) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute cumulative summation of exponentially weighted `(v, t)`.

    This implementation uses either a parallel or serial implementation, depending on
    argument `parallel`. The parallel implementation requires O(log(n)) steps on O(n)
    threads, while the serial version requires O(n) steps on a single thread. If running
    on an accelerator, `parallel=True` will generally be faster unless that is
    significant parallelism on other axes.

    Args:
        v: same shape as t
        t: same shape as v

    Returns:
        (v_out, t_out) structure, same shape/dtype as inputs.
    """
    return (cumsum_parallel if parallel else cumsum_serial)(
        v, t, axis=axis, reverse=reverse
    )
