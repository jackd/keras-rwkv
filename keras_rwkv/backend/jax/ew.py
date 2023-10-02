import typing as tp
import jax
import jax.numpy as jnp


def evaluate(v: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Get the value corresponding to the exponentially weighted representation."""
    return jnp.exp(t) * v


def add(
    v1: jnp.ndarray,
    t1: jnp.ndarray,
    v2: jnp.ndarray,
    t2: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Add two exponentially weighted v.

    Args:
        v1: v of the first ew value
        t1: t of the first ew value
        v2: v of the second ew value
        t2: t of the second ew value
    """
    t_out = jnp.logaddexp(t1, t2)
    v1 = evaluate(v1, t1 - t_out)
    v2 = evaluate(v2, t2 - t_out)
    return v1 + v2, t_out


def cumsum_parallel(
    v: jnp.ndarray, t: jnp.ndarray, axis: int = 0, reverse: bool = False
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
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
    return jax.lax.associative_scan(
        lambda a, b: add(*a, *b),
        (v, t),
        axis=axis,
        reverse=reverse,
    )


def cumsum_serial(v: jnp.ndarray, t: jnp.ndarray, axis: int = 0, reverse: bool = False):
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
        v = jnp.swapaxes(v, 0, axis)
        t = jnp.swapaxes(t, 0, axis)

    def f(carry, vt):
        result = add(*carry, *vt)
        return result, result

    shape = v.shape[1:]
    dtype = v.dtype

    _, (v_out, t_out) = jax.lax.scan(
        f, (jnp.zeros(shape, dtype), jnp.full(shape, -jnp.inf)), (v, t), reverse=reverse
    )

    if axis != 0:
        v_out = jnp.swapaxes(v_out, 0, axis)
        t_out = jnp.swapaxes(t_out, 0, axis)

    return v_out, t_out


def cumsum(
    v: jnp.ndarray,
    t: jnp.ndarray,
    axis: int = 0,
    reverse: bool = False,
    parallel: bool = True,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
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
    if parallel:
        return cumsum_parallel(v, t, axis=axis, reverse=reverse)
    return cumsum_serial(v, t, axis=axis, reverse=reverse)
