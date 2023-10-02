import os
import typing as tp
import functools

from absl import app, flags
import numpy as np
import torch
import tensorflow as tf
import triton
import jax
import jax.numpy as jnp
import torch._dynamo
import jax2torch

from keras_rwkv.backend.jax import wkv as wkv_jax
from keras_rwkv.backend.torch import wkv, wkv_cuda
from keras_rwkv.backend.tensorflow import wkv as wkv_tf


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


torch._dynamo.config.suppress_errors = True

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

flags.DEFINE_integer("d", "32", "hidden dimension")
flags.DEFINE_bool("debug", False, "Show more torch dynamo output")
flags.DEFINE_integer(
    "triton_parallel_threshold",
    -1,
    "threshold above which we don't run triton_parallel implementation",
)
flags.DEFINE_integer(
    "tensorflow_serial_threshold",
    -1,
    "threshold above which we don't run tensorflow_serial implementation",
)
flags.DEFINE_integer("b", 1, "batch size")

FLAGS = flags.FLAGS

kwargs = {
    "x_names": ["sequence_length"],
    "x_vals": [2**i for i in range(10, 20)],
}
line_vals, line_names, styles = (
    list(x)
    for x in zip(
        ["cuda", "cuda (serial)", ("green", "dashed")],
        ["triton-serial", "triton (serial)", ("orange", "dashed")],
        ["triton-parallel", "triton (parallel)", ("orange", "dotted")],
        ["jax2torch-serial", "jax2torch (serial)", ("red", "dashed")],
        ["jax2torch-parallel", "jax2torch (parallel)", ("red", "dotted")],
        ["jax-serial", "jax (serial)", ("blue", "dashed")],
        ["jax-parallel", "jax (parallel)", ("blue", "dotted")],
        ["tf-serial", "tensorflow (serial)", ("black", "dashed")],
        ["tf-parallel", "tensorflow (parallel)", ("black", "dotted")],
    )
)


@functools.cache
def _wkv_jax_fn(parallel: bool):
    return jax.jit(functools.partial(wkv_jax.wkv, parallel=parallel))


@functools.cache
def _wkv_tf_fn(parallel: bool):
    return tf.function(
        functools.partial(wkv_tf.wkv, parallel=parallel), jit_compile=True
    )


@functools.cache
def _wkv_jax2torch(parallel: bool):
    def fn(k, v, w, u):
        result, state = wkv_jax.wkv(k, v, w, u, current_index=None, parallel=parallel)
        return result, *state

    return jax2torch.jax2torch(jax.jit(fn))


wkv_compiled = torch.compile(wkv.wkv)


def create_torch_args(b, t, c):
    shape = (b, t, c)
    k = torch.rand(shape, device="cuda", dtype=torch.float32)
    v = torch.rand(shape, device="cuda", dtype=torch.float32)
    w = torch.rand((c,), device="cuda", dtype=torch.float32)
    u = torch.rand((c,), device="cuda", dtype=torch.float32)
    return k, v, w, u


def create_jax_args(b, t, c):
    shape = (b, t, c)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    k = jax.random.normal(keys[0], shape, dtype=jnp.float32)
    v = jax.random.normal(keys[1], shape, dtype=jnp.float32)
    w = jax.random.normal(keys[2], (c,), dtype=jnp.float32)
    u = jax.random.normal(keys[3], (c,), dtype=jnp.float32)
    return k, v, w, u


def create_tf_args(b, t, c):
    shape = (b, t, c)
    k = tf.random.normal(shape, dtype=tf.float32)
    v = tf.random.normal(shape, dtype=tf.float32)
    w = tf.random.normal((c,), dtype=tf.float32)
    u = tf.random.normal((c,), dtype=tf.float32)
    return k, v, w, u


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=line_vals,  # Possible values for `line_arg`.
        line_names=line_names,  # Label name for the lines.
        styles=styles,  # Line styles.
        ylabel="time (ms)",  # Label name for the y-axis.
        plot_name="WKV",  # Name for the plot. Used also as a file name for saving the plot.
        y_log=True,
        args={},
        **kwargs,
    )
)
def benchmark(provider, sequence_length):
    hidden_dim = FLAGS.d
    batch_size = FLAGS.b
    shape = (batch_size, sequence_length, hidden_dim)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cuda":
        ms, min_ms, max_ms = triton.testing.do_bench(
            functools.partial(wkv_cuda.wkv, *create_torch_args(*shape)),
            quantiles=quantiles,
        )
    elif provider == "triton-serial":
        func = functools.partial(
            wkv_compiled, *create_torch_args(*shape), parallel=False
        )
        ms, min_ms, max_ms = triton.testing.do_bench(func, quantiles=quantiles)
    elif provider == "triton-parallel":
        if (
            FLAGS.triton_parallel_threshold != -1
            and sequence_length > FLAGS.triton_parallel_threshold
        ):
            ms, min_ms, max_ms = (np.nan,) * 3
        else:
            func = functools.partial(
                wkv_compiled, *create_torch_args(*shape), parallel=True
            )

            ms, min_ms, max_ms = triton.testing.do_bench(
                func,
                quantiles=quantiles,
            )
    elif provider == "jax2torch-serial":
        ms, min_ms, max_ms = triton.testing.do_bench(
            functools.partial(
                _wkv_jax2torch(parallel=False), *create_torch_args(*shape)
            ),
            quantiles=quantiles,
        )
    elif provider == "jax2torch-parallel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            functools.partial(
                _wkv_jax2torch(parallel=True), *create_torch_args(*shape)
            ),
            quantiles=quantiles,
        )
    elif provider == "jax-serial":
        ms, min_ms, max_ms = triton.testing.do_bench(
            functools.partial(_wkv_jax_fn(parallel=False), *create_jax_args(*shape)),
            quantiles=quantiles,
        )
    elif provider == "jax-parallel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            functools.partial(_wkv_jax_fn(parallel=True), *create_jax_args(*shape)),
            quantiles=quantiles,
        )
    elif provider == "tf-serial":
        if (
            FLAGS.tensorflow_serial_threshold != -1
            and sequence_length > FLAGS.tensorflow_serial_threshold
        ):
            ms, min_ms, max_ms = (np.nan,) * 3
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                functools.partial(_wkv_tf_fn(parallel=False), *create_tf_args(*shape)),
                quantiles=quantiles,
            )
    elif provider == "tf-parallel":
        ms, min_ms, max_ms = triton.testing.do_bench(
            functools.partial(_wkv_tf_fn(parallel=True), *create_tf_args(*shape)),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"provider '{provider}' not implemented")

    return ms, max_ms, min_ms


def main(_):
    if FLAGS.debug:
        torch._dynamo.config.verbose = True
    benchmark.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    app.run(main)
