import functools

import tensorflow as tf
from jax.experimental import jax2tf


@functools.cache
def convert_and_compile(jax_func, polymorphic_shapes=None, **kwargs):
    if kwargs:
        jax_func = functools.partial(jax_func, **kwargs)
    if polymorphic_shapes is not None and isinstance(polymorphic_shapes, tuple):
        polymorphic_shapes = list(polymorphic_shapes)
    wrapped = jax2tf.convert(jax_func, polymorphic_shapes=polymorphic_shapes)
    return tf.function(
        wrapped,
        jit_compile=True,
        autograph=False,
    )
