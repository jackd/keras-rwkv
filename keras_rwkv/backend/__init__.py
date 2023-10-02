from keras_nlp.backend import keras, ops

if keras.backend.backend() == "tensorflow":
    import tensorflow as tf

    from .tensorflow import ew
    from .tensorflow import wkv_cuda

    Array = tf.Tensor
elif keras.backend.backend() == "jax":
    import jax.numpy as jnp

    from .jax import ew
    from .jax import wkv_cuda

    Array = jnp.ndarray
elif keras.backend.backend() == "torch":
    import torch

    Array = torch.Tensor  # must be before next .torch import for some reason...
    from .torch import ew
    from .torch import wkv_cuda


else:
    raise ValueError(f"Unsupported keras backend {keras.backend.backend()}")


__all__ = ["Array", "ew", "keras", "ops", "wkv_cuda"]
