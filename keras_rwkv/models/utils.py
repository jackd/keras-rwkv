import typing as tp

import tree
from keras_rwkv.backend import keras


def as_model(func: tp.Callable, inputs, **model_kwargs):
    model_inputs = tree.map_structure(
        lambda i: keras.Input(shape=i.shape[1:], dtype=i.dtype, batch_size=i.shape[0]),
        inputs,
    )
    outputs = func(model_inputs)
    model = keras.Model(model_inputs, outputs, **model_kwargs)
    return model(inputs)
