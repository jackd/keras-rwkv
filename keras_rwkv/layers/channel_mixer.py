import typing as tp
from keras_rwkv.backend import keras, ops

# pylint:disable=attribute-defined-outside-init,arguments-differ


@keras.utils.register_keras_serializable("keras_rwkv")
class ChannelMixer(keras.layers.Layer):  # pylint:disable=abstract-method
    def __init__(
        self,
        layer_index: int,
        num_layers: int,
        hidden_dim: tp.Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_index = layer_index
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update(
            layer_index=self.layer_index,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
        )
        return config

    def build(self, input_shape):
        if self.built:
            return
        hidden_dim = self.hidden_dim or input_shape[-1]

        ratio_1_to_almost0 = 1.0 - (self.layer_index / self.num_layers)  # 1 to ~0

        def init(shape, dtype=None):
            return ops.arange(0, 1, 1 / shape[0], dtype=dtype) ** ratio_1_to_almost0

        self.time_mix_k = self.add_weight(
            name="time_mix_k", shape=(hidden_dim,), initializer=init
        )
        self.time_mix_r = self.add_weight(
            name="time_mix_r", shape=(hidden_dim,), initializer=init
        )

        self.key_layer = keras.layers.Dense(
            hidden_dim * 4,
            use_bias=False,
            activation=lambda x: ops.square(ops.relu(x)),
            name="key",
        )
        self.receptance_layer = keras.layers.Dense(
            hidden_dim, use_bias=False, activation="sigmoid", name="receptance"
        )
        self.value_layer = keras.layers.Dense(hidden_dim, use_bias=False, name="value")
        self.key_layer.build(input_shape)
        self.value_layer.build((*input_shape[:-1], hidden_dim * 4))
        self.receptance_layer.build(input_shape)
        super().build(input_shape)

    def _get_rkv(self, x, xx):
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key_layer(xk)
        kv = self.value_layer(k)
        return self.receptance_layer(xr) * kv

    def call(self, inputs):
        x = inputs
        xx = ops.pad(x[:, :-1], [[0, 0], [1, 0], [0, 0]])  # time shift
        return self._get_rkv(x, xx)

    def call_with_cache(self, inputs, *, cache, current_index, mask=None):
        assert self.built
        del current_index, mask
        x = inputs
        xx = cache
        rkv = self._get_rkv(x, xx)
        return rkv, x

    def call_and_create_cache(self, inputs, *, current_index, max_length, mask=None):
        assert self.built
        del max_length, mask
        x = inputs
        xx = ops.pad(x[:, :-1], [[0, 0], [1, 0], [0, 0]])  # time shift
        output = self._get_rkv(x, xx)
        cache = ops.expand_dims(x[:, current_index], axis=1)
        return output, cache
