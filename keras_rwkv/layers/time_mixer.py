import typing as tp
import math

from ..backend import keras, ops

from ..ops import wkv
from ..ops import wkv_cuda

# pylint:disable=attribute-defined-outside-init


@keras.utils.register_keras_serializable("keras_rwkv")
class TimeMixer(keras.layers.Layer):  # pylint:disable=abstract-method
    def __init__(
        self,
        layer_index: int,
        num_layers: int,
        hidden_dim: tp.Optional[int] = None,
        use_original_cuda_wkv: bool = False,
        parallel_wkv: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_index = layer_index
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_original_cuda_wkv = use_original_cuda_wkv
        self.parallel_wkv = parallel_wkv
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update(
            layer_index=self.layer_index,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            use_original_cuda_wkv=self.use_original_cuda_wkv,
            parallel_wkv=self.parallel_wkv,
        )
        return config

    def build(self, input_shape):
        if self.built:
            return
        ratio_0_to_1 = self.layer_index / (self.num_layers - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (self.layer_index / self.num_layers)  # 1 to ~0

        hidden_dim = self.hidden_dim or input_shape[-1]
        self.time_decay = self.add_weight(
            name="time_decay",
            shape=(hidden_dim,),
            # initializer=TimeDecayInitializer(self.layer_index, self.num_layers),
            initializer=lambda shape, dtype: 8
            * (ops.arange(shape[0], dtype=dtype) / (shape[0] - 1))
            ** (0.7 + 1.3 * ratio_0_to_1)
            - 5,
        )
        self.time_first = self.add_weight(
            name="time_first",
            shape=(hidden_dim,),
            # initializer=FirstTimeInitializer()
            initializer=lambda shape, dtype: ops.cast(
                (ops.arange(1, 1 + shape[0], dtype="int32") % 3 - 1), dtype
            )
            / 2
            + ops.convert_to_tensor(math.log(3), dtype),
        )
        self.time_mix_k = self.add_weight(
            name="time_mix_k",
            shape=(hidden_dim,),
            initializer=lambda shape, dtype: ops.arange(0, 1, 1 / shape[0], dtype=dtype)
            ** ratio_1_to_almost0,
        )
        self.time_mix_v = self.add_weight(
            name="time_mix_v",
            shape=(hidden_dim,),
            initializer=lambda shape, dtype: ops.linspace(0, 1, shape[0], dtype=dtype)
            ** ratio_1_to_almost0
            + 0.3 * ratio_0_to_1,
        )
        self.time_mix_r = self.add_weight(
            name="time_mix_r",
            shape=(hidden_dim,),
            initializer=lambda shape, dtype: ops.linspace(0, 1, shape[0], dtype=dtype)
            ** (ratio_1_to_almost0 / 2),
        )

        self.key_layer = keras.layers.Dense(hidden_dim, use_bias=False, name="key")
        self.value_layer = keras.layers.Dense(hidden_dim, use_bias=False, name="value")
        self.receptance_layer = keras.layers.Dense(
            hidden_dim, use_bias=False, activation="sigmoid", name="receptance"
        )
        self.output_layer = keras.layers.Dense(
            hidden_dim, use_bias=False, name="output"
        )
        for layer in (
            self.key_layer,
            self.value_layer,
            self.receptance_layer,
            self.output_layer,
        ):
            layer.build(input_shape)
        super().build(input_shape)

    # def call(
    #     self, inputs, cache=None, current_length=None
    # ):  # pylint:disable=arguments-differ
    #     if cache is None:
    #         return self._call_parallel(inputs)
    #     return self._call_cached(inputs, cache=cache, current_length=current_length)

    def _get_transformed_components(self, x, xx):
        # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        # Use xk, xv, xr to produce k, v, r
        k = self.key_layer(xk)
        v = self.value_layer(xv)
        sr = self.receptance_layer(xr)
        return k, v, sr

    def _wkv(self, k, v, current_index=None):
        w = ops.exp(ops.convert_to_tensor(self.time_decay))
        u = ops.convert_to_tensor(self.time_first)
        if current_index is not None:
            current_index = ops.convert_to_tensor(current_index, "int32")
        if self.use_original_cuda_wkv:
            return wkv_cuda.wkv(k, v, w=w, u=u, current_index=current_index)

        return wkv.wkv(
            k, v, w=w, u=u, parallel=self.parallel_wkv, current_index=current_index
        )

    def _wkv_update(self, k, v, state):
        w = ops.exp(ops.convert_to_tensor(self.time_decay))
        u = ops.convert_to_tensor(self.time_first)
        if self.use_original_cuda_wkv:
            return wkv_cuda.wkv_update(k, v, w=w, u=u, state=state)
        return wkv.wkv_update(k, v, w=w, u=u, state=state)

    def call(self, inputs):
        x = inputs
        xx = ops.pad(x[:, :-1], [[0, 0], [1, 0], [0, 0]])  # time shift
        k, v, sr = self._get_transformed_components(x, xx)
        wkv_, _ = self._wkv(k, v)
        rwkv = sr * wkv_
        return self.output_layer(rwkv)

    def call_with_cache(self, inputs, *, cache, current_index=None, mask=None):
        del current_index
        del mask
        x = inputs
        xx, state = cache
        k, v, sr = self._get_transformed_components(x, xx)
        wkv_, state = self._wkv_update(k, v, state)
        rwkv = sr * wkv_
        out_cache = (x, state)
        return self.output_layer(rwkv), out_cache

    def call_and_create_cache(
        self, inputs, *, current_index=None, max_length=None, mask=None
    ):
        assert self.built
        del max_length, mask
        x = inputs
        xx = ops.pad(x[:, :-1], [[0, 0], [1, 0], [0, 0]])  # time shift
        k, v, sr = self._get_transformed_components(x, xx)

        wkv_, state = self._wkv(k, v, current_index=current_index)
        rwkv = sr * wkv_
        output = self.output_layer(rwkv)
        xx_next = ops.expand_dims(x[:, current_index], axis=1)
        return output, (xx_next, state)
