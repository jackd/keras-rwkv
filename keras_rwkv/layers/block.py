import abc
import typing as tp

from ..backend import keras
from . import time_mixer, channel_mixer

# pylint:disable=arguments-differ,attribute-defined-outside-init


class _MixBlock(keras.layers.Layer):
    def __init__(
        self,
        layer_index: int,
        num_layers: int,
        layer_norm_epsilon: int = 1e-5,
        **kwargs,
    ):
        self.layer_index = layer_index
        self.num_layers = num_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        super().__init__(**kwargs)
        # self references above here

    def build(self, input_shape):
        if self.built:
            return
        hidden_dim = input_shape[-1]
        self.mixers = tuple(self._create_mixers(hidden_dim))
        self.layer_norms = tuple(
            keras.layers.LayerNormalization(
                epsilon=self.layer_norm_epsilon, name=f"layer_norm{i}"
            )
            for i in range(len(self.mixers))
        )
        for layer in self.mixers + self.layer_norms:
            layer.build(input_shape)
        super().build(input_shape)

    def call(self, input):  # pylint:disable=redefined-builtin
        x = input
        for layer_norm, mixer in zip(self.layer_norms, self.mixers):
            x = x + mixer(layer_norm(x))
        return x

    @abc.abstractmethod
    def _create_mixers(self, hidden_dim: int) -> tp.Iterable[keras.layers.Layer]:
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update(
            layer_index=self.layer_index,
            num_layers=self.num_layers,
            layer_norm_epsilon=self.layer_norm_epsilon,
        )
        return config


@keras.utils.register_keras_serializable("keras_rwkv")
class ChannelMixBlock(_MixBlock):  # pylint:disable=abstract-method
    """Two channel mixing layers in series."""

    def _create_mixers(self, hidden_dim: int):
        return (
            channel_mixer.ChannelMixer(
                self.layer_index,
                num_layers=self.num_layers,
                hidden_dim=hidden_dim,
                name=f"channel_mixer{d}",
            )
            for d in range(2)
        )


@keras.utils.register_keras_serializable("keras_rwkv")
class HybridMixBlock(_MixBlock):  # pylint:disable=abstract-method
    """Time mixing followed by channel mixing."""

    def __init__(
        self,
        layer_index: int,
        num_layers: int,
        layer_norm_epsilon: int = 1e-5,
        use_original_cuda_wkv: bool = False,
        parallel_wkv: bool = True,
        **kwargs,
    ):
        super().__init__(
            layer_index, num_layers, layer_norm_epsilon=layer_norm_epsilon, **kwargs
        )
        self.use_original_cuda_wkv = use_original_cuda_wkv
        self.parallel_wkv = parallel_wkv

    def get_config(self):
        config = super().get_config()
        config.update(
            use_original_cuda_wkv=self.use_original_cuda_wkv,
            parallel_wkv=self.parallel_wkv,
        )
        return config

    def _create_mixers(self, hidden_dim: int):
        return (
            time_mixer.TimeMixer(
                self.layer_index,
                num_layers=self.num_layers,
                hidden_dim=hidden_dim,
                use_original_cuda_wkv=self.use_original_cuda_wkv,
                parallel_wkv=self.parallel_wkv,
                name="time_mixer",
            ),
            channel_mixer.ChannelMixer(
                self.layer_index,
                num_layers=self.num_layers,
                hidden_dim=hidden_dim,
                name="channel_mixer",
            ),
        )
