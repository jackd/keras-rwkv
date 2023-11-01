import os
import copy

from keras_nlp.models.backbone import Backbone
from keras_nlp.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_nlp.utils.python_utils import classproperty
from ...backend import keras
from ...layers.block import ChannelMixBlock, HybridMixBlock

from .presets import backbone_presets


@keras.utils.register_keras_serializable("keras_rwkv")
class RwkvBackbone(Backbone):  # pylint:disable=abstract-method
    def __init__(
        self,
        vocabulary_size: int,
        hidden_dim: int,
        num_layers: int,
        ffn_pre: bool = False,
        layer_norm_epsilon: float = 1e-5,
        use_original_cuda_wkv: bool = False,
        parallel_wkv: bool = True,
        **kwargs,
    ):
        # Inputs
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(shape=(None,), dtype="int32", name="padding_mask")

        # Embed tokens, positions.
        token_embedding_layer = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            name="token_embedding",
            embeddings_initializer=keras.initializers.RandomUniform(-1e-4, 1e-4),
            tie_weights=False,
        )
        x = token_embedding_layer(token_ids)
        x = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="input_norm"
        )(x)
        if ffn_pre:
            x = ChannelMixBlock(
                0, num_layers, layer_norm_epsilon=layer_norm_epsilon, name="block0"
            )(x)
        else:
            x = HybridMixBlock(
                0,
                num_layers,
                layer_norm_epsilon=layer_norm_epsilon,
                use_original_cuda_wkv=use_original_cuda_wkv,
                parallel_wkv=parallel_wkv,
                name="block0",
            )(x)
        for i in range(1, num_layers):
            x = HybridMixBlock(
                i,
                num_layers,
                layer_norm_epsilon=layer_norm_epsilon,
                use_original_cuda_wkv=use_original_cuda_wkv,
                parallel_wkv=parallel_wkv,
                name=f"block{i}",
            )(x)
        sequence_output = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="output_norm"
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=sequence_output,
            **kwargs,
        )

        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.ffn_pre = ffn_pre
        self.token_embedding = token_embedding_layer
        self.use_original_cuda_wkv = use_original_cuda_wkv
        self.parallel_wkv = parallel_wkv

    def get_config(self):
        config = super().get_config()
        config.update(
            vocabulary_size=self.vocabulary_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            ffn_pre=self.ffn_pre,
            use_original_cuda_wkv=self.use_original_cuda_wkv,
            parallel_wkv=self.parallel_wkv,
        )
        return config

    @classproperty
    def presets(cls):  # pylint:disable=no-self-argument
        return copy.deepcopy(backbone_presets)

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate RwkvBackbone model from preset architecture and weights.

        Args:
            preset: string. Must be one of "{{preset_names}}".
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.

        Examples:
        ```python
        # Load architecture and weights from preset
        model = keras_rwkv.models.v4.RwkvBackbone.from_preset(
            "{{example_preset_name}}"
        )

        # Load randomly initialized model from preset architecture
        model = keras_rwkv.models.v4.RwkvBackbone.from_preset(
            "{{example_preset_name}}",
            load_weights=False
        )
        ```
        """

        if not cls.presets:
            raise NotImplementedError("No presets have been created for this class.")

        if preset not in cls.presets:  # pylint:disable=unsupported-membership-test
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )
        metadata = cls.presets[preset]  # pylint:disable=unsubscriptable-object
        config = metadata["config"]
        model = cls.from_config({**config, **kwargs})

        if not load_weights:
            return model

        weights_path = keras.utils.get_file(
            "weights.pth",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["weights_hash"],
        )
        import torch  # pylint:disable=import-outside-toplevel

        weights = torch.load(weights_path, weights_only=True, map_location="cpu")
        weights = {
            k: v.to(torch.float32)  # pylint:disable=no-member
            for k, v in weights.items()
        }
        # forward/reverse embeddings
        model.token_embedding.embeddings.assign(weights.pop("emb.weight"))
        model.token_embedding.reverse_embeddings.assign(weights.pop("head.weight").T)

        model.get_layer("input_norm").gamma.assign(weights.pop("blocks.0.ln0.weight"))
        model.get_layer("input_norm").beta.assign(weights.pop("blocks.0.ln0.bias"))

        # blocks
        assert not model.ffn_pre, "not implemented"
        for b in range(model.num_layers):
            block = model.get_layer(f"block{b}")
            for i in range(2):
                block.layer_norms[i].gamma.assign(
                    weights.pop(f"blocks.{b}.ln{i+1}.weight")
                )
                block.layer_norms[i].beta.assign(
                    weights.pop(f"blocks.{b}.ln{i+1}.bias")
                )

            # time mixer
            tm = block.mixers[0]
            tm.time_decay.assign(weights.pop(f"blocks.{b}.att.time_decay").squeeze())
            tm.time_first.assign(weights.pop(f"blocks.{b}.att.time_first").squeeze())
            tm.time_mix_k.assign(weights.pop(f"blocks.{b}.att.time_mix_k").squeeze())
            tm.time_mix_v.assign(weights.pop(f"blocks.{b}.att.time_mix_v").squeeze())
            tm.time_mix_r.assign(weights.pop(f"blocks.{b}.att.time_mix_r").squeeze())

            tm.key_layer.kernel.assign(weights.pop(f"blocks.{b}.att.key.weight").T)
            tm.value_layer.kernel.assign(weights.pop(f"blocks.{b}.att.value.weight").T)
            tm.receptance_layer.kernel.assign(
                weights.pop(f"blocks.{b}.att.receptance.weight").T
            )
            tm.output_layer.kernel.assign(
                weights.pop(f"blocks.{b}.att.output.weight").T
            )

            # channel mixer
            cm = block.mixers[1]
            cm.time_mix_k.assign(weights.pop(f"blocks.{b}.ffn.time_mix_k").squeeze())
            cm.time_mix_r.assign(weights.pop(f"blocks.{b}.ffn.time_mix_r").squeeze())

            cm.key_layer.kernel.assign(weights.pop(f"blocks.{b}.ffn.key.weight").T)
            cm.value_layer.kernel.assign(weights.pop(f"blocks.{b}.ffn.value.weight").T)
            cm.receptance_layer.kernel.assign(
                weights.pop(f"blocks.{b}.ffn.receptance.weight").T
            )
        model.get_layer("output_norm").gamma.assign(weights.pop("ln_out.weight"))
        model.get_layer("output_norm").beta.assign(weights.pop("ln_out.bias"))

        assert not weights, list(weights)
        return model
