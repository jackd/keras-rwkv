import copy

import tree
from keras_nlp.models.generative_task import GenerativeTask
from keras_nlp.utils.python_utils import classproperty

from ...backend import keras, ops
from . import backbone
from . import causal_lm_preprocessor
from ..cached import call_and_create_cache, call_with_cache


@keras.utils.register_keras_serializable("keras_rwkv")
class RwkvCausalLM(GenerativeTask):  # pylint:disable=abstract-method
    """A thin wrapper around RWKV backbone."""

    def __init__(
        self,
        backbone: backbone.RwkvBackbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        hidden_states = backbone(inputs)
        logits = backbone.token_embedding(hidden_states, reverse=True)

        # Instantiate using Functional API Model constructor.
        super().__init__(
            inputs=inputs,
            outputs=logits,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.generate_function = None
        self._sampler = None

        # # Default compilation
        # self.compile(
        #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     optimizer=keras.optimizers.Adam(2e-5),
        #     metrics=[keras.metrics.SparseCategoricalAccuracy()],
        #     jit_compile=True,
        # )

    @classproperty
    def presets(cls):  # pylint:disable=no-self-argument
        return copy.deepcopy(backbone.backbone_presets)

    @classproperty
    def backbone_cls(cls):  # pylint:disable=no-self-argument
        return backbone.RwkvBackbone

    @classproperty
    def preprocessor_cls(cls):  # pylint:disable=no-self-argument
        return causal_lm_preprocessor.RwkvCausalLMPreprocessor

    def _call_and_create_cache(self, inputs, *, current_index, max_length):
        hidden_states, cache = call_and_create_cache(
            self.backbone, inputs, current_index=current_index, max_length=max_length
        )
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return (logits, hidden_states), cache

    def _call_with_cache(
        self,
        prompt,
        *,
        cache,
        current_index,
    ):
        """Forward pass of `RwkvGPTCausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous cached values and avoids recomputing the outputs
        of seen tokens.

        Args:
            prompt: a dense int Tensor with shape `(batch_size, max_length)`.
            cache: a structure of previously cached values.
            current_index: index of the most recent valid token.

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the decoding cache.
        """
        inputs = {"token_ids": prompt, "padding_mask": ops.ones_like(prompt, "bool")}
        hidden_states, next_cache = call_with_cache(
            self.backbone, inputs, cache=cache, current_index=current_index
        )
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return (logits, hidden_states), next_cache

    def generate_step(  # pylint:disable=arguments-differ
        self, inputs, end_token_id=None
    ):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"token_ids"` and `"padding_mask"`.

        Args:
            inputs: A dictionary with two keys `"token_ids"` and
                `"padding_mask"` and batched tensor values.
            end_token_id: The id of the end token to stop on. If all
                sequences have produced a new `end_token_id`, generation
                will stop.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        # Create and seed cache with a single forward pass.
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)
        (_, hidden_states), cache = self._call_and_create_cache(
            inputs,
            current_index=index - 2,
            max_length=ops.shape(token_ids)[1],
        )
        cache_structure = cache
        cache = tuple(tree.flatten(cache))

        def next(prompt, cache, index):  # pylint:disable=redefined-builtin
            # pytorch stacks the cache
            if not isinstance(cache, tuple):
                cache = ops.unstack(cache, axis=0)
            cache = tree.unflatten_as(cache_structure, cache)
            current_index = index - 1
            prompt = ops.expand_dims(prompt[:, current_index], axis=1)
            (logits, hidden_states), cache = self._call_with_cache(
                prompt,
                cache=cache,
                current_index=current_index,
            )
            cache = tuple(tree.flatten(cache))
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        hidden_states = ops.zeros_like(self.backbone.output)

        token_ids = self._sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            end_token_id=end_token_id,
            hidden_states=hidden_states,
        )

        # Compute an output padding mask with the token ids we updated.
        if end_token_id is not None:
            # Build a mask of `end_token_id` locations not in the original
            # prompt (not in locations where `padding_mask` is True).
            end_locations = ops.logical_and(
                ops.equal(token_ids, end_token_id),
                ops.logical_not(padding_mask),
            )
            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after end_locations.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations will have been updated.
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
