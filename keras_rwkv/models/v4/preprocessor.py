import copy

from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.layers.preprocessing.start_end_packer import StartEndPacker
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty

from ...backend import keras
from .presets import backbone_presets
from .tokenizer import RwkvTokenizer


@keras.utils.register_keras_serializable("keras_rwkv")
class RwkvPreprocessor(Preprocessor):  # pylint:disable=abstract-method
    def __init__(
        self,
        tokenizer,
        sequence_length=1024,
        add_start_token=False,
        add_end_token=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token
        self.packer = StartEndPacker(
            start_value=tokenizer.start_token_id,
            end_value=tokenizer.end_token_id,
            pad_value=tokenizer.pad_token_id,
            sequence_length=sequence_length,
            return_padding_mask=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "add_start_token": self.add_start_token,
                "add_end_token": self.add_end_token,
            }
        )
        return config

    def call(  # pylint:disable=arguments-differ
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        x = convert_inputs_to_list_of_tensor_segments(x)
        if len(x) != 1:
            raise ValueError(
                "Rwkv requires each input feature to contain only "
                f"one segment, but received {len(x)}. If you are using Rwkv "
                "for a multi-segment classification task, please refer to "
                "classification models like BERT or RoBERTa."
            )
        sequence_length = sequence_length or self.sequence_length
        token_ids, padding_mask = self.packer(
            self.tokenizer(x[0]),  # pylint:disable=not-callable
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def presets(cls):  # pylint:disable=no-self-argument
        return copy.deepcopy(backbone_presets)

    @classproperty
    def tokenizer_cls(cls):  # pylint:disable=no-self-argument
        return RwkvTokenizer

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate {{preprocessor_name}} from preset architecture.

        Args:
            preset: string. Must be one of "{{preset_names}}".

        Examples:
        ```python
        # Load a preprocessor layer from a preset.
        preprocessor = keras_nlp.models.{{preprocessor_name}}.from_preset(
            "{{example_preset_name}}",
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

        tokenizer = cls.tokenizer_cls.from_preset(preset)

        metadata = cls.presets[preset]  # pylint:disable=unsubscriptable-object
        # Use model's `max_sequence_length` if `sequence_length` unspecified;
        # otherwise check that `sequence_length` not too long.
        sequence_length = kwargs.pop("sequence_length", None)
        max_sequence_length = metadata["max_sequence_length"]
        if sequence_length is not None:
            if sequence_length > max_sequence_length:
                raise ValueError(
                    f"`sequence_length` cannot be longer than `{preset}` "
                    f"preset's `max_sequence_length` of {max_sequence_length}. "
                    f"Received: {sequence_length}."
                )
        else:
            sequence_length = max_sequence_length

        return cls(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            **kwargs,
        )
