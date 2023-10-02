import json
import copy
import os

from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_nlp.utils.python_utils import classproperty

from ...backend import keras
from . import backbone_presets


@keras.utils.register_keras_serializable("keras_rwkv")
class RwkvTokenizer(BytePairTokenizer):  # pylint:disable=abstract-method
    """A GPT-2-like tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by GPT-2
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a GPT-2 preset.

    This tokenizer does not provide truncation or padding of inputs.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_rwkv.models.v4.RwkvTokenizer.from_preset("rwkv-4-pile-169m")
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    vocab = {"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    merges += ["Ġ f", "o x", "Ġf ox"]
    tokenizer = keras_rwkv.models.v4.RwkvTokenizer(vocabulary=vocab, merges=merges)
    tokenizer("a quick fox.")
    ```
    """

    def __init__(
        self,
        vocabulary,
        merges,
        **kwargs,
    ):
        # Special tokens.
        end_token = "<|endoftext|>"
        padding_token = "<|padding|>"
        unsplittable_tokens = [end_token, padding_token]

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=unsplittable_tokens,
            **kwargs,
        )

        # Check whether special tokens are present in the vocabulary.
        for token in unsplittable_tokens:
            if token not in self.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.end_token_id = self.token_to_id(end_token)
        # Rwkv uses the same start as end token, i.e., "<|endoftext|>".
        self.start_token_id = self.end_token_id
        self.pad_token_id = self.token_to_id(padding_token)

    @classproperty
    def presets(cls):  # pylint:disable=no-self-argument
        return copy.deepcopy(backbone_presets)

    def get_config(self):
        config = super().get_config()
        # In the constructor, we pass the list of special tokens to the
        # `unsplittable_tokens` arg of the superclass' constructor. Hence, we
        # delete it from the config here.
        del config["unsplittable_tokens"]
        return config

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate {{model_name}} tokenizer from preset vocabulary.

        Args:
            preset: string. Must be one of "{{preset_names}}".

        Examples:
        ```python
        # Load a preset tokenizer.
        tokenizer = {{model_name}}.from_preset("{{example_preset_name}}")

        # Tokenize some input.
        tokenizer("The quick brown fox tripped.")

        # Detokenize some input.
        tokenizer.detokenize([5, 6, 7, 8, 9])
        ```
        """

        if not cls.presets:
            raise NotImplementedError("No presets have been created for this class")

        if preset not in cls.presets:  # pylint:disable=unsupported-membership-test
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )
        metadata = cls.presets[preset]  # pylint:disable=unsubscriptable-object

        tokenizer_path = keras.utils.get_file(
            "tokenizer.json",
            metadata["tokenizer_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["tokenizer_hash"],
        )
        with open(tokenizer_path, "r", encoding="utf8") as fp:
            tokenizer_data = json.load(fp)
        vocabulary = tokenizer_data["model"]["vocab"]

        for added_token in tokenizer_data["added_tokens"]:
            content = added_token["content"]
            if content in ("<|endoftext|>", "<|padding|>"):
                continue
            id_ = added_token["id"]
            assert content not in vocabulary
            vocabulary[content] = id_

        merges = tokenizer_data["model"]["merges"]

        config = {
            "vocabulary": vocabulary,
            "merges": merges,
        }
        return cls.from_config({**config, **kwargs})
