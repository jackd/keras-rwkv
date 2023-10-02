import os
import unittest

os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import tree
from keras_rwkv.models.v4 import RwkvBackbone, RwkvCausalLM, RwkvPreprocessor
from keras_nlp.tokenizers import WordPieceTokenizer
from keras_rwkv.backend import keras, ops


class RwkvCausalLMTest(unittest.TestCase):
    def test_generation_consistent(self):
        tokenizer = WordPieceTokenizer(
            ["[UNK]", "[START]", "[END]", "[PAD]", "hello", "world", "test"]
        )
        tokenizer.start_token_id = 1
        tokenizer.end_token_id = 2
        tokenizer.pad_token_id = 3
        preprocessor = RwkvPreprocessor(tokenizer, sequence_length=5)
        backbone = RwkvBackbone(tokenizer.vocabulary_size(), 32, 2)
        lm = RwkvCausalLM(backbone, preprocessor)

        inputs = {
            "token_ids": keras.ops.convert_to_tensor([[1, 4, 3, 3, 3]], "int32"),
            "padding_mask": keras.ops.convert_to_tensor(
                [[True, True, False, False, False]], "bool"
            ),
        }
        inputs = {k: keras.ops.convert_to_tensor(v) for k, v in inputs.items()}
        current_index = 2
        max_length = 5

        _, cache = lm._call_and_create_cache(
            inputs, current_index=current_index - 1, max_length=max_length
        )
        (it_logits, it_hidden_states), it_cache = lm._call_with_cache(
            keras.ops.expand_dims(inputs["token_ids"][:, current_index], 1),
            cache=cache,
            current_index=current_index,
        )

        (block_logits, block_hidden_states), block_cache = lm._call_and_create_cache(
            inputs, current_index=current_index, max_length=max_length
        )
        np.testing.assert_allclose(
            it_logits, ops.expand_dims(block_logits[:, current_index], 1), rtol=1e-4
        )
        np.testing.assert_allclose(
            ops.expand_dims(block_hidden_states[:, current_index], 1),
            it_hidden_states,
            rtol=1e-4,
        )

        tree.assert_same_structure(it_cache, block_cache)
        for iterative, block in zip(tree.flatten(it_cache), tree.flatten(block_cache)):
            np.testing.assert_allclose(iterative, block, rtol=2e-4)

        # compare to raw call
        logits = lm(inputs)
        np.testing.assert_allclose(
            it_logits, ops.expand_dims(logits[:, current_index], 1), rtol=1e-4
        )
        hidden_states = lm.backbone(inputs)  # pylint:disable=not-callable
        np.testing.assert_allclose(
            it_hidden_states,
            ops.expand_dims(hidden_states[:, current_index], 1),
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
