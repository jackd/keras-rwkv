import unittest

import numpy as np
import tree
from keras_rwkv.layers.channel_mixer import ChannelMixer
from keras_rwkv.backend import ops


class ChannelMixerTest(unittest.TestCase):
    def test_iterative_cache(
        self,
        seed: int = 0,
        hidden_dim: int = 16,
        max_length: int = 11,
        current_index: int = 5,
        batch_size: int = 2,
    ):
        rng = np.random.default_rng(seed)
        mixer = ChannelMixer(0, 2)
        mixer.build((None, max_length, hidden_dim))

        inputs = rng.normal(size=(batch_size, max_length, hidden_dim)).astype("float32")

        outputs, cache = mixer.call_and_create_cache(
            inputs, current_index=current_index, max_length=max_length
        )

        # iterative implementation
        _, prev_cache = mixer.call_and_create_cache(
            inputs, current_index=current_index - 1, max_length=max_length
        )
        it_outputs, it_cache = mixer.call_with_cache(
            ops.expand_dims(inputs[:, current_index], 1),
            cache=prev_cache,
            current_index=current_index,
        )
        np.testing.assert_allclose(
            it_outputs, ops.expand_dims(outputs[:, current_index], 1), rtol=1e-4
        )
        tree.assert_same_structure(it_cache, cache)
        for itc, c in zip(tree.flatten(it_cache), tree.flatten(cache)):
            np.testing.assert_allclose(itc, c)

        # compare to raw call
        outputs = mixer(inputs)
        np.testing.assert_allclose(
            it_outputs, ops.expand_dims(outputs[:, current_index], 1), rtol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
