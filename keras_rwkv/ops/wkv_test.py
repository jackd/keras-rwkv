import unittest

import numpy as np
from absl.testing import parameterized
from keras_core import ops
from keras_core.src.testing import TestCase

from keras_rwkv.ops import wkv


def _wkv_baseline(k, v, w, u):
    output = np.zeros_like(v)
    _, T, _ = k.shape
    for t in range(T):
        for i in range(t):
            output[:, t] += np.exp(-(t - i - 1) * w + k[:, i]) * v[:, i]
        output[:, t] += np.exp(u + k[:, t]) * v[:, t]
    return output


def wkv_baseline(k, v, w, u):
    return _wkv_baseline(k, v, w, u) / _wkv_baseline(k, np.ones_like(v), w, u)


class WkvCorrectnessTest(TestCase, parameterized.TestCase):
    def test_wkv_is_consistent(
        self, seed: int = 0, b: int = 2, t: int = 128, c: int = 3
    ):
        rng = np.random.default_rng(seed)
        shape = (b, t, c)

        k = rng.normal(size=shape)
        v = rng.normal(size=shape)
        w = rng.normal(size=c)
        u = rng.normal(size=c)

        k, v, w, u = (x.astype(np.float32) for x in (k, v, w, u))

        baseline = wkv_baseline(k, v, w, u)
        actual, _ = wkv.wkv(k, v, w, u)

        np.testing.assert_allclose(baseline, actual, rtol=1e-3)

    def test_update_consistent(
        self, seed: int = 0, b: int = 2, t: int = 128, c: int = 3
    ):
        rng = np.random.default_rng(seed)
        shape = (b, t, c)
        dtype = "float32"

        k = rng.normal(size=shape)
        v = rng.normal(size=shape)
        w = rng.normal(size=c)
        u = rng.normal(size=c)

        k, v, w, u = (ops.convert_to_tensor(x, dtype) for x in (k, v, w, u))

        block_vals, block_state = wkv.wkv(k, v, w, u)

        shape = (b, 1, c)
        k0 = rng.normal(size=shape)
        v0 = rng.normal(size=shape)
        k0, v0 = (ops.convert_to_tensor(x, dtype) for x in (k0, v0))
        next_vals, next_state = wkv.wkv_update(k0, v0, w, u, block_state)
        cat_vals = np.concatenate((block_vals, next_vals), axis=1)

        full_block_vals, full_state = wkv.wkv(
            np.concatenate((k, k0), axis=1), np.concatenate((v, v0), axis=1), w, u
        )

        np.testing.assert_allclose(cat_vals, full_block_vals, rtol=1e-4)
        self.assertEqual(len(next_state), len(full_state))
        for next_val, full_val in zip(next_state, full_state):
            np.testing.assert_allclose(next_val, full_val, rtol=1e-6)

    def test_update_consistent_with_current_index(
        self,
        seed: int = 0,
        b: int = 2,
        t: int = 128,
        c: int = 3,
        current_index: int = 5,
    ):
        rng = np.random.default_rng(seed)
        shape = (b, t, c)
        dtype = "float32"

        k = rng.normal(size=shape)
        v = rng.normal(size=shape)
        w = rng.normal(size=c)
        u = rng.normal(size=c)

        k, v, w, u = (ops.convert_to_tensor(x, dtype) for x in (k, v, w, u))

        block_vals, block_state = wkv.wkv(k, v, w, u, current_index=current_index - 1)
        updated_vals, updated_state = wkv.wkv_update(
            ops.expand_dims(k[:, current_index], 1),
            ops.expand_dims(v[:, current_index], 1),
            w,
            u,
            block_state,
        )
        block_vals, block_state = wkv.wkv(k, v, w, u, current_index=current_index)
        np.testing.assert_allclose(
            updated_vals, block_vals[:, current_index : current_index + 1], rtol=1e-4
        )
        self.assertEqual(len(updated_state), len(block_state))
        for updated, block in zip(updated_state, block_state):
            np.testing.assert_allclose(updated, block, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
