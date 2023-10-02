import unittest
from absl.testing import parameterized
import numpy as np
import jax.numpy as jnp

from keras_rwkv.backend.jax import ew

# pylint:disable=no-member


class ExponentiallyWeightedTest(parameterized.TestCase):
    def test_evaluate(self, seed=0, shape=(3, 5)):
        rng = np.random.default_rng(seed)

        v, w = (rng.normal(size=shape).astype(np.float32) for _ in range(2))
        expected = np.exp(w) * v
        actual = ew.evaluate(jnp.asarray(v), jnp.asarray(w))
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_add(self, seed=0, shape=(3, 5)):
        rng = np.random.default_rng(seed)
        v1, w1, v2, w2 = (jnp.asarray(rng.normal(size=shape)) for _ in range(4))

        expected = ew.evaluate(v1, w1) + ew.evaluate(v2, w2)
        actual = ew.evaluate(*ew.add(v1, w1, v2, w2))
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    @parameterized.product(parallel=(False, True), axis=(0, 1, 2))
    def test_cumsum(self, seed=0, shape=(3, 7, 2), axis=1, parallel: bool = True):
        rng = np.random.default_rng(seed)
        v, w = (
            jnp.asarray(rng.normal(size=shape), dtype=jnp.float32) for _ in range(2)
        )
        actual_v, actual_w = ew.cumsum(v, w, axis=axis, parallel=parallel)

        # ensure jnp.cumsum(ew.evaluate(v, w)) == ew.evaluate(*ew.cumsum(v, w))
        expected_evaluated = jnp.cumsum(ew.evaluate(v, w), axis)
        actual_evaluated = ew.evaluate(actual_v, actual_w)
        np.testing.assert_allclose(expected_evaluated, actual_evaluated, rtol=1e-5)

        v = list(jnp.moveaxis(v, axis, 0))
        w = list(jnp.moveaxis(w, axis, 0))

        for i in range(1, shape[axis]):
            v[i], w[i] = ew.add(v[i - 1], w[i - 1], v[i], w[i])
        expected_v = jnp.stack(v, axis=axis)
        expected_w = jnp.stack(w, axis=axis)

        np.testing.assert_allclose(actual_v, expected_v, rtol=1e-5)
        np.testing.assert_allclose(actual_w, expected_w, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
    # ExponentiallyWeightedTest().test_cumsum_grad()
    # unittest.main()
    # ExponentiallyWeightedTest().test_add()
