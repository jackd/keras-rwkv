import functools
import unittest
from absl.testing import parameterized
import numpy as np
import torch

from keras_rwkv.backend.torch import ew

# pylint:disable=no-member


class ExponentiallyWeightedTest(parameterized.TestCase):
    def test_evaluate(self, seed=0, shape=(3, 5)):
        rng = np.random.default_rng(seed)

        v, w = (rng.normal(size=shape).astype(np.float32) for _ in range(2))
        expected = np.exp(w) * v
        actual = ew.evaluate(torch.tensor(v), torch.tensor(w))
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_add(self, seed=0, shape=(3, 5)):
        rng = np.random.default_rng(seed)
        v1, w1, v2, w2 = (torch.tensor(rng.normal(size=shape)) for _ in range(4))

        expected = ew.evaluate(v1, w1) + ew.evaluate(v2, w2)
        actual = ew.evaluate(*ew.add(v1, w1, v2, w2))
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    @parameterized.product(axis=(0, 1, 2), parallel=(False, True))
    def test_cumsum(self, seed=0, shape=(3, 7, 2), axis=1, parallel: bool = False):
        rng = np.random.default_rng(seed)
        v, w = (
            torch.tensor(rng.normal(size=shape), dtype=torch.float32, device="cuda")
            for _ in range(2)
        )
        actual_v, actual_w = ew.cumsum(v, w, axis=axis, parallel=parallel)

        expected_evaluated = torch.cumsum(ew.evaluate(v, w), axis)
        actual_evaluated = ew.evaluate(actual_v, actual_w)
        np.testing.assert_allclose(
            expected_evaluated.cpu(), actual_evaluated.cpu(), rtol=1e-5
        )

        v = list(torch.unbind(v, axis))
        w = list(torch.unbind(w, axis))

        for i in range(1, shape[axis]):
            v[i], w[i] = ew.add(v[i - 1], w[i - 1], v[i], w[i])
        expected_v = torch.stack(v, axis)
        expected_w = torch.stack(w, axis)

        np.testing.assert_allclose(actual_v.cpu(), expected_v.cpu(), rtol=1e-5)
        np.testing.assert_allclose(actual_w.cpu(), expected_w.cpu(), rtol=1e-5)

        actual_evaluated = ew.evaluate(actual_v, actual_w)

    # def test_cumsum_grad(
    #     self, seed: int = 0, shape=(8,), axis: int = 0, reverse: bool = False
    # ):
    #     rng = np.random.default_rng(seed)
    #     v = rng.normal(size=shape)
    #     t = rng.normal(size=shape)
    #     v = torch.tensor(v, dtype=torch.float32, device="cuda", requires_grad=True)
    #     t = torch.tensor(t, dtype=torch.float32, device="cuda", requires_grad=True)

    #     torch.autograd.gradcheck(
    #         functools.partial(ew.cumsum, reverse=reverse, axis=axis),
    #         (v, t),
    #         rtol=1e-1,
    #         atol=1e-3,
    #         eps=1e-4,
    #     )


if __name__ == "__main__":
    unittest.main()
