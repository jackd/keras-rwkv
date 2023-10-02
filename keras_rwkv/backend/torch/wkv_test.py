import unittest

import tree
import numpy as np
import torch

from keras_rwkv.backend.torch import wkv, wkv_cuda, wkv_jax

# pylint:disable=no-member


class TorchWkvTest(unittest.TestCase):
    def test_wkv_consistent(self, seed: int = 0, b: int = 2, t: int = 128, c: int = 3):
        rng = np.random.default_rng(seed)

        k = rng.normal(size=(b, t, c))
        v = rng.normal(size=(b, t, c))
        w = rng.uniform(size=(c,))
        u = rng.normal(size=(c,))

        k, v, w, u = (
            torch.tensor(x, dtype=torch.float32, device="cuda") for x in (k, v, w, u)
        )

        cuda_impl, _ = wkv_cuda.wkv(k, v, w, u)
        jax_impl, _ = wkv_jax.wkv(k, v, w, u)
        cuda_impl = cuda_impl.cpu()
        jax_impl = jax_impl.cpu()
        np.testing.assert_allclose(cuda_impl.squeeze(), jax_impl.squeeze(), rtol=1e-3)

    def test_wkv_cuda_consistent_across_channels(
        self, seed: int = 0, b: int = 2, t: int = 5
    ):
        rng = np.random.default_rng(seed)
        c = 2

        k = rng.normal(size=(b, t, c))
        v = rng.normal(size=(b, t, c))
        w = rng.uniform(size=(c,))
        u = rng.normal(size=(c,))

        k, v, w, u = (
            torch.tensor(x, dtype=torch.float32, device="cuda") for x in (k, v, w, u)
        )

        y, state = wkv_cuda.wkv(k, v, w, u)
        r0 = wkv_cuda.wkv(k[..., :1], v[..., :1], w[:1], u[:1])
        r1 = wkv_cuda.wkv(k[..., 1:], v[..., 1:], w[1:], u[1:])

        y_stacked, state_stacked = tree.map_structure(
            lambda *args: torch.concat(args, axis=-1), r0, r1
        )
        np.testing.assert_allclose(y_stacked.cpu(), y.cpu())
        np.testing.assert_allclose(state_stacked[0].cpu(), state[0].cpu())  # aa
        np.testing.assert_allclose(state_stacked[1].cpu(), state[1].cpu())  # bb
        np.testing.assert_allclose(state_stacked[2].cpu(), state[2].cpu())  # cc

    def test_wkv_cuda_update_consistent(
        self, seed: int = 0, b: int = 2, t: int = 5, c: int = 3
    ):
        rng = np.random.default_rng(seed)

        k = rng.normal(size=(b, t, c))
        v = rng.normal(size=(b, t, c))
        w = rng.uniform(size=(c,))
        u = rng.normal(size=(c,))

        k, v, w, u = (
            torch.tensor(x, dtype=torch.float32, device="cuda") for x in (k, v, w, u)
        )

        _, state = wkv_cuda.wkv(k[:, :-1], v[:, :-1], w, u)
        it_val, it_state = wkv_cuda.wkv_update(k[:, -1:], v[:, -1:], w, u, state)

        block_val, block_state = wkv_cuda.wkv(k, v, w, u)
        np.testing.assert_allclose(it_val.cpu(), block_val[:, -1:].cpu())
        np.testing.assert_allclose(it_state[0].cpu(), block_state[0].cpu())  # aa
        np.testing.assert_allclose(it_state[1].cpu(), block_state[1].cpu())  # bb
        np.testing.assert_allclose(it_state[2].cpu(), block_state[2].cpu())  # bb

    def test_wkv_triton_jax_consistent(
        self, seed: int = 0, b: int = 1, t: int = 8, c: int = 1
    ):
        rng = np.random.default_rng(seed)

        k = rng.normal(size=(b, t, c))
        v = rng.normal(size=(b, t, c))
        w = rng.uniform(size=(c,))
        u = rng.normal(size=(c,))

        k, v, w, u = (
            torch.tensor(x, dtype=torch.float32, device="cuda") for x in (k, v, w, u)
        )
        out_jax, state_jax = wkv_jax.wkv(k, v, w, u)
        out_triton, state_triton = wkv.wkv(k, v, w, u)
        np.testing.assert_allclose(out_jax.cpu(), out_triton.cpu(), rtol=1e-6)
        np.testing.assert_allclose(state_jax[0].cpu(), state_triton[0].cpu(), rtol=1e-6)
        np.testing.assert_allclose(state_jax[1].cpu(), state_triton[1].cpu(), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
