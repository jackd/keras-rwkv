import unittest
import numpy as np
import jax.numpy as jnp

from keras_rwkv.backend.jax import wkv as wkv_ops


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


def wkv_from_cuda(k, v, w, u):
    B, T, C = k.shape
    aa = np.zeros((B, C), dtype=k.dtype)
    bb = np.zeros((B, C), dtype=k.dtype)
    pp = np.full((B, C), -np.inf, dtype=k.dtype)
    output = np.zeros_like(v)
    for i in range(T):
        kk = k[:, i]
        vv = v[:, i]

        qq = np.maximum(pp, u + kk)
        e1 = np.exp(pp - qq)
        e2 = np.exp(u + kk - qq)
        output[:, i] = (e1 * aa + e2 * vv) / (e1 * bb + e2)

        qq = np.maximum(pp - w, kk)
        e1 = np.exp(pp - w - qq)
        e2 = np.exp(kk - qq)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        pp = qq
    return output


class WkvTestCase(unittest.TestCase):
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
        k, v, w, u = (jnp.asarray(x) for x in (k, v, w, u))
        actual, _ = wkv_ops.wkv(k, v, w, u)

        np.testing.assert_allclose(baseline, actual, rtol=1e-3)

    def test_wkv_is_stable(self, seed: int = 0, b: int = 2, t: int = 128, c: int = 3):
        rng = np.random.default_rng(seed)
        shape = (b, t, c)

        k = rng.normal(size=shape)
        v = rng.normal(size=shape)
        w = rng.normal(size=c)
        u = rng.normal(size=c)

        u += 100

        k, v, w, u = (x.astype(np.float32) for x in (k, v, w, u))

        baseline = wkv_from_cuda(k, v, w, u)
        k, v, w, u = (jnp.asarray(x) for x in (k, v, w, u))
        actual, _ = wkv_ops.wkv(k, v, w, u)

        np.testing.assert_allclose(baseline, actual)

        u -= 200
        baseline = wkv_from_cuda(k, v, w, u)
        assert np.all(np.isfinite(baseline))
        actual, _ = wkv_ops.wkv(k, v, w, u)
        assert np.all(np.isfinite(actual)), actual

        np.testing.assert_allclose(baseline, actual, rtol=2e-3)

    def test_wkv_update(self, seed: int = 0, b: int = 2, t: int = 128, c: int = 3):
        rng = np.random.default_rng(seed)
        shape = (b, t, c)

        k = rng.normal(size=shape)
        v = rng.normal(size=shape)
        w = rng.normal(size=c)
        u = rng.normal(size=c)

        block_vals, block_state = wkv_ops.wkv(k, v, w, u)

        shape = (b, 1, c)
        k0 = rng.normal(size=shape)
        v0 = rng.normal(size=shape)
        next_vals, next_state = wkv_ops.wkv_update(k0, v0, w, u, block_state)
        cat_vals = np.concatenate((block_vals, next_vals), axis=1)

        full_block_vals, full_state = wkv_ops.wkv(
            np.concatenate((k, k0), axis=1), np.concatenate((v, v0), axis=1), w, u
        )

        np.testing.assert_allclose(cat_vals, full_block_vals, rtol=1e-4)
        assert len(next_state) == 2
        assert len(full_state) == 2
        np.testing.assert_allclose(next_state[0], full_state[0], rtol=1e-5)
        np.testing.assert_allclose(next_state[1], full_state[1], rtol=1e-5)

    def test_wkv_update_with_padding(
        self,
        seed: int = 0,
        b: int = 11,
        t: int = 128,
        c: int = 3,
        current_index: int = 13,
    ):
        rng = np.random.default_rng(seed)
        shape = (b, t, c)

        k = rng.normal(size=shape)
        v = rng.normal(size=shape)
        w = rng.uniform(size=c)
        u = rng.normal(size=c)

        block_vals, block_state = wkv_ops.wkv(k, v, w, u, current_index=current_index)
        _, cropped_state = wkv_ops.wkv(
            k[:, : current_index + 1], v[:, : current_index + 1], w, u
        )
        self.assertEqual(len(block_state), len(cropped_state))
        assert len(block_state) == 2
        np.testing.assert_allclose(block_state[0], cropped_state[0], rtol=1e-4)
        np.testing.assert_allclose(block_state[1], cropped_state[1], rtol=1e-4)

        next_vals, next_state = wkv_ops.wkv_update(
            jnp.expand_dims(k[:, current_index + 1], 1),
            jnp.expand_dims(v[:, current_index + 1], 1),
            w,
            u,
            block_state,
        )
        np.testing.assert_allclose(
            next_vals, jnp.expand_dims(block_vals[:, current_index + 1], 1), rtol=1e-4
        )

        _, full_state = wkv_ops.wkv(k, v, w, u, current_index=current_index + 1)

        assert len(next_state) == 2
        assert len(full_state) == 2
        np.testing.assert_allclose(next_state[0], full_state[0], rtol=1e-5)
        np.testing.assert_allclose(next_state[1], full_state[1], rtol=1e-5)

    def test_parallel_consistent(
        self,
        seed: int = 0,
        b: int = 11,
        t: int = 128,
        c: int = 3,
        current_index: int = 13,
    ):
        rng = np.random.default_rng(seed)
        shape = (b, t, c)

        k = rng.normal(size=shape)
        v = rng.normal(size=shape)
        w = rng.uniform(size=c)
        u = rng.normal(size=c)

        k, v, w, u = (jnp.array(x) for x in (k, v, w, u))
        parallel_vals, parallel_state = wkv_ops.wkv(
            k, v, w, u, current_index=current_index, parallel=True
        )
        serial_vals, serial_state = wkv_ops.wkv(
            k, v, w, u, current_index=current_index, parallel=False
        )

        np.testing.assert_allclose(parallel_vals, serial_vals, rtol=2e-4)
        assert len(parallel_state) == 2
        assert len(serial_state) == 2
        np.testing.assert_allclose(parallel_state[0], serial_state[0], rtol=2e-4)
        np.testing.assert_allclose(parallel_state[1], serial_state[1], rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
