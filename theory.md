# WKV with Exponentially Weighted Cumulative Sums

## WKV as a ratio of cumulative sums

From the [original paper](https://arxiv.org/abs/2305.13048) $wkv$ (which we will denote $z^\prime$) is defined as

$$
z^\prime_t = \frac{\sum_{i=1}^{t-1}\exp(-(t - 1 - i)w + k_i) v_i + \exp(u + k_t) v_t}{\sum_{i=1}^{t-1}\exp(-(t - 1 - i)w + k_i) + \exp(u + k_t)}.
$$

Multiplying top and bottom by $\exp((t - 1)w)$ yields

$$
z^\prime_t = \frac{\sum_{i=1}^{t-1} \exp(k_i + i w)v_i + \exp(u - w + k_t + t_w)v_t}{\sum_{i=1}^{t-1} \exp(k_i + i w) + \exp(u - w + k_t + t_w)}.
$$

If we let $\tilde{k}_n = k_n + n w$, this simplifies to

$$
z^\prime_t = \frac{\sum_{i=1}^{t-1} \exp(\tilde{k}_i)v_i + \exp(u - w + \tilde{k}_t)v_t}{\sum_{i=1}^{t-1} \exp(\tilde{k}_i) + \exp(u - w + \tilde{k}_t)}.
$$

This can be computed efficiently using a cumulative sum.

```python
import jax.numpy as jnp

def wkv_numerator(
    v, # [T, C]
    k, # [T, C]
    u, # [C]
    w, # [C]
):
    T, C = v.shape
    kt = k + jnp.arange(T, dtype=k.dtype)[:, None] * w
    numer_acc = jnp.cumsum(jnp.exp(kt) * v, axis=0)
    numer_offset = jnp.exp(u - w + kt) * v
    numer = numer_acc[:-1] + numer_offset[1:]
    return jnp.concatenate((v[:1], numer), axis=0)

def wkv(v, k, u, w):
    return wkv_numerator(v, k, u, w) / wkv_numerator(jnp.ones_like(v), k, u, w)
```

There are multiple benefits to this include:

- simplicity: no custom cuda kernels or hand-written backward passes; and
- parallelism: `cumsum` can be parallelized along the `T` dimension.

The major downside is that computing `exp(kt)` is numerically infeasible for long time sequences. To resolve this, we introduce an _exponentially weighted_ parameterization.

## Exponentially Weighted Parameterization

We define an exponentially weighted parameterization of a value $z$ as

$
z = \exp(t) v,
$

where we assume $t$ and $v$ are both $\mathcal{O}(1)$. Due to the exponential however, the scales of $z$ can vary dramatically. We can add two exponentially weighted values and return the exponentially weighted parameterization without explicitly evaluating either of them,

```python
import jax.numpy as jnp

def add(z1, z2):
    v1, t1 = z1
    v2, t2 = z2
    t_out = jnp.logaddexp(t1, t2)
    v_out = jnp.exp(t1 - t_out) * v1 + jnp.exp(t2 - t_out) * v2
    return v_out, t_out
```

## Exponentially Weighted WKV

To make out `wkv` implementation numerically stable, we simply replace the `cumsum` with a version that supports a custom `add` operation - `jax.lax.associative_scan`. Note the resulting exponentially weighted values have `t` values corresponding to the denominator in the original expression, so there's no need to compute a separate denominator.

```python
def wkv(v, k, w, u):
    sequence_length = k.shape[1]
    kt = k + jnp.arange(sequence_length)[:, None] * w
    v_acc, t_acc = jax.lax.assoociative_scan(add, (v, kt), axis=0)
    v_out, t_out = add((v_acc[:-1], t_acc[:-1]), (v[1:], u - w + kt))
    return jnp.concatenate((v[:1], v_out), axis=0)
```

Note that `associative_scan` (a.k.a. `prefix_sum`) is a fundamental operation in computer science that has been extensively studied. In particular, it is worth noting that work-efficient parallel implementations exist and are available in `cuda`, `jax`, `triton` (currently only nightly) and `tensorflow-probability`. That said, it is worth noting that if sufficient parallelism is possible across other dimensions of the data (in this case, the batch/channel dimension), then there may not be sufficient cores available to speed things up across the time dimension. In this case, the parallel implementation may result in a slow down rather than a speed up - but this depends significantly on the available hardware and training setup.

## Associative Scan / Prefix Sum Resources

- [nvidia guide](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [thrust documentation](https://thrust.github.io/doc/group__prefixsums.html) (included in cuda)
- [jax implementation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html)
- [tensorflow_probability implementation](https://www.tensorflow.org/probability/api_docs/python/tfp/math/scan_associative)
- [merged triton PR adding support](https://github.com/openai/triton/pull/1858)
- [pytorch feature request](https://github.com/pytorch/pytorch/issues/95408)
