"""
This script demonstrates that the custom gradient implemented in torch is consistent
with the vjp from the jax implementation.
"""
import functools
import jax
import jax.numpy as jnp
import numpy as np
from keras_rwkv.backend.jax import ew

seed = 0
n = 5
reverse = True

rng = np.random.default_rng(seed)

v_in = rng.normal(size=n).astype(np.float32)
t_in = rng.normal(size=n).astype(np.float32)

grad_t_out = rng.normal(size=(n,)).astype(np.float32)
grad_v_out = rng.normal(size=(n,)).astype(np.float32)

(v_out, t_out), vjp_fn = jax.vjp(
    functools.partial(ew.cumsum, reverse=reverse), v_in, t_in
)
grad_v_expected, grad_t_expected = vjp_fn((grad_v_out, grad_t_out))

# below this is a jax port of the torch custom gradient
v_temp, t_temp = ew.cumsum(grad_v_out, -t_out, reverse=not reverse)
grad_v_in = ew.evaluate(v_temp, t_temp + t_in)

q = grad_t_out - grad_v_out * v_out
v_temp, t_temp = ew.cumsum(q, -t_out, reverse=not reverse)
grad_t_via_t_out = ew.evaluate(v_temp, t_temp + t_in)

grad_t_via_v_out = grad_v_in * v_in
grad_t_in = grad_t_via_t_out + grad_t_via_v_out

print(jnp.max(jnp.abs(grad_v_in - grad_v_expected)))
print(jnp.max(jnp.abs(grad_t_in - grad_t_expected)))
