import typing as tp
import triton
import triton.language as tl


@triton.jit
def log1p(x: tl.tensor) -> tl.tensor:
    return tl.log(x + 1)


@triton.jit
def logaddexp(x1: tl.tensor, x2: tl.tensor) -> tl.tensor:
    amax = tl.maximum(x1, x2)
    delta = tl.abs(x1 - x2)
    return amax + log1p(tl.exp(-delta))


@triton.jit
def add(
    v1: tl.tensor, t1: tl.tensor, v2: tl.tensor, t2: tl.tensor
) -> tp.Tuple[tl.tensor, tl.tensor]:
    t_out = logaddexp(t1, t2)
    v_out = tl.exp(t1 - t_out) * v1 + tl.exp(t2 - t_out) * v2
    return v_out, t_out
