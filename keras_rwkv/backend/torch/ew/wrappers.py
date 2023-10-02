import typing as tp
import torch
from .parallel import cumsum as _cumsum_parallel
from .serial import cumsum as _cumsum_serial
from .utils import evaluate


def _cumsum(v: torch.Tensor, t: torch.Tensor, axis: int, reverse: bool, parallel: bool):
    if reverse:
        v = torch.flip(v, (axis,))
        t = torch.flip(t, (axis,))
    if parallel:
        v, t = _cumsum_parallel(v, t, axis)
    else:
        v, t = _cumsum_serial(v, t, axis)
    if reverse:
        v = torch.flip(v, (axis,))
        t = torch.flip(t, (axis,))
    return v, t


class ExponentiallyWeightedCumsum(  # pylint:disable=abstract-method
    torch.autograd.Function
):
    @staticmethod
    def forward(  # pylint:disable=arguments-differ
        ctx,
        v: torch.Tensor,
        t: torch.Tensor,
        axis: int,
        reverse: bool,
        parallel: bool,
    ):
        ctx.axis = axis
        ctx.reverse = reverse
        ctx.parallel = parallel
        v_out, t_out = _cumsum(v, t, axis=axis, reverse=reverse, parallel=parallel)
        ctx.save_for_backward(v, t, v_out, t_out)
        return v_out, t_out

    @staticmethod
    def backward(ctx, grad_v_out, grad_t_out):  # pylint:disable=arguments-differ
        v_in, t_in, v_out, t_out = ctx.saved_tensors
        kwargs = {
            "axis": ctx.axis,
            "reverse": not ctx.reverse,
            "parallel": ctx.parallel,
        }

        v_temp, t_temp = cumsum(grad_v_out, -t_out, **kwargs)
        grad_v_in = evaluate(v_temp, t_temp + t_in)

        q = grad_t_out - grad_v_out * v_out
        v_temp, t_temp = cumsum(q, -t_out, **kwargs)
        grad_t_via_t_out = evaluate(v_temp, t_temp + t_in)

        grad_t_via_v_out = grad_v_in * v_in
        grad_t_in = grad_t_via_t_out + grad_t_via_v_out
        return grad_v_in, grad_t_in, None, None, None


def cumsum(
    v: torch.Tensor,
    t: torch.Tensor,
    axis: int = 0,
    reverse: bool = False,
    parallel: bool = False,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cumulative summation of exponentially weighted values.

    Exponentially weighted values (v, t) represent `exp(t)*v`.

    Args:
        v: same shape as t
        t: same shape as v

    Returns:
        (v_out, t_out) structure, same shape/dtype as inputs.
    """
    return ExponentiallyWeightedCumsum.apply(v, t, axis, reverse, parallel)
