import typing as tp
import torch


def add(
    v1: torch.Tensor,
    t1: torch.Tensor,
    v2: torch.Tensor,
    t2: torch.Tensor,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Add two exponentially weighted values.

    Each (vi, ti) represents `exp(ti) * vi`

    Args:
        v1: v of the first ew value
        t1: t of the first ew value
        v2: v of the second ew value
        t2: t of the second ew value
    """
    t_out = torch.logaddexp(t1, t2)
    v1 = evaluate(v1, t1 - t_out)
    v2 = evaluate(v2, t2 - t_out)
    return v1 + v2, t_out


def evaluate(v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Get the value corresponding to the exponentially weighted representation."""
    return torch.exp(t) * v
