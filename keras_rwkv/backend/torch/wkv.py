import typing as tp
import torch
from . import ew

# pylint:disable=no-member


def wkv(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    current_index: tp.Optional[tp.Union[int, torch.Tensor]] = None,
    parallel: bool = True,
) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        k: [B, T, C]
        v: [B, T, C]
        w: [C]
        u: [C]

    Returns:
        [B, T, C]
    """
    sequence_length = k.shape[1]
    offset = torch.arange(-sequence_length + 1, 1, dtype=k.dtype, device=k.device)
    k = k + offset.unsqueeze(-1) * w
    v_acc, t_acc = ew.cumsum(v, k, axis=1, parallel=parallel)

    if current_index is None:
        current_index = sequence_length - 1
    v_next = v_acc[:, current_index].unsqueeze(1)
    t_next = t_acc[:, current_index].unsqueeze(1)
    t_next = t_next + w * (sequence_length - 1 - current_index)
    state_next = (v_next, t_next)
    v_acc = v_acc[:, :-1]
    t_acc = t_acc[:, :-1]
    # split values. First goes straight to output, rest are used with lagged acc.
    v_start = v[:, :1]
    v_rest = v[:, 1:]
    v_out, t_out = ew.add(v_acc, t_acc, v_rest, k[:, 1:] + u - w)
    del t_out  # exp(t_out) is the denominator term
    v_out = torch.concatenate((v_start, v_out), dim=1)
    return v_out, state_next


# def wkv(
#     k: torch.Tensor,
#     v: torch.Tensor,
#     w: torch.Tensor,
#     u: torch.Tensor,
#     current_index: tp.Optional[tp.Union[int, torch.Tensor]] = None,
#     parallel: bool = True,
# ) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
#     """
#     Args:
#         k: [B, T, C]
#         v: [B, T, C]
#         w: [C]
#         u: [C]

#     Returns:
#         [B, T, C]
#     """
#     sequence_length = k.shape[1]
#     offset = torch.arange(-sequence_length + 1, 1, dtype=k.dtype, device=k.device)
#     k = k + offset.unsqueeze(-1) * w

#     # v_acc, t_acc = ew.cumsum(v, k, axis=1)
#     v_swapped = torch.swapaxes(v, 1, 2)
#     k_swapped = torch.swapaxes(k, 1, 2)
#     v_acc, t_acc = ew.cumsum(v_swapped, k_swapped, axis=2, parallel=parallel)
#     v_acc = torch.swapaxes(v_acc, 1, 2)
#     t_acc = torch.swapaxes(t_acc, 1, 2)

#     sequence_length = k.shape[1]
#     if current_index is None:
#         v_acc, v_next = torch.split(v_acc, (sequence_length - 1, 1), dim=1)
#         t_acc, t_next = torch.split(t_acc, (sequence_length - 1, 1), dim=1)
#     else:
#         current_index = torch.where(
#             current_index < 0, current_index + sequence_length, current_index
#         )
#         v_next = v_acc[:, current_index]
#         t_next = t_acc[:, current_index]
#         # undo effect of k offset
#         t_next = t_next + w * (sequence_length - 1 - current_index)
#         v_next = torch.unsqueeze(v_next, 1)
#         t_next = torch.unsqueeze(t_next, 1)
#         v_acc = v_acc[:, :-1]
#         t_acc = t_acc[:, :-1]
#     state_next = (v_next, t_next)
#     # split values. First goes straight to output, rest are used with lagged acc.
#     v_start, v_rest = torch.split(v, (1, sequence_length - 1), dim=1)
#     # -w below is necessary to accommodate for k offset + lag
#     v_out, t_out = ew.add(v_acc, t_acc, v_rest, k[:, 1:] + u - w)
#     del t_out  # exp(t_out) is the denominator term
#     v_out = torch.concatenate((v_start, v_out), axis=1)
#     return v_out, state_next


def wkv_update(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    state: tp.Tuple[torch.Tensor, torch.Tensor],
) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
    v_acc, t_acc = state
    v_out, t_out = ew.add(v_acc, t_acc, v, k + u)
    del t_out  # t_out is just the denominator
    state_next = ew.add(v_acc, t_acc - w, v, k)
    return v_out, state_next
