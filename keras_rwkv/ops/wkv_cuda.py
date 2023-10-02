import typing as tp
from keras_core import Operation, KerasTensor
from ..backend import wkv_cuda as wkv_cuda_backend, Array


class WKVCuda(Operation):
    def compute_output_spec(
        self, k: Array, v: Array, w: Array, u: Array, current_index=None
    ):
        b, t, c = k.shape
        dtype = k.dtype
        out = KerasTensor((b, t, c), dtype, name="wkv")
        state_next = (
            KerasTensor((b, c), dtype, name="aa"),
            KerasTensor((b, c), dtype, name="pp"),
            KerasTensor((b, c), dtype, name="pp"),
        )
        return out, state_next

    def call(
        self,
        k: Array,
        v: Array,
        w: Array,
        u: Array,
        current_index: tp.Optional[tp.Union[int, Array]],
    ):
        return wkv_cuda_backend.wkv(k, v, w, u, current_index=current_index)


def wkv(
    k: Array,
    v: Array,
    w: Array,
    u: Array,
    current_index: tp.Optional[tp.Union[int, Array]] = None,
):
    return WKVCuda()(k, v, w, u, current_index=current_index)


wkv.__doc__ = wkv_cuda_backend.wkv.__doc__


class WKVCudaUpdate(Operation):
    def compute_output_spec(self, k, v, w, u, state):
        b, t, c = k.shape
        assert t == 1
        dtype = k.dtype
        state_out = (KerasTensor((b, 1, c), dtype), KerasTensor((b, 1, c), dtype))
        return KerasTensor((b, 1, c), dtype), state_out

    def call(self, k, v, w, u, state):
        return wkv_cuda_backend.wkv_update(k, v, w, u, state)


def wkv_update(k, v, w, u, state):
    return WKVCudaUpdate()(k, v, w, u, state)


wkv_update.__doc__ = wkv_cuda_backend.wkv.__doc__
