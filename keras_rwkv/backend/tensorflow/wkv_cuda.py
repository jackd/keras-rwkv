import typing as tp
import tensorflow as tf

# pylint:disable=no-member


def wkv(
    k: tf.Tensor,
    v: tf.Tensor,
    w: tf.Tensor,
    u: tf.Tensor,
    current_index: tp.Optional[tp.Union[int, tf.Tensor]] = None,
) -> tp.Tuple[tf.Tensor, tp.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    raise NotImplementedError("wkv_cuda currently only implemented with torch backend")


def wkv_update(
    k: tf.Tensor,
    v: tf.Tensor,
    w: tf.Tensor,
    u: tf.Tensor,
    state: tp.Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
) -> tp.Tuple[tf.Tensor, tp.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    raise NotImplementedError("wkv_cuda currently only implemented with torch backend")
