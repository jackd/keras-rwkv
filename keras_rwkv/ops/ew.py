import typing as tp
from keras_core import Operation, KerasTensor
from ..backend import ew as ew_backend, Array

# pylint:disable=arguments-differ


class ExponentiallyWeighted(tp.NamedTuple):
    v: Array
    t: Array

    def __add__(self, other: "ExponentiallyWeighted") -> "ExponentiallyWeighted":
        assert isinstance(other, ExponentiallyWeighted)
        return add(self, other)


class ExponentiallyWeightedAdd(Operation):
    def compute_output_spec(self, x1: ExponentiallyWeighted, x2: ExponentiallyWeighted):
        assert isinstance(x1, tuple) and len(x1) == 2
        assert isinstance(x2, tuple) and len(x2) == 2

        assert x1[0].shape == x2[0].shape
        assert x1[0].dtype == x2[0].dtype
        assert x1[1].shape == x2[1].shape
        assert x1[1].dtype == x2[1].dtype
        v = x1[0]
        v = KerasTensor(shape=v.shape, dtype=v.dtype, name="v")
        t = KerasTensor(shape=v.shape, dtype=v.dtype, name="t")
        return v, t

    def call(
        self, x1: ExponentiallyWeighted, x2: ExponentiallyWeighted
    ) -> ExponentiallyWeighted:
        return ExponentiallyWeighted(*ew_backend.add(*x1, *x2))


def add(x1: ExponentiallyWeighted, x2: ExponentiallyWeighted) -> ExponentiallyWeighted:
    return ExponentiallyWeightedAdd()(x1, x2)


class ExponentiallyWeightedCumsum(Operation):
    def __init__(
        self,
        axis: int,
        reverse: bool,
        parallel: bool,
        name: tp.Optional[str] = None,
    ):
        self.axis = axis
        self.reverse = reverse
        self.parallel = parallel
        super().__init__(name=name)

    def compute_output_spec(self, x):
        assert isinstance(x, tuple) and len(x) == 2

        x0 = x[0]
        v = KerasTensor(x0.shape, x0.dtype, name="v")
        t = KerasTensor(x0.shape, x0.dtype, name="t")
        return ExponentiallyWeighted(v, t)

    def call(self, x):
        return ExponentiallyWeighted(
            *ew_backend.cumsum(
                *x, axis=self.axis, reverse=self.reverse, parallel=self.parallel
            )
        )


def cumsum(
    x: tp.Tuple[Array, Array],
    axis: int = 0,
    reverse: bool = False,
    parallel: bool = True,
) -> ExponentiallyWeighted:
    return ExponentiallyWeightedCumsum(axis=axis, reverse=reverse, parallel=parallel)(x)
