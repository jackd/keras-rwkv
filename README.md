# keras-rwkv

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Multi-backend [keras](https://github.com/keras-team/keras-core) implementation of RWKV.

This is a port of the models in the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) package for keras. I claim no credit for the network design, though I do offer a novel implementation based on a `cumsum` variant.

See also:

- [theory](./theory.md) for how this formulation works; and
- [performance](./performance.md) for how these implementations compare to the implementation shipped with the `rwkv` pip package.

![Implementation forward pass times with 256 channels](./images/benchmark-256.png)

This repository has no affiliation with `keras-team` - it's just a `keras-core` / `keras-nlp` implementation.

## Installation

This package could admittedly do with a cleaner installation process - that's a non-trivial amount of work though, because the required packages depend on what backend you want to use. For the time being, the following should be enough to cover most cases:

- all will require `keras-core`, `keras-nlp` and (for the moment) `tensorflow` (even if using other backends)
- to use tensorflow's parallel implementation you'll need `tensorflow-probability`
- to use torch's `original_cuda` implementation you'll need `rwkv`
- to use `jax` implementations wrapped with `torch` backend you'll need `jax2torch`
- to use `torch`'s parallel triton implementation you'll need `triton-nightly` (see [here](https://github.com/openai/triton#quick-installation) for installation instructions)

If errors occur with tensorflow backend, try installing nightly versions of things.

Getting all backends to work in the same environment is non-trivial. I had success using `conda` to install `jax` and pip for `tensorflow`/`torch` (following conda installation instructions for `tensorflow`/`torch` tends to break `jax` installations).

Installing this package can be done via

```bash
git clone https://github.com/jackd/keras-rwkv.git
pip install -e keras-rwkv
```

Note there are fully independent backend implementations for `wkv` and exponentially weighted (`ew`) cumsum

- torch
  - [triton wkv](./keras_rwkv/backend/torch/wkv.py)
  - [triton ew](./keras_rwkv/backend/torch/ew/wrappers.py)
- jax
  - [wkv](./keras_rwkv/backend/jax/wkv.py)
  - [ew](./keras_rwkv/backend/jax/ew.py)
- tensorflow
  - [wkv](./keras_rwkv/backend/tensorflow/wkv.py)
  - [ew](./keras_rwkv/backend/tensorflow/ew.py)

Note the standard `keras` implementation only requires `ew.cumsum` custom backend implementations (see [ops/wkv.py](./keras_rwkv/ops/wkv.py)) - the rest can be done via `keras`. The `wkv` implementations in the individual backends are provided mostly as a convenience for anyone who wants to take them and use them externally in a non-keras environment.

## Quickstart

See [examples](./examples/) directory for basic usage. It is strongly recommended that you set the `KERAS_BACKEND` environment variable - failure to do so will revert to using `tf.keras`, which isn't nearly as well tested.

```bash
KERAS_BACKEND=jax python examples/generate.py
```

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
