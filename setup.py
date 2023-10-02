import os

from setuptools import find_packages, setup

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = "0"
_MINOR_VERSION = "0"
_PATCH_VERSION = "1"

with open(
    os.path.join(os.path.dirname(__file__), "requirements.txt"), "r", encoding="utf8"
) as fp:
    install_requires = fp.read().split("\n")

setup(
    name="keras-rwkv",
    description="RWKV language model in keras",
    url="https://github.com/jackd/keras-rwkv",
    author="Dominic Jack",
    author_email="thedomjack@gmail.com",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=install_requires,
    zip_safe=True,
    python_requires=">=3.6",
    version=".".join([_MAJOR_VERSION, _MINOR_VERSION, _PATCH_VERSION]),
)
