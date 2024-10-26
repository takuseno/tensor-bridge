import numpy as np

from ._version import __version__
from .tensor_bridge import copy_tensor
from .types import Array
from .utils import get_numpy_data

__all__ = ["copy_tensor", "copy_tensor_with_assertion", "__version__"]


def copy_tensor_with_assertion(src: Array, dst: Array) -> None:
    copy_tensor(src, dst)
    assert np.all(
        get_numpy_data(src) == get_numpy_data(dst)
    ), "Copied tensor doesn't match the source tensor. Layout of tensors can be different."
