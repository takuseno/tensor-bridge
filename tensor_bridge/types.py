from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:
    import jax
    import torch


__all__ = ["NumpyArray", "Array"]


NumpyArray = np.ndarray[Any, Any]
Array = Union["torch.Tensor", "jax.Array"]
