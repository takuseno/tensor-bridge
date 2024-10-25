from typing import Any, Union

import jax
import numpy as np
import torch

__all__ = ["NumpyArray", "Array"]


NumpyArray = np.ndarray[Any, Any]
Array = Union[torch.Tensor, jax.Array]
