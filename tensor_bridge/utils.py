import jax
import numpy as np
import torch

from .types import Array, NumpyArray

__all__ = ["get_numpy_data"]


def get_numpy_data(tensor: Array) -> NumpyArray:
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()  # type: ignore
    elif isinstance(tensor, jax.Array):
        return np.array(tensor)
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")
