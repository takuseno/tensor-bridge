import numpy as np

from .imports import get_jax, get_torch
from .types import Array, NumpyArray

__all__ = ["get_numpy_data"]


torch = get_torch()
jax = get_jax()


def get_numpy_data(tensor: Array) -> NumpyArray:
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()  # type: ignore
    elif jax is not None and isinstance(tensor, jax.Array):
        return np.array(tensor)
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")
