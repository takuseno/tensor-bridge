import jax
import numpy as np
import torch

from tensor_bridge.utils import get_numpy_data


def test_get_numpy_data_with_torch() -> None:
    tensor = torch.rand(2, 3, 4)
    assert np.all(tensor.numpy() == get_numpy_data(tensor))


def test_get_numpy_data_with_jax() -> None:
    tensor = jax.random.uniform(jax.random.key(123), shape=(2, 3, 4))
    assert np.all(np.array(tensor) == get_numpy_data(tensor))
