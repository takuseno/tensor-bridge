from typing import Union

import jax
import torch

Array = Union[torch.Tensor, jax.Array]

def copy_tensor(src: Array, dst: Array) -> None: ...
