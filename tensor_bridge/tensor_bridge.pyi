from typing import Union

import torch
import jax


Array = Union[torch.Tensor, jax.Array]

def copy_tensor(src: Array, dst: Array) -> None: ...
