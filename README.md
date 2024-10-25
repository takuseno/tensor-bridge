[![test](https://github.com/takuseno/tensor-bridge/actions/workflows/test.yaml/badge.svg)](https://github.com/takuseno/tensor-bridge/actions/workflows/test.yaml)

# tensor-bridge
`tensor-bridge` is a light-weight library that achieves inter-library tensor transfer by native `cudaMemcpy` call for minimal overheads.

```py
import torch
import jax
from tensor_bridge import copy_tensor


# PyTorch tensor
torch_data = torch.rand(2, 3, 4, device="cuda:0")

# Jax tensor
jax_data = jax.random.uniform(jax.random.key(123), shape=(2, 3, 4))

# Copy PyTorch tensor to Jax tensor
copy_tensor(torch_data, jax_data)

# And, other way around
copy_tensor(jax_data, torch_data)
```

## Supported deep learning libraries
- PyTorch
- Jax


## Installation
Your macine needs to install `nvcc` to compile a native code.
```
pip install git+https://github.com/takuseno/tensor-bridge
```
Pre-built package release is in progress.


## Unit test
You machine needs to install NVIDIA's GPU and nvidia-driver to execute tests.
```
./bin/build-docker
./bin/test
```
