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

:warning: Currently, this repository is under active development. Espeically, transfer between different layout of tensors are not implemented yet. I recommend to try `copy_tensor_with_assertion` before starting experiments. `copy_tensor_with_assertion` will raise an error if copy doesn't work.
If `copy_tensor_with_assertion` raises an error, you need to force the tensor to be contiguous:
```py
# PyTorch example

# different layout raises an error
a = torch.rand(2, 3, device="cuda:0")
b = torch.rand(3, 2, device="cuda:0").transpose(0, 1)
copy_tensor_with_assertion(a, b)  # AssertionError !!

# make both tensors contiguous layout
b = b.contiguous()
copy_tensor_with_assertion(a, b)
```
Since `copy_tensor_with_assertion` does additional GPU-CPU transfer internally, make sure that you switch to `copy_tensor` in your experiments. Otherwise your training loop will be significantly slower.

## Features
- Fast inter-library tensor copy.
- Inter-GPU copy (I believe this is supported with the current implementation. But, not tested yet.)

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
