[![test](https://github.com/takuseno/tensor-bridge/actions/workflows/test.yaml/badge.svg)](https://github.com/takuseno/tensor-bridge/actions/workflows/test.yaml)

# tensor-bridge
`tensor-bridge` is a light-weight library that achieves inter-library tensor transfer by native `cudaMemcpy` call with minimal overheads.

```py
import torch
import jax
from tensor_bridge import copy_tensor


# PyTorch tensor
torch_data = torch.rand(2, 3, 4, device="cuda:0")

# Jax tensor
jax_data = jax.random.uniform(jax.random.key(123), shape=(2, 3, 4))

# Copy Jax tensor to PyTorch tensor
copy_tensor(torch_data, jax_data)

# And, other way around
copy_tensor(jax_data, torch_data)
```

:warning: Currently, this repository is under active development. Especially, transfer between different layout of tensors is not implemented yet. I recommend to try `copy_tensor_with_assertion` before starting experiments. `copy_tensor_with_assertion` will raise an error if copy doesn't work.
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
- nnabla


## Installation
### PyPI
If pip installation doesn't work, please try installation from source code.

#### Python 3.10.x
You can install a pre-built package.
```
pip install tensor-bridge
```

#### Other Python version
Your macine needs to install `nvcc` to compile a native code and `Cython` to compile `.pyx` files.
```
pip install Cython==0.29.36
pip install tensor-bridge
```
Pre-built packages for other Python versions are in progress.


### From souce code
Your macine needs to install `nvcc` to compile a native code and `Cython` to compile `.pyx` files.
```
git clone git@github.com:takuseno/tensor-bridge
cd tensor-bridge
pip install Cython==0.29.36
pip install -e .
```

## Unit test
Your machine needs to install NVIDIA's GPU and nvidia-driver to execute tests.
```
./bin/build-docker
./bin/test
```

## Benchmark
To benchmark round trip copies between Jax and PyTorch:
```
./bin/build-docker
./bin/benchmark
```

This is result with my local desktop with RTX4070.
```
Benchmarking copy_tensor...
Average compute time: 1.3043880462646485e-05 sec
Benchmarking copy via CPU...
Average compute time: 0.0016725873947143555 sec
Benchmarking dlpack...
Average compute time: 7.467031478881836e-05 sec
```
`copy_tensor` is surprisingly faster than DLPack. Looking at PyTorch's implementation, it seems that PyTorch does additional CUDA stream synchronization, which adds additional compute time.
