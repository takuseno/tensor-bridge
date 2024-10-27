import time

import jax
import numpy as np
import torch

from tensor_bridge import copy_tensor


def benchmark_copy_tensor() -> None:
    torch_data = torch.rand(1024, 1024, device="cuda:0")
    jax_data = jax.random.uniform(jax.random.key(123), shape=(1024, 1024))

    times = []
    for _ in range(100):
        start = time.time()
        copy_tensor(jax_data, torch_data)
        copy_tensor(torch_data, jax_data)
        times.append(time.time() - start)

    print(f"Average compute time: {sum(times) / len(times)} sec")


def benchmark_copy_via_cpu() -> None:
    jax_data = jax.random.uniform(jax.random.key(123), shape=(1024, 1024))

    times = []
    for _ in range(100):
        start = time.time()
        torch_data = torch.tensor(np.array(jax_data), device="cuda:0")
        jax.numpy.array(torch_data.cpu().numpy())
        times.append(time.time() - start)

    print(f"Average compute time: {sum(times) / len(times)} sec")


def benchmark_dlpack() -> None:
    jax_data = jax.random.uniform(jax.random.key(123), shape=(1024, 1024))

    times = []
    for _ in range(100):
        start = time.time()
        torch_data = torch.from_dlpack(jax.dlpack.to_dlpack(jax_data))
        # jax.dlpack.from_dlpack(torch.to_dlpack(torch_data))
        # dlpack conversion above raises an error as follows:
        # jaxlib.xla_extension.XlaRuntimeError: UNIMPLEMENTED: PJRT C API does not support GetDefaultLayout
        # But, PyTorch's Tensor class implments __dlpack__ property.
        # So this works too.
        jax.dlpack.from_dlpack(torch_data)
        times.append(time.time() - start)

    print(f"Average compute time: {sum(times) / len(times)} sec")


if __name__ == "__main__":
    print("Benchmarking copy_tensor...")
    benchmark_copy_tensor()

    print("Benchmarking copy via CPU...")
    benchmark_copy_via_cpu()

    print("Benchmarking dlpack...")
    benchmark_dlpack()
