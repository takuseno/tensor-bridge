import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tensor_bridge import copy_tensor, copy_tensor_with_assertion


def test_copy_tensor_between_torch() -> None:
    a = torch.rand(2, 3, 4, device="cuda:0")
    b = torch.rand(2, 3, 4, device="cuda:0")
    c = torch.zeros_like(a)
    c.copy_(a)
    assert not torch.all(a == b)

    copy_tensor(b, a)

    assert torch.all(a == b)
    assert torch.all(a == c)


def test_copy_tensor_between_jax() -> None:
    key1 = jax.random.key(123)
    key2 = jax.random.key(321)
    a = jax.random.uniform(key1, shape=(2, 3, 4))
    b = jax.random.uniform(key2, shape=(2, 3, 4))
    c = a.copy()

    assert not jnp.all(a == b)

    copy_tensor(b, a)

    assert jnp.all(a == b)
    assert jnp.all(a == c)


def test_copy_tensor_between_torch_and_jax() -> None:
    torch_data = torch.rand(2, 3, 4, device="cuda:0")

    key = jax.random.key(123)
    jax_data = jax.random.uniform(key, shape=(2, 3, 4))
    jax_data_copy = jax_data.copy()

    assert not np.all(torch_data.cpu().numpy() == np.array(jax_data))

    copy_tensor(torch_data, jax_data)

    assert np.all(torch_data.cpu().numpy() == np.array(jax_data))
    assert np.all(np.array(jax_data) == np.array(jax_data_copy))


def test_copy_tensor_with_assertion() -> None:
    # same layout
    a = torch.rand(2, 3, 4, device="cuda:0")
    b = torch.rand(2, 3, 4, device="cuda:0")
    assert not torch.all(a == b)
    copy_tensor_with_assertion(a, b)
    assert torch.all(a == b)

    # different layout
    a = torch.rand(2, 3, device="cuda:0")
    b = torch.rand(3, 2, device="cuda:0").transpose(0, 1)
    with pytest.raises(AssertionError):
        copy_tensor_with_assertion(b, a)

    # make it contiguous
    b = b.contiguous()
    copy_tensor_with_assertion(b, a)
    assert torch.all(a == b)
