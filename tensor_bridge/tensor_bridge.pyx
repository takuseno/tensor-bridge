from .imports import get_jax, get_torch

from _tensor_bridge cimport DataPtr, native_copy_tensor
from libcpp.pair cimport pair

torch = get_torch()
jax = get_jax()


cdef DataPtr get_ptr_and_size(data):
    cdef unsigned long ptr
    cdef unsigned long size
    cdef DataPtr ret
    if torch is not None and isinstance(data, torch.Tensor):
        ret.ptr = data.data_ptr()
        ret.size = torch.numel(data) * data.element_size()
        ret.device = data.device.index
    elif jax is not None and isinstance(data, jax.Array):
        ret.ptr = data.unsafe_buffer_pointer()
        ret.size = data.size * data.dtype.itemsize
        ret.device = next(iter(data.devices())).id
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    return ret


def copy_tensor(src, dst):
    assert src.shape == dst.shape, f"Shape doesn't match: src={src.shape}, dst={dst.shape}"
    src_ptr = get_ptr_and_size(src)
    dst_ptr = get_ptr_and_size(dst)
    assert src_ptr.size == dst_ptr.size, f"Tensor size doesn't match: src={src_ptr.size}, dst={dst_ptr.size}"
    native_copy_tensor(src_ptr, dst_ptr)
