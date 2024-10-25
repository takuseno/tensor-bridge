cdef extern from "native_tensor_bridge.hpp" nogil:
    cdef struct DataPtr:
        unsigned long ptr
        unsigned long size
        unsigned int device
    cdef void native_copy_tensor(DataPtr& src, DataPtr& dst)
