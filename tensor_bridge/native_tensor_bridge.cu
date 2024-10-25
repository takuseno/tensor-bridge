#include "native_tensor_bridge.hpp"

void native_copy_tensor(const DataPtr& src_ptr, DataPtr& dst_ptr) {
    cudaSetDevice(src_ptr.device);
    if (src_ptr.device != dst_ptr.device) {
        cudaDeviceEnablePeerAccess(dst_ptr.device, 0);
    }
    cudaMemcpy((void*) src_ptr.ptr, (const void*) dst_ptr.ptr, src_ptr.size, cudaMemcpyDeviceToDevice);
}
