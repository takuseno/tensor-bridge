struct DataPtr {
    unsigned long ptr;
    unsigned long size;
    unsigned int device;
};

void native_copy_tensor(const DataPtr& src_ptr, DataPtr& dst_ptr);
