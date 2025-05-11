import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import time

class CUDAOperations:
    def __init__(self):
        self.stream = cuda.Stream()
        
    def allocate_device_memory(self, size):
        """Allocate device memory"""
        return cuda.mem_alloc(size)
        
    def copy_to_device(self, host_array, device_ptr):
        """Copy data from host to device"""
        cuda.memcpy_htod(device_ptr, host_array)
        
    def copy_to_host(self, device_ptr, host_array):
        """Copy data from device to host"""
        cuda.memcpy_dtoh(host_array, device_ptr)
        
    def async_copy_to_device(self, host_array, device_ptr):
        """Async copy from host to device"""
        cuda.memcpy_htod_async(device_ptr, host_array, self.stream)
        
    def async_copy_to_host(self, device_ptr, host_array):
        """Async copy from device to host"""
        cuda.memcpy_dtoh_async(host_array, device_ptr, self.stream)
        
    def synchronize(self):
        """Synchronize stream"""
        self.stream.synchronize()
        
    def compile_kernel(self, kernel_code):
        """Compile CUDA kernel"""
        mod = SourceModule(kernel_code)
        return mod
        
    def get_device_properties(self):
        """Get CUDA device properties"""
        device = cuda.Device(0)
        return {
            'name': device.name(),
            'total_memory': device.total_memory(),
            'compute_capability': device.compute_capability(),
            'max_threads_per_block': device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        }
        
    def __del__(self):
        """Clean up CUDA resources"""
        if hasattr(self, 'stream'):
            self.stream.synchronize()
            del self.stream

def get_cuda_operations():
    """Get CUDA operations instance"""
    return CUDAOperations()

if __name__ == '__main__':
    # Test CUDA operations
    cuda_ops = get_cuda_operations()
    print("CUDA Device Properties:")
    print(cuda_ops.get_device_properties())
