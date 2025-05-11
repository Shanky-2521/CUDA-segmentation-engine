import torch
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Check CUDA device
print("\nCUDA Device Info:")
print("Device name:", torch.cuda.get_device_name(0))
print("Device capability:", torch.cuda.get_device_capability(0))
print("Total memory:", torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB")

# Test tensor operations
device = torch.device("cuda")
print("\nTesting tensor operations...")

# Create a large tensor
size = (512, 512, 512)
print(f"Creating tensor of size {size}")
start = time.time()
tensor = torch.randn(size, device=device)
print(f"Time to create tensor: {time.time() - start:.2f} seconds")

# Perform a matrix multiplication
print("\nTesting matrix multiplication...")
start = time.time()
result = torch.matmul(tensor, tensor)
print(f"Time for matrix multiplication: {time.time() - start:.2f} seconds")

# Test memory usage
print("\nTesting memory usage...")
print("Current memory allocated:", torch.cuda.memory_allocated() / (1024**3), "GB")
print("Max memory allocated:", torch.cuda.max_memory_allocated() / (1024**3), "GB")

# Test pycuda operations
print("\nTesting pycuda operations...")

# Create CUDA stream
stream = cuda.Stream()

# Create a simple CUDA kernel
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
    const int i = threadIdx.x;
    dest[i] = a[i] * b[i];
}
""")

# Get function from module
multiply_them = mod.get_function("multiply_them")

# Create some test data
a = np.random.randn(400).astype(np.float32)
b = np.random.randn(400).astype(np.float32)
dest = np.zeros_like(a)

# Allocate device memory
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
dest_gpu = cuda.mem_alloc(dest.nbytes)

# Copy data to device
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Execute kernel
multiply_them(
    dest_gpu, a_gpu, b_gpu,
    block=(400,1,1),
    grid=(1,1)
)

# Copy result back to host
cuda.memcpy_dtoh(dest, dest_gpu)

# Verify results
print("\nVerifying pycuda results...")
np.testing.assert_allclose(dest, a * b)
print("Pycuda kernel executed correctly!")

# Test TensorRT optimization
try:
    import tensorrt as trt
    print("\nTensorRT is available")
except ImportError:
    print("\nTensorRT is not available")
