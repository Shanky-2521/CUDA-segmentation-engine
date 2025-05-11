import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2

def create_test_kernel():
    # Create CUDA kernel
    mod = SourceModule("""
    __global__ void process_image(float *input, float *output, int width, int height)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < width * height * 3) {
            int c = idx % 3;
            int pixel_idx = idx / 3;
            int x = pixel_idx % width;
            int y = pixel_idx / width;
            
            // Simple processing - just copy with some modification
            float val = input[idx];
            output[idx] = val * 1.1f; // Simple amplification
        }
    }
    """)
    
    # Get kernel function
    kernel = mod.get_function("process_image")
    
    # Create test image
    width, height = 1024, 512
    image = np.random.rand(height, width, 3).astype(np.float32)
    
    # Allocate device memory
    d_input = cuda.mem_alloc(image.nbytes)
    d_output = cuda.mem_alloc(image.nbytes)
    
    # Copy input to device
    cuda.memcpy_htod(d_input, image)
    
    # Launch kernel
    block_size = (1024, 1, 1)
    grid_size = ((width * height * 3 + block_size[0] - 1) // block_size[0], 1, 1)
    
    kernel(d_input, d_output, np.int32(width), np.int32(height),
           block=block_size, grid=grid_size)
    
    # Copy result back
    output = np.empty_like(image)
    cuda.memcpy_dtoh(output, d_output)
    
    # Clean up
    d_input.free()
    d_output.free()
    
    # Convert to uint8 for visualization
    output = (output * 255).astype(np.uint8)
    
    # Save result
    cv2.imwrite('processed_image.png', output)
    print("Created processed_image.png")

if __name__ == '__main__':
    create_test_kernel()
