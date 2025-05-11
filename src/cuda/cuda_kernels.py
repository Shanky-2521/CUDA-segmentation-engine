import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

class CUDAInferenceKernels:
    def __init__(self):
        # CUDA kernel for preprocessing
        self.preprocess_kernel = SourceModule("""
        __global__ void preprocess(float *input, float *output, float *mean, float *std)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < 3 * 512 * 1024) {
                int c = idx / (512 * 1024);
                int h = (idx % (512 * 1024)) / 1024;
                int w = idx % 1024;
                
                float pixel = input[idx];
                float normalized = (pixel / 255.0 - mean[c]) / std[c];
                output[idx] = normalized;
            }
        }
        """)
        
        # CUDA kernel for postprocessing
        self.postprocess_kernel = SourceModule("""
        __global__ void postprocess(float *input, int *output, int num_classes)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < 512 * 1024) {
                int max_idx = 0;
                float max_val = input[idx * num_classes];
                
                for (int c = 1; c < num_classes; c++) {
                    float val = input[idx * num_classes + c];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = c;
                    }
                }
                
                output[idx] = max_idx;
            }
        }
        """)
        
        # CUDA kernel for visualization
        self.visualize_kernel = SourceModule("""
        __global__ void visualize(int *input, float *colors, float *output)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < 512 * 1024) {
                int class_idx = input[idx];
                int offset = class_idx * 3;
                
                output[idx * 3] = colors[offset];
                output[idx * 3 + 1] = colors[offset + 1];
                output[idx * 3 + 2] = colors[offset + 2];
            }
        }
        """)
        
        # Get kernel functions
        self.preprocess = self.preprocess_kernel.get_function("preprocess")
        self.postprocess = self.postprocess_kernel.get_function("postprocess")
        self.visualize = self.visualize_kernel.get_function("visualize")
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
    def preprocess_image(self, image, mean, std):
        """Preprocess image using CUDA"""
        # Convert image to float32
        image = image.astype(np.float32)
        
        # Allocate device memory
        image_gpu = cuda.mem_alloc(image.nbytes)
        output_gpu = cuda.mem_alloc(image.nbytes)
        
        # Copy to device
        cuda.memcpy_htod_async(image_gpu, image, self.stream)
        
        # Launch kernel
        block_size = (1024, 1, 1)
        grid_size = ((3 * 512 * 1024 + block_size[0] - 1) // block_size[0], 1, 1)
        
        self.preprocess(
            image_gpu, output_gpu,
            cuda.In(mean), cuda.In(std),
            block=block_size,
            grid=grid_size,
            stream=self.stream
        )
        
        # Copy result back
        output = np.empty_like(image)
        cuda.memcpy_dtoh_async(output, output_gpu, self.stream)
        self.stream.synchronize()
        
        # Free memory
        image_gpu.free()
        output_gpu.free()
        
        return output
        
    def postprocess_prediction(self, prediction, num_classes):
        """Postprocess prediction using CUDA"""
        # Allocate device memory
        prediction_gpu = cuda.mem_alloc(prediction.nbytes)
        output_gpu = cuda.mem_alloc(prediction.nbytes // 4)  # int32 is 4 bytes
        
        # Copy to device
        cuda.memcpy_htod_async(prediction_gpu, prediction, self.stream)
        
        # Launch kernel
        block_size = (1024, 1, 1)
        grid_size = ((512 * 1024 + block_size[0] - 1) // block_size[0], 1, 1)
        
        self.postprocess(
            prediction_gpu, output_gpu,
            np.int32(num_classes),
            block=block_size,
            grid=grid_size,
            stream=self.stream
        )
        
        # Copy result back
        output = np.empty((512 * 1024,), dtype=np.int32)
        cuda.memcpy_dtoh_async(output, output_gpu, self.stream)
        self.stream.synchronize()
        
        # Free memory
        prediction_gpu.free()
        output_gpu.free()
        
        return output.reshape((512, 1024))
        
    def visualize_prediction(self, prediction, colors):
        """Visualize prediction using CUDA"""
        # Allocate device memory
        prediction_gpu = cuda.mem_alloc(prediction.nbytes)
        colors_gpu = cuda.mem_alloc(colors.nbytes)
        output_gpu = cuda.mem_alloc(3 * prediction.nbytes)
        
        # Copy to device
        cuda.memcpy_htod_async(prediction_gpu, prediction, self.stream)
        cuda.memcpy_htod_async(colors_gpu, colors, self.stream)
        
        # Launch kernel
        block_size = (1024, 1, 1)
        grid_size = ((512 * 1024 + block_size[0] - 1) // block_size[0], 1, 1)
        
        self.visualize(
            prediction_gpu, colors_gpu, output_gpu,
            block=block_size,
            grid=grid_size,
            stream=self.stream
        )
        
        # Copy result back
        output = np.empty((512 * 1024 * 3,), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, output_gpu, self.stream)
        self.stream.synchronize()
        
        # Free memory
        prediction_gpu.free()
        colors_gpu.free()
        output_gpu.free()
        
        return output.reshape((512, 1024, 3))
        
    def __del__(self):
        """Clean up CUDA resources"""
        if hasattr(self, 'stream'):
            self.stream.synchronize()
            del self.stream
