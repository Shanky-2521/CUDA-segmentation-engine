import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import time
from pathlib import Path

class CUDAInferenceEngine:
    def __init__(self):
        """Initialize CUDA inference engine"""
        # Create CUDA kernels
        self._create_kernels()
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        # Image dimensions
        self.img_height = 512
        self.img_width = 1024
        
    def _create_kernels(self):
        """Create CUDA kernels for preprocessing, inference, and postprocessing"""
        # Preprocessing kernel
        self.preprocess_kernel = SourceModule("""
        __global__ void preprocess(float *input, float *output, float *mean, float *std)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < 3 * 512 * 1024) {
                int c = idx % 3;
                int pixel_idx = idx / 3;
                int x = pixel_idx % 1024;
                int y = pixel_idx / 1024;
                
                float val = input[idx];
                float normalized = (val / 255.0f - mean[c]) / std[c];
                output[idx] = normalized;
            }
        }
        """)
        
        # Inference kernel (simple identity for testing)
        self.inference_kernel = SourceModule("""
        __global__ void inference(float *input, float *output)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < 19 * 512 * 1024) {
                output[idx] = input[idx];
            }
        }
        """)
        
        # Postprocessing kernel
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
        
        # Visualization kernel
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
        self.inference = self.inference_kernel.get_function("inference")
        self.postprocess = self.postprocess_kernel.get_function("postprocess")
        self.visualize = self.visualize_kernel.get_function("visualize")
        
    def preprocess(self, image):
        """Preprocess image using CUDA"""
        # Convert to float32
        image = image.astype(np.float32)
        
        # Allocate device memory
        d_input = cuda.mem_alloc(image.nbytes)
        d_output = cuda.mem_alloc(image.nbytes)
        
        # Copy to device
        cuda.memcpy_htod(d_input, image)
        
        # Launch kernel
        block_size = (1024, 1, 1)
        grid_size = ((3 * self.img_height * self.img_width + block_size[0] - 1) // block_size[0], 1, 1)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        self.preprocess(
            d_input, d_output,
            cuda.In(mean), cuda.In(std),
            block=block_size,
            grid=grid_size,
            stream=self.stream
        )
        
        # Copy result back
        output = np.empty_like(image)
        cuda.memcpy_dtoh(output, d_output)
        
        # Free memory
        d_input.free()
        d_output.free()
        
        return output
        
    def infer(self, image):
        """Run inference using CUDA"""
        # Preprocess image
        processed = self.preprocess(image)
        
        # Allocate device memory
        d_input = cuda.mem_alloc(processed.nbytes)
        d_output = cuda.mem_alloc(processed.nbytes)
        
        # Copy to device
        cuda.memcpy_htod(d_input, processed)
        
        # Launch inference kernel
        block_size = (1024, 1, 1)
        grid_size = ((19 * self.img_height * self.img_width + block_size[0] - 1) // block_size[0], 1, 1)
        
        self.inference(
            d_input, d_output,
            block=block_size,
            grid=grid_size,
            stream=self.stream
        )
        
        # Copy result back
        output = np.empty_like(processed)
        cuda.memcpy_dtoh(output, d_output)
        
        # Free memory
        d_input.free()
        d_output.free()
        
        return output
        
    def postprocess(self, prediction):
        """Postprocess prediction using CUDA"""
        # Allocate device memory
        d_input = cuda.mem_alloc(prediction.nbytes)
        d_output = cuda.mem_alloc(prediction.nbytes // 4)  # int32 is 4 bytes
        
        # Copy to device
        cuda.memcpy_htod(d_input, prediction)
        
        # Launch kernel
        block_size = (1024, 1, 1)
        grid_size = ((self.img_height * self.img_width + block_size[0] - 1) // block_size[0], 1, 1)
        
        self.postprocess(
            d_input, d_output,
            np.int32(19),
            block=block_size,
            grid=grid_size,
            stream=self.stream
        )
        
        # Copy result back
        output = np.empty((self.img_height * self.img_width,), dtype=np.int32)
        cuda.memcpy_dtoh(output, d_output)
        
        # Free memory
        d_input.free()
        d_output.free()
        
        return output.reshape((self.img_height, self.img_width))
        
    def visualize(self, image, prediction):
        """Visualize prediction using CUDA"""
        # Create color map for classes
        colors = np.random.randint(0, 255, (19, 3), dtype=np.float32)
        
        # Allocate device memory
        d_input = cuda.mem_alloc(prediction.nbytes)
        d_colors = cuda.mem_alloc(colors.nbytes)
        d_output = cuda.mem_alloc(3 * prediction.nbytes)
        
        # Copy to device
        cuda.memcpy_htod(d_input, prediction)
        cuda.memcpy_htod(d_colors, colors)
        
        # Launch kernel
        block_size = (1024, 1, 1)
        grid_size = ((self.img_height * self.img_width + block_size[0] - 1) // block_size[0], 1, 1)
        
        self.visualize(
            d_input, d_colors, d_output,
            block=block_size,
            grid=grid_size,
            stream=self.stream
        )
        
        # Copy result back
        output = np.empty((self.img_height * self.img_width * 3,), dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        
        # Free memory
        d_input.free()
        d_colors.free()
        d_output.free()
        
        # Convert to uint8 and reshape
        output = (output * 255).astype(np.uint8)
        return output.reshape((self.img_height, self.img_width, 3))
        
    def benchmark(self, image, num_runs=100):
        """Benchmark inference performance"""
        import time
        
        # Warm up
        for _ in range(5):
            self.infer(image)
        
        # Measure performance
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            self.infer(image)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_latency = np.mean(latencies)
        fps = 1000 / avg_latency
        
        return avg_latency, fps
        
    def __del__(self):
        """Clean up CUDA resources"""
        if hasattr(self, 'stream'):
            self.stream.synchronize()
            del self.stream
