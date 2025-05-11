import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import time
from pathlib import Path

class SimpleInferenceEngine:
    def __init__(self, engine_path):
        """Initialize TensorRT inference engine"""
        self.engine_path = Path(engine_path)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.input_shape = (1, 3, 512, 1024)
        self.output_shape = (1, 19, 512, 1024)
        
        # Allocate device memory
        self.d_input = cuda.mem_alloc(np.zeros(self.input_shape, dtype=np.float32).nbytes)
        self.d_output = cuda.mem_alloc(np.zeros(self.output_shape, dtype=np.float32).nbytes)
        
    def infer(self, image):
        """Run inference on a single image"""
        # Preprocess image
        image = self.preprocess(image)
        
        # Transfer input to device
        cuda.memcpy_htod(self.d_input, image)
        
        # Execute model
        bindings = [int(self.d_input), int(self.d_output)]
        self.context.execute_v2(bindings)
        
        # Transfer results back to host
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, self.d_output)
        
        # Postprocess output
        prediction = self.postprocess(output)
        
        return prediction

    def preprocess(self, image):
        """Preprocess image"""
        # Resize
        image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Transpose to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, 0)
        
        return image

    def postprocess(self, output):
        """Postprocess model output"""
        # Get predicted class
        pred = np.argmax(output, axis=1)
        return pred

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

    def visualize(self, image, prediction):
        """Visualize segmentation results"""
        # Create color map for classes
        colors = np.random.randint(0, 255, (19, 3), dtype=np.uint8)
        
        # Create colorized prediction
        colorized = colors[prediction]
        
        # Resize back to original size
        original_size = image.shape[:2]
        colorized = cv2.resize(colorized, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Overlay prediction on original image
        alpha = 0.5
        overlay = cv2.addWeighted(image, alpha, colorized, 1-alpha, 0)
        
        return overlay

    def __del__(self):
        """Clean up CUDA resources"""
        if hasattr(self, 'd_input'):
            self.d_input.free()
        if hasattr(self, 'd_output'):
            self.d_output.free()
