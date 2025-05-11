import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import time
from pathlib import Path
from src.cuda.cuda_operations import CUDAOperations
from src.cuda.cuda_kernels import CUDAInferenceKernels

class TensorRTInferenceEngine:
    def __init__(self, engine_path, batch_size=1, image_size=(512, 1024)):
        """
        Initialize TensorRT inference engine with CUDA optimizations
        """
        self.engine_path = Path(engine_path)
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Initialize CUDA components
        self.cuda_ops = CUDAOperations()
        self.cuda_kernels = CUDAInferenceKernels()
        self.stream = self.cuda_ops.stream
        
        # Load TensorRT engine
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()
        
        # CUDA memory allocation
        self.d_input = self.cuda_ops.allocate_device_memory(
            1 * trt.volume(self.inputs[0].shape) * np.float32().itemsize
        )
        self.d_output = self.cuda_ops.allocate_device_memory(
            1 * trt.volume(self.outputs[0].shape) * np.float32().itemsize
        )
        
    def _load_engine(self):
        """
        Load and deserialize TensorRT engine
        """
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
        return self.engine
        
    def _allocate_buffers(self):
        """
        Allocate input and output buffers for inference
        """
        inputs = []
        outputs = []
        bindings = []
        
        if self.engine:
            input_name = self.engine.get_binding_name(0)
            output_name = self.engine.get_binding_name(1)
            input_shape = tuple(self.engine.get_binding_shape(0))
            output_shape = tuple(self.engine.get_binding_shape(1))
            
            # Allocate host buffers
            host_mem_input = np.zeros(input_shape, dtype=np.float32)
            host_mem_output = np.zeros(output_shape, dtype=np.float32)
            
            # Allocate device buffers
            device_mem_input = self.cuda_ops.allocate_device_memory(host_mem_input.nbytes)
            device_mem_output = self.cuda_ops.allocate_device_memory(host_mem_output.nbytes)
            
            # Append to appropriate lists
            inputs.append(host_mem_input)
            outputs.append(host_mem_output)
            bindings.append(int(device_mem_input))
            bindings.append(int(device_mem_output))
        
        return inputs, outputs, bindings

    def preprocess(self, image):
        """Preprocess image using CUDA"""
        # Resize image
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize using CUDA kernel
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Flatten and process with CUDA
        image_flat = image.reshape(-1)
        image_processed = self.cuda_kernels.preprocess_image(image_flat, mean, std)
        
        # Reshape back to original dimensions
        image_processed = image_processed.reshape(image.shape)
        
        # Transpose to CHW
        image = np.transpose(image_processed, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, 0)
        
        return image

    def postprocess(self, output):
        """Postprocess model output using CUDA"""
        # Get number of classes
        num_classes = output.shape[1]
        
        # Flatten output for CUDA processing
        output_flat = output.reshape(-1, num_classes)
        
        # Process with CUDA kernel
        prediction = self.cuda_kernels.postprocess_prediction(output_flat, num_classes)
        
        return prediction

    def infer(self, image):
        """Run inference on a single image"""
        # Preprocess image
        input_tensor = self.preprocess(image)
        
        if self.engine:  
            # Get input and output indices
            input_idx = self.engine.get_binding_index('input')
            output_idx = self.engine.get_binding_index('output')
            
            # Get input and output shapes
            input_shape = tuple(self.engine.get_binding_shape(input_idx))
            output_shape = tuple(self.engine.get_binding_shape(output_idx))
            
            # Allocate buffers
            inputs = [np.zeros(input_shape, dtype=np.float32)]
            outputs = [np.zeros(output_shape, dtype=np.float32)]
            
            # Set input
            inputs[0] = input_tensor
            
            # Transfer input to device
            self.cuda_ops.async_copy_to_device(inputs[0], self.d_input)
            
            # Execute model
            bindings = [int(self.d_input), int(self.d_output)]
            self.context.execute_async_v2(bindings, self.stream.handle)
            
            # Transfer results back to host
            self.cuda_ops.async_copy_to_host(self.d_output, outputs[0])
            self.cuda_ops.synchronize()
            
            # Postprocess output
            prediction = self.postprocess(outputs[0])
            
        return prediction

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
        """Visualize segmentation results using CUDA"""
        # Create color map for classes
        colors = np.random.randint(0, 255, (19, 3), dtype=np.float32)
        
        # Process with CUDA kernel
        colorized = self.cuda_kernels.visualize_prediction(prediction, colors)
        
        # Resize back to original size
        original_size = image.shape[:2]
        colorized = cv2.resize(colorized, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Convert back to uint8
        colorized = (colorized * 255).astype(np.uint8)
        
        # Overlay prediction on original image
        alpha = 0.5
        overlay = cv2.addWeighted(image, alpha, colorized, 1-alpha, 0)
        
        return overlay

    def __del__(self):
        """
        Clean up CUDA resources
        """
        if hasattr(self, 'cuda_ops'):
            del self.cuda_ops
            del self.cuda_kernels
            self.stream.synchronize()
