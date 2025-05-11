import tensorrt as trt
import numpy as np
import cv2

class SimpleInferenceEngine:
    def __init__(self, engine_path):
        """Initialize TensorRT inference engine"""
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            serialized_engine = f.read()
            
        # Create engine from serialized data
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")
            
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        if self.context is None:
            raise RuntimeError("Failed to create execution context")
            
        # Get binding indices
        self.input_binding = 0  # Input is always first binding
        self.output_binding = 1  # Output is always second binding
        
        # Get tensor shapes
        input_name = self.engine.get_tensor_name(self.input_binding)
        output_name = self.engine.get_tensor_name(self.output_binding)
        
        input_shape = tuple(self.engine.get_tensor_shape(input_name))
        output_shape = tuple(self.engine.get_tensor_shape(output_name))
        
        # Create buffers
        self.input_buffer = np.zeros(input_shape, dtype=np.float32)
        self.output_buffer = np.zeros(output_shape, dtype=np.float32)
        
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")
        
    def infer(self, image):
        """Run inference on an image"""
        # Preprocess image
        image = cv2.resize(image, (224, 224))  # ResNet18 expects 224x224 input
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        # Copy input to buffer
        self.input_buffer = image.copy()  # Ensure we have a contiguous array
        
        # Execute inference
        bindings = {
            0: self.input_buffer.ctypes.data,  # Input is binding 0
            1: self.output_buffer.ctypes.data  # Output is binding 1
        }
        
        self.context.execute_async_v3(
            bindings=list(bindings.values()),
            stream_handle=0
        )
        
        # Postprocess output
        output = np.squeeze(self.output_buffer)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255).astype(np.uint8)
        
        return output
        # Preprocess image
        image = cv2.resize(image, (1024, 512))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        # Copy input to buffer
        np.copyto(self.input_buffer, image)
        
        # Execute inference
        self.context.execute_async_v2(
            bindings=[self.input_buffer.ctypes.data, self.output_buffer.ctypes.data],
            stream_handle=0
        )
        
        # Postprocess output
        output = np.squeeze(self.output_buffer)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255).astype(np.uint8)
        
        return output
    
    def benchmark(self, image, num_runs=100):
        """Benchmark inference performance"""
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
        try:
            if hasattr(self, 'stream'):
                self.stream.synchronize()
                del self.stream
            if hasattr(self, 'input_buffer'):
                self.input_buffer.free()
            if hasattr(self, 'output_buffer'):
                self.output_buffer.free()
            if hasattr(self, 'ctx'):
                self.ctx.pop()
        except:
            pass
