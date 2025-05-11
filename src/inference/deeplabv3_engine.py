import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import time
from pathlib import Path
from ..utils.logger import Logger

class DeepLabV3EngineError(Exception):
    """Base class for exceptions in this module."""
    pass

class EngineLoadError(DeepLabV3EngineError):
    """Exception raised for errors in engine loading."""
    pass

class InferenceError(DeepLabV3EngineError):
    """Exception raised for errors during inference."""
    pass

class DeepLabV3Engine:
    def __init__(self, engine_path):
        """Initialize TensorRT inference engine"""
        self.logger = Logger('deeplabv3_engine').get_logger()
        self.logger.info("Initializing DeepLabV3 Engine...")
        
        # Get CUDA device
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        
        try:
            # Load engine
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
                engine_data = f.read()
                
            # Create engine from serialized data
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("Failed to deserialize engine")
                
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            if self.context is None:
                raise RuntimeError("Failed to create execution context")
                
            # Get input and output shapes
            self.input_shape = self.engine.get_tensor_shape("input")
            self.output_shape = self.engine.get_tensor_shape("output")
            
            # Create buffers
            self.input_buffer = np.zeros(self.input_shape, dtype=np.float32)
            self.output_buffer = np.zeros(self.output_shape, dtype=np.float32)
            
            # Create CUDA stream
            self.stream = cuda.Stream()
            
            # Allocate device memory
            self.input_d = cuda.mem_alloc(self.input_buffer.nbytes)
            self.output_d = cuda.mem_alloc(self.output_buffer.nbytes)
            
        except Exception as e:
            # Clean up CUDA resources
            if hasattr(self, 'input_d'):
                self.input_d.free()
            if hasattr(self, 'output_d'):
                self.output_d.free()
            if hasattr(self, 'ctx'):
                self.ctx.pop()
            raise e
            
    def __del__(self):
        """Clean up CUDA context and memory"""
        if hasattr(self, 'input_d'):
            self.input_d.free()
        if hasattr(self, 'output_d'):
            self.output_d.free()
        if hasattr(self, 'ctx'):
            self.ctx.pop()
            del self.ctx
        
    def preprocess(self, image):
        """Preprocess input image for inference"""
        # Resize to model input size
        image = cv2.resize(image, (1024, 512))
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
        
    def postprocess(self, output):
        """Postprocess model output"""
        # Get class with highest probability
        prediction = np.argmax(output, axis=1)
        
        # Convert to color-coded segmentation map
        # Using Cityscapes color palette
        color_palette = [
            [128, 64, 128],    # road
            [244, 35, 232],    # sidewalk
            [70, 70, 70],      # building
            [102, 102, 156],   # wall
            [190, 153, 153],   # fence
            [153, 153, 153],   # pole
            [250, 170, 30],    # traffic light
            [220, 220, 0],     # traffic sign
            [107, 142, 35],    # vegetation
            [152, 251, 152],   # terrain
            [70, 130, 180],    # sky
            [220, 20, 60],     # person
            [255, 0, 0],       # rider
            [0, 0, 142],       # car
            [0, 0, 70],        # truck
            [0, 60, 100],      # bus
            [0, 80, 100],      # train
            [0, 0, 230],       # motorcycle
            [119, 11, 32],     # bicycle
            [0, 0, 0]          # void
        ]
        
        # Create color-coded segmentation map
        colored_prediction = np.zeros((output.shape[2], output.shape[3], 3), dtype=np.uint8)
        for i in range(len(color_palette)):
            colored_prediction[prediction[0] == i] = color_palette[i]
            
        return colored_prediction
        
    def infer(self, image):
        """Run inference on an image"""
        try:
            # Push CUDA context
            self.ctx.push()
            
            # Preprocess image
            image = self.preprocess(image)
            
            # Copy input to buffer
            np.copyto(self.input_buffer, image)
            
            # Copy input data to device
            cuda.memcpy_htod(self.input_d, self.input_buffer)
            
            # Execute inference
            bindings = [int(self.input_d), int(self.output_d)]
            
            self.context.execute_v2(bindings)
            
            # Copy output data to host
            cuda.memcpy_dtoh(self.output_buffer, self.output_d)
            
            # Synchronize stream
            self.stream.synchronize()
            
            # Postprocess output
            output = self.postprocess(self.output_buffer)
            
            return output
            
        finally:
            # Pop CUDA context
            self.ctx.pop()
        
    def benchmark(self, image, num_iterations=100):
        """Benchmark inference performance"""
        try:
            # Push CUDA context
            self.ctx.push()
            
            # Preprocess image
            image = self.preprocess(image)
            
            # Copy input to buffer
            np.copyto(self.input_buffer, image)
            
            # Use pre-allocated device memory
            input_d = self.input_d
            output_d = self.output_d
            
            # Warm up
            for _ in range(10):
                # Copy input data to device
                cuda.memcpy_htod(input_d, self.input_buffer)
                
                # Execute inference
                bindings = [int(input_d), int(output_d)]
                self.context.execute_v2(bindings)
                
                # Copy output data to host
                cuda.memcpy_dtoh(self.output_buffer, output_d)
                
                # Synchronize stream
                self.stream.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                # Copy input data to device
                cuda.memcpy_htod(input_d, self.input_buffer)
                
                # Execute inference
                bindings = [int(input_d), int(output_d)]
                self.context.execute_v2(bindings)
                
                # Copy output data to host
                cuda.memcpy_dtoh(self.output_buffer, output_d)
                
                # Synchronize stream
                self.stream.synchronize()
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / num_iterations
            fps = num_iterations / total_time
            
            return avg_time, fps
            
        finally:
            # Pop CUDA context
            self.ctx.pop()

if __name__ == '__main__':
    # Test the engine
    engine_path = 'models/trt_engines/deeplabv3_fp16.engine'
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found at {engine_path}")
        exit(1)
        
    # Create engine
    engine = DeepLabV3Engine(engine_path)
    
    # Test with a random image
    test_image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    
    # Run inference
    print("\nRunning inference...")
    output = engine.infer(test_image)
    
    # Show results
    cv2.imshow('Segmentation Result', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = engine.benchmark(test_image)
    print(f"\nBenchmark Results:")
    print(f"Inference time: {results['inference_time_ms']:.2f} ms")
    print(f"FPS: {results['fps']:.2f}")
    print(f"Input shape: {results['input_shape']}")
    print(f"Output shape: {results['output_shape']}")
