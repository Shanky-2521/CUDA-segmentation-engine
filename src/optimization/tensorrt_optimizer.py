import tensorrt as trt
import numpy as np
import onnx
from pathlib import Path
import torch
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTOptimizer:
    def __init__(self, onnx_path, trt_path, precision='fp16'):
        self.onnx_path = Path(onnx_path)
        self.trt_path = Path(trt_path)
        self.precision = precision
        self.logger = trt.Logger(trt.Logger.INFO)
        
    def build_engine(self, batch_size=1, image_size=(512, 1024)):
        """
        Build and save TensorRT engine with CUDA optimization
        """
        with trt.Builder(self.logger) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, self.logger) as parser:
            
            # Configure builder
            builder.max_batch_size = batch_size
            config = builder.create_builder_config()
            
            # Set workspace size based on available GPU memory
            total_memory = cuda.Device(0).total_memory()
            config.max_workspace_size = min(1 << 32, total_memory // 2)  # Use up to half of GPU memory
            
            # Enable CUDA optimizations
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            
            # Set precision
            if self.precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.FORCE_FP16)
            elif self.precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
                config.set_flag(trt.BuilderFlag.FORCE_INT8)
            
            # Parse ONNX model
            with open(self.onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print('Failed to parse ONNX file')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            
            # Optimize network
            network.get_input(0).shape = [batch_size, 3, *image_size]
            
            # Add CUDA optimizations
            builder.max_workspace_size = 1 << 30  # 1GB
            builder.strict_type_constraints = True
            
            # Build engine
            engine = builder.build_serialized_network(network, config)
            
            if engine is None:
                print("Failed to build TensorRT engine")
                return None
            
            # Save engine
            with open(self.trt_path, 'wb') as f:
                f.write(engine)
            
            # Print engine information
            print(f"Engine built successfully:")
            print(f"- Precision: {self.precision.upper()}")
            print(f"- Batch Size: {batch_size}")
            print(f"- Image Size: {image_size}")
            print(f"- Workspace Size: {config.max_workspace_size / (1<<30):.1f}GB")
            
            return engine
            
    def optimize_engine(self, engine_path):
        """
        Optimize existing TensorRT engine with CUDA optimizations
        """
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(f.read())
            
        # Optimize CUDA context
        context = engine.create_execution_context()
        context.set_optimization_profile_async(0, 0)
        
        # Create CUDA streams
        stream = cuda.Stream()
        
        # Optimize memory allocation
        input_size = trt.volume(engine.get_binding_shape(0))
        output_size = trt.volume(engine.get_binding_shape(1))
        dtype = np.float32
        
        # Allocate device memory
        d_input = cuda.mem_alloc(1 * input_size * dtype.itemsize)
        d_output = cuda.mem_alloc(1 * output_size * dtype.itemsize)
        
        # Create bindings
        bindings = [int(d_input), int(d_output)]
        
        return {
            'engine': engine,
            'context': context,
            'stream': stream,
            'bindings': bindings,
            'd_input': d_input,
            'd_output': d_output
        }

    def load_engine(self):
        """
        Load TensorRT engine
        """
        with open(self.trt_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())
