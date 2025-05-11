import tensorrt as trt
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_size=1, max_calib_size=500):
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        # Create a dataset of random images for calibration
        self.batch_size = batch_size
        self.max_calib_size = max_calib_size
        self.current_idx = 0
        
        # Create calibration data
        self.calib_data = []
        for _ in range(max_calib_size):
            img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
            self.calib_data.append(img)
        
        # Create cache file
        self.cache_file = "calibration_cache"
        
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_idx + self.batch_size > self.max_calib_size:
            return None
            
        batch = []
        for i in range(self.batch_size):
            img = self.calib_data[self.current_idx + i]
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            batch.append(img)
            
        batch = np.stack(batch)
        self.current_idx += self.batch_size
        return [np.ascontiguousarray(batch)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def create_deeplabv3_engine(batch_size=4, workspace_size=1 << 30, 
                  use_fp16=True, use_int8=True, 
                  calib_size=500, onnx_path='models/deeplabv3.onnx',
                  engine_path='models/trt_engines/deeplabv3_int8.engine'):
    """
    Create a TensorRT engine for DeepLabV3
    
    Args:
        batch_size: Batch size for inference
        workspace_size: Workspace size in bytes
        use_fp16: Enable FP16 precision
        use_int8: Enable INT8 precision
        calib_size: Number of calibration images for INT8
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
    
    Returns:
        Path to created engine
    """
    try:
        # Initialize CUDA
        import pycuda.driver as cuda
        cuda.init()
        device = cuda.Device(0)
        ctx = device.make_context()
        
        try:
            # Create builder and network
            builder = trt.Builder(trt.Logger(trt.Logger.INFO))
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Create parser
            parser = trt.OnnxParser(network, trt.Logger(trt.Logger.INFO))
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    errors = [parser.get_error(i) for i in range(parser.num_errors)]
                    raise RuntimeError(f"Failed to parse ONNX file. Errors: {errors}")
            
            # Create builder configuration
            builder_config = builder.create_builder_config()
            builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
            
            # Enable precision flags
            if use_fp16 and builder.platform_has_fast_fp16:
                builder_config.set_flag(trt.BuilderFlag.FP16)
                print("FP16 enabled")
            
            if use_int8 and builder.platform_has_fast_int8:
                builder_config.set_flag(trt.BuilderFlag.INT8)
                print("INT8 enabled")
                
                # Create calibration data
                calibrator = Int8Calibrator(batch_size=batch_size, max_calib_size=calib_size)
                builder_config.int8_calibrator = calibrator
                print(f"Using calibration dataset of size {calib_size}")
            
            # Set optimization profiles
            min_shape = (1, 3, 512, 1024)
            opt_shape = (batch_size, 3, 512, 1024)
            max_shape = (batch_size, 3, 512, 1024)
            
            profile = builder.create_optimization_profile()
            profile.set_shape('input', min_shape, opt_shape, max_shape)
            builder_config.add_optimization_profile(profile)
            
            # Build engine
            print(f"\nBuilding engine with batch size {batch_size}...")
            engine = builder.build_serialized_network(network, builder_config)
            
            if engine is None:
                raise RuntimeError("Failed to build engine")
            
            # Save engine
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
            with open(engine_path, 'wb') as f:
                f.write(engine)
            
            engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
            print(f"Created TensorRT engine at {engine_path}")
            print(f"Engine size: {engine_size_mb:.2f} MB")
            print(f"Engine configuration:")
            print(f"  Batch size: {batch_size}")
            print(f"  Workspace size: {workspace_size / (1024 * 1024):.2f} MB")
            print(f"  FP16 enabled: {use_fp16}")
            print(f"  INT8 enabled: {use_int8}")
            
            return engine_path
            
        finally:
            # Clean up CUDA context
            ctx.pop()
            del ctx
            
    except Exception as e:
        print(f"Error creating engine: {str(e)}")
        raise

if __name__ == '__main__':
    create_deeplabv3_engine()
