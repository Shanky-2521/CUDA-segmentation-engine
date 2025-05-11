import tensorrt as trt
import os

def create_tensorrt_engine(onnx_path, engine_path):
    """Create and save TensorRT engine from ONNX model"""
    # Create builder and network
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create parser
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Create builder configuration
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Enable FP16
    if builder.platform_has_fast_fp16:
        builder_config.set_flag(trt.BuilderFlag.FP16)
    
    # Set optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape('input', 
                     min=(1, 3, 512, 1024),
                     opt=(1, 3, 512, 1024),
                     max=(1, 3, 512, 1024))
    builder_config.add_optimization_profile(profile)
    
    # Build engine
    print("\nBuilding TensorRT engine...")
    
    # Create builder configuration
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Enable FP16
    if builder.platform_has_fast_fp16:
        builder_config.set_flag(trt.BuilderFlag.FP16)
    
    # Set optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape('input', 
                     min=(1, 3, 512, 1024),
                     opt=(1, 3, 512, 1024),
                     max=(1, 3, 512, 1024))
    builder_config.add_optimization_profile(profile)
    
    # Build engine
    engine = builder.build_serialized_network(network, builder_config)
    
    if engine is None:
        raise RuntimeError("Failed to build engine")
    
    # Save engine
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    print(f"Created TensorRT engine at {engine_path}")
    print("Engine size:", os.path.getsize(engine_path) / (1024 * 1024), "MB")
    return engine

if __name__ == '__main__':
    onnx_path = 'models/deeplabv3.onnx'
    engine_path = 'models/trt_engines/deeplabv3_fp16.engine'
    create_tensorrt_engine(onnx_path, engine_path)
