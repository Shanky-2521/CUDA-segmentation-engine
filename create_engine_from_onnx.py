import tensorrt as trt
import numpy as np
import os

def create_engine_from_onnx(onnx_model_path, engine_path):
    """Create a TensorRT engine from an ONNX model"""
    # Create builder and network
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create parser
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.INFO))
    
    # Parse ONNX model
    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Create builder configuration
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Enable FP16 if supported
    if builder.platform_has_fast_fp16:
        builder_config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    print("Building engine...")
    engine = builder.build_serialized_network(network, builder_config)
    
    if engine is None:
        raise RuntimeError("Failed to build engine")
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    print(f"Created TensorRT engine at {engine_path}")
    print("Engine size:", os.path.getsize(engine_path) / (1024 * 1024), "MB")

if __name__ == '__main__':
    # Create engine from ONNX model
    create_engine_from_onnx(
        onnx_model_path='models/onnx/deeplabv3.onnx',
        engine_path='models/trt_engines/deeplabv3_fp16.engine'
    )
