import numpy as np
import tensorrt as trt
import os

def create_minimal_engine():
    # Create a builder and network
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create input tensor
    input_shape = (1, 3, 512, 1024)
    input_tensor = network.add_input(name='input', dtype=trt.float32, shape=input_shape)
    
    # Add a simple layer that just passes through the input
    layer = network.add_identity(input_tensor)
    layer.get_output(0).name = 'output'
    network.mark_output(layer.get_output(0))
    
    # Set optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, 3, 512, 1024), (1, 3, 512, 1024), (1, 3, 512, 1024))
    
    # Create builder configuration
    builder_config = builder.create_builder_config()
    builder_config.max_workspace_size = 1 << 30  # 1GB
    builder_config.add_optimization_profile(profile)
    
    # Build engine
    print("Building engine...")
    engine = builder.build_engine(network, builder_config)
    
    if engine is None:
        raise RuntimeError("Failed to build engine")
    
    # Save engine
    engine_path = 'models/trt_engines/minimal_engine.engine'
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"Created minimal engine at {engine_path}")
    print("Engine size:", os.path.getsize(engine_path) / (1024 * 1024), "MB")

if __name__ == '__main__':
    create_minimal_engine()
