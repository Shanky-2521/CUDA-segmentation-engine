import tensorrt as trt
import numpy as np
import os

def create_test_engine(output_path):
    """Create a simple test TensorRT engine"""
    # Create a builder and network
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create input tensor
    input_shape = (1, 3, 512, 1024)
    input_tensor = network.add_input(name='input', dtype=trt.float32, shape=input_shape)
    
    # Add a simple identity layer
    identity = network.add_identity(input_tensor)
    identity.get_output(0).name = 'output'
    network.mark_output(identity.get_output(0))
    
    # Set optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, 3, 512, 1024), (1, 3, 512, 1024), (1, 3, 512, 1024))
    builder_config = builder.create_builder_config()
    builder_config.add_optimization_profile(profile)
    
    # Build engine
    engine = builder.build_engine(network, builder_config)
    
    # Save engine
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine

if __name__ == '__main__':
    engine_path = os.path.join('models', 'trt_engines', 'test_engine.engine')
    create_test_engine(engine_path)
    print(f"Created test engine at {engine_path}")
