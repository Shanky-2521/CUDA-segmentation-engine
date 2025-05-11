import numpy as np
import tensorrt as trt
import os

def create_simple_engine():
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
    engine_path = 'models/trt_engines/simple_engine.engine'
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"Created simple engine at {engine_path}")

if __name__ == '__main__':
    create_simple_engine()
