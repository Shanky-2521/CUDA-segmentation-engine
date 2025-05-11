import torch
import torch.nn as nn
import torch.onnx
import tensorrt as trt
import os

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 19, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

def create_test_engine():
    # Create and export ONNX model
    model = SimpleModel()
    dummy_input = torch.randn(1, 3, 512, 1024)
    
    # Export to ONNX
    onnx_path = 'models/test_model.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Create TensorRT engine
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.INFO))
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    
    # Set optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, 3, 512, 1024), (1, 3, 512, 1024), (1, 3, 512, 1024))
    builder_config = builder.create_builder_config()
    builder_config.add_optimization_profile(profile)
    
    # Build engine
    engine = builder.build_engine(network, builder_config)
    
    # Save engine
    engine_path = 'models/trt_engines/test_engine.engine'
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    # Clean up
    os.remove(onnx_path)
    
    print(f"Created test engine at {engine_path}")

if __name__ == '__main__':
    create_test_engine()
