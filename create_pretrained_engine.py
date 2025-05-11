import torch
import torch.nn as nn
import torch.onnx
import tensorrt as trt
import os
from torchvision import models

def create_pretrained_engine():
    # Load pre-trained model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Modify output layer to match our needs
    model.fc = nn.Linear(model.fc.in_features, 19)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 512, 1024)
    
    # Export to ONNX
    onnx_path = 'models/pretrained_model.onnx'
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
    print("Building engine...")
    engine = builder.build_serialized_network(network, builder_config)
    
    if engine is None:
        raise RuntimeError("Failed to build engine")
    
    # Save engine
    engine_path = 'models/trt_engines/pretrained_engine.engine'
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    print(f"Created TensorRT engine at {engine_path}")
    print("Engine size:", os.path.getsize(engine_path) / (1024 * 1024), "MB")
    
    # Clean up
    os.remove(onnx_path)
    
    print(f"Created pretrained engine at {engine_path}")

if __name__ == '__main__':
    create_pretrained_engine()
