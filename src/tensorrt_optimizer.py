import tensorrt as trt
import torch
import numpy as np
from pathlib import Path
from src.config import Config

class TensorRTOptimizer:
    def __init__(self, config):
        """Initialize TensorRT optimizer"""
        self.config = config
        self.logger = trt.Logger(trt.Logger.INFO)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.parser = trt.OnnxParser(self.network, self.logger)
        
        # Set builder configuration
        self.builder_config = self.builder.create_builder_config()
        self.builder_config.max_workspace_size = 1 << 30  # 1GB
        
    def load_onnx_model(self, onnx_path):
        """Load ONNX model"""
        with open(onnx_path, 'rb') as model_file:
            if not self.parser.parse(model_file.read()):
                print('Failed to parse ONNX file')
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                return False
        return True
    
    def optimize(self, precision='fp16'):
        """Optimize the model with TensorRT"""
        if precision == 'fp16':
            if not self.builder.platform_has_fast_fp16:
                print("FP16 is not supported on this platform")
                return None
            self.builder_config.set_flag(trt.BuilderFlag.FP16)
        
        # Build the engine
        engine = self.builder.build_engine(self.network, self.builder_config)
        if engine is None:
            print("Failed to build TensorRT engine")
            return None
        
        return engine
    
    def save_engine(self, engine, output_path):
        """Save TensorRT engine to file"""
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"TensorRT engine saved to {output_path}")
    
    def optimize_model(self, onnx_path, precision='fp16'):
        """Complete optimization pipeline"""
        # Load ONNX model
        if not self.load_onnx_model(onnx_path):
            return False
        
        # Optimize with TensorRT
        engine = self.optimize(precision)
        if engine is None:
            return False
        
        # Save optimized engine
        output_dir = Path(self.config.MODELS_DIR) / 'tensorrt'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = str(output_dir / f'deeplabv3_{precision}.engine')
        self.save_engine(engine, output_path)
        
        return output_path

def optimize_model(config, onnx_path, precision='fp16'):
    """Optimize model using TensorRT"""
    optimizer = TensorRTOptimizer(config)
    engine_path = optimizer.optimize_model(onnx_path, precision)
    
    if engine_path:
        print(f"Model optimized successfully. TensorRT engine saved to: {engine_path}")
        return engine_path
    else:
        print("Model optimization failed")
        return None

if __name__ == '__main__':
    # Example usage
    config = Config()
    onnx_path = 'path/to/onnx/model.onnx'
    optimize_model(config, onnx_path)
