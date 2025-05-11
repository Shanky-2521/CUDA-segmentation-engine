import torch
import onnx
import onnxruntime
from src.config import Config
from src.model.deeplabv3 import DeepLabV3

class ModelExporter:
    def __init__(self, config):
        """Initialize model exporter with configuration"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, model_path):
        """Load trained model from checkpoint"""
        model = DeepLabV3(num_classes=self.config.NUM_CLASSES)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(self.device)
        return model
    
    def export_to_onnx(self, model, output_path):
        """Export PyTorch model to ONNX format"""
        # Create dummy input matching the model's input size
        dummy_input = torch.randn(1, 3, 512, 1024, device=self.device)
        
        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},  # variable batch size
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model successfully exported to ONNX format at {output_path}")
    
    def verify_onnx_model(self, onnx_path):
        """Verify the exported ONNX model"""
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification passed")
            
            # Test inference with ONNX Runtime
            ort_session = onnxruntime.InferenceSession(onnx_path)
            dummy_input = np.random.randn(1, 3, 512, 1024).astype(np.float32)
            
            # Get model input and output names
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            # Run inference
            ort_inputs = {input_name: dummy_input}
            ort_outs = ort_session.run(None, ort_inputs)
            print("ONNX Runtime inference test successful")
            
            return True
        except Exception as e:
            print(f"ONNX model verification failed: {str(e)}")
            return False

def export_model(config, model_path):
    """Main function to export model to ONNX"""
    exporter = ModelExporter(config)
    model = exporter.load_model(model_path)
    
    # Create output directory if it doesn't exist
    output_dir = Path(config.MODELS_DIR) / 'onnx'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    onnx_path = str(output_dir / 'deeplabv3.onnx')
    
    # Export model
    exporter.export_to_onnx(model, onnx_path)
    
    # Verify export
    if exporter.verify_onnx_model(onnx_path):
        print("Model export and verification completed successfully")
    else:
        print("There was an issue with the ONNX export. Please check the error messages above.")

if __name__ == '__main__':
    # Example usage
    config = Config()
    export_model(config, 'path/to/trained/model.pth')
