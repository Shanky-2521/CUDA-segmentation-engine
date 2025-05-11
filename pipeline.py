import os
import sys
import time
from pathlib import Path
from src.config import Config
from src.training.trainer import Trainer
from src.model_export import export_model
from src.tensorrt_optimizer import optimize_model
from src.evaluation.model_evaluator import evaluate_models
from src.visualization.visualization import Visualization

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.trainer = None
        self.visualizer = None
        self.results = {}
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config.MODELS_DIR,
            self.config.RESULTS_DIR,
            os.path.join(self.config.MODELS_DIR, 'onnx'),
            os.path.join(self.config.MODELS_DIR, 'tensorrt'),
            os.path.join(self.config.RESULTS_DIR, 'visualizations')
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def train_model(self):
        """Train the model"""
        print("\nStarting model training...")
        self.trainer = Trainer(self.config)
        self.trainer.train()
        
        # Save trained model
        model_path = os.path.join(self.config.MODELS_DIR, 'deeplabv3.pth')
        torch.save(self.trainer.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
        return model_path
    
    def export_to_onnx(self, model_path):
        """Export trained model to ONNX"""
        print("\nExporting model to ONNX...")
        onnx_path = export_model(self.config, model_path)
        print(f"ONNX model saved to: {onnx_path}")
        return onnx_path
    
    def optimize_with_tensorrt(self, onnx_path):
        """Optimize ONNX model with TensorRT"""
        print("\nOptimizing model with TensorRT...")
        engine_path = optimize_model(self.config, onnx_path)
        print(f"TensorRT engine saved to: {engine_path}")
        return engine_path
    
    def evaluate_models(self):
        """Evaluate all models"""
        print("\nEvaluating models...")
        results = evaluate_models(self.config)
        self.results = results
        return results
    
    def visualize_results(self):
        """Create visualizations"""
        print("\nCreating visualizations...")
        self.visualizer = Visualization(self.config)
        
        # Get test image
        test_dir = Path(self.config.DATA_DIR) / 'test'
        test_images = sorted(list((test_dir / 'img').glob('*.png')))
        
        if test_images:
            # Visualize single prediction
            self.visualizer.visualize_segmentation(
                cv2.imread(str(test_images[0])),
                self.results['deeplabv3_fp16']['prediction'],
                output_path=os.path.join(self.config.RESULTS_DIR, 'visualizations', 'sample_prediction.png')
            )
            
            # Create model comparison
            models = {
                'ONNX': os.path.join(self.config.MODELS_DIR, 'onnx', 'deeplabv3.onnx'),
                'TensorRT': os.path.join(self.config.MODELS_DIR, 'tensorrt', 'deeplabv3_fp16.engine')
            }
            self.visualizer.create_comparison_grid(
                models,
                test_images[0],
                self.config.RESULTS_DIR
            )
            
            # Create performance chart
            self.visualizer.create_performance_chart(
                self.results,
                self.config.RESULTS_DIR
            )
    
    def run(self):
        """Run complete pipeline"""
        print("Starting Semantic Segmentation Optimization Pipeline...")
        
        # Setup directories
        self.setup_directories()
        
        # Run pipeline steps
        model_path = self.train_model()
        onnx_path = self.export_to_onnx(model_path)
        engine_path = self.optimize_with_tensorrt(onnx_path)
        evaluation_results = self.evaluate_models()
        self.visualize_results()
        
        print("\nPipeline completed successfully!")
        print("\nFinal Results:")
        for model_name, metrics in evaluation_results.items():
            print(f"\nModel: {model_name}")
            print(f"Mean IoU: {metrics['mean_iou']:.4f}")
            print(f"FPS: {metrics['fps']:.2f}")
            print(f"Latency: {metrics['latency_ms']:.2f} ms")

def main():
    """Main function"""
    config = Config()
    pipeline = Pipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()
