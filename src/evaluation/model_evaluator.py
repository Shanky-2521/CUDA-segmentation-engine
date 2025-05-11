import numpy as np
import cv2
import os
from pathlib import Path
from src.config import Config
from src.inference.inference_engine import TensorRTInferenceEngine
from sklearn.metrics import jaccard_score

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.engine = None
        self.results = {}
        
    def load_model(self, model_path):
        """Load model for evaluation"""
        self.engine = TensorRTInferenceEngine(self.config)
        self.engine.load_model(model_path)
        
    def evaluate_accuracy(self, image_paths, mask_paths):
        """Evaluate model accuracy on test set"""
        iou_scores = []
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            # Load image and mask
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Run inference
            prediction = self.engine.infer(image)
            
            # Flatten predictions and masks for IoU calculation
            pred_flat = prediction.flatten()
            mask_flat = mask.flatten()
            
            # Calculate IoU
            iou = jaccard_score(mask_flat, pred_flat, average='macro')
            iou_scores.append(iou)
        
        return np.mean(iou_scores), np.std(iou_scores)
    
    def benchmark_performance(self, image_path, num_runs=100):
        """Benchmark model performance"""
        # Load test image
        image = cv2.imread(str(image_path))
        
        # Get benchmark results
        latency, fps = self.engine.benchmark(image, num_runs)
        
        return latency, fps
    
    def evaluate_model(self, model_path, test_images, test_masks, num_benchmark_runs=100):
        """Complete evaluation pipeline"""
        # Load model
        self.load_model(model_path)
        
        # Get model name from path
        model_name = Path(model_path).name
        
        # Evaluate accuracy
        mean_iou, std_iou = self.evaluate_accuracy(test_images, test_masks)
        
        # Benchmark performance
        latency, fps = self.benchmark_performance(test_images[0], num_benchmark_runs)
        
        # Store results
        self.results[model_name] = {
            'mean_iou': mean_iou,
            'std_iou': std_iou,
            'latency_ms': latency,
            'fps': fps
        }
        
        return self.results[model_name]
    
    def compare_models(self, model_paths, test_images, test_masks):
        """Compare multiple models"""
        results = {}
        
        for model_path in model_paths:
            print(f"Evaluating {Path(model_path).name}...")
            results[Path(model_path).name] = self.evaluate_model(
                model_path,
                test_images,
                test_masks
            )
        
        return results
    
    def save_results(self, results, output_path):
        """Save evaluation results to file"""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

def evaluate_models(config):
    """Main evaluation function"""
    evaluator = ModelEvaluator(config)
    
    # Get test data paths
    test_dir = Path(config.DATA_DIR) / 'test'
    test_images = sorted(list((test_dir / 'img').glob('*.png')))
    test_masks = sorted(list((test_dir / 'label').glob('*.png')))
    
    # Get model paths to evaluate
    model_paths = [
        str(Path(config.MODELS_DIR) / 'onnx' / 'deeplabv3.onnx'),
        str(Path(config.MODELS_DIR) / 'tensorrt' / 'deeplabv3_fp16.engine')
    ]
    
    # Run evaluation
    results = evaluator.compare_models(model_paths, test_images, test_masks)
    
    # Save results
    output_dir = Path(config.RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'evaluation_results.json'
    evaluator.save_results(results, output_path)
    
    return results

if __name__ == '__main__':
    config = Config()
    results = evaluate_models(config)
    print("\nEvaluation Results:")
    for model, metrics in results.items():
        print(f"\nModel: {model}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"FPS: {metrics['fps']:.2f}")
        print(f"Latency: {metrics['latency_ms']:.2f} ms")
