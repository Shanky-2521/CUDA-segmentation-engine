import os
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from config import *
from src.model.deeplabv3 import DeepLabV3
from src.optimization.tensorrt_optimizer import TensorRTOptimizer
from src.inference.inference import SemanticSegmentationInference

class Benchmark:
    def __init__(self):
        self.results = []
        self.class_names = CLASS_NAMES
        
    def create_engines(self):
        """
        Create TensorRT engines for different configurations
        """
        # Create output directory
        os.makedirs(TRT_ENGINE_DIR, exist_ok=True)
        
        # Create ONNX model if it doesn't exist
        if not os.path.exists(ONNX_PATH):
            model = DeepLabV3(num_classes=NUM_CLASSES)
            model.export_onnx(ONNX_PATH)
            print(f"ONNX model saved to {ONNX_PATH}")
        
        # Create engines for each configuration
        for config_name, config in OPTIMIZATION_CONFIGS.items():
            engine_path = os.path.join(TRT_ENGINE_DIR, f"deeplabv3_{config_name}.engine")
            optimizer = TensorRTOptimizer(
                onnx_path=ONNX_PATH,
                trt_path=engine_path,
                precision=config['precision']
            )
            engine = optimizer.build_engine(
                batch_size=config['max_batch_size'],
                image_size=INPUT_SIZE
            )
            print(f"Created {config_name} engine: {engine_path}")
            
    def measure_performance(self):
        """
        Measure performance for each configuration
        """
        test_images = [f for f in os.listdir(TEST_IMAGES) if f.endswith('.png')]
        
        for config_name, config in OPTIMIZATION_CONFIGS.items():
            engine_path = os.path.join(TRT_ENGINE_DIR, f"deeplabv3_{config_name}.engine")
            
            # Initialize inference
            inferencer = SemanticSegmentationInference(
                engine_path=engine_path,
                class_names=self.class_names
            )
            
            # Warm up
            for _ in range(10):
                dummy_input = np.random.randn(1, 3, *INPUT_SIZE).astype(np.float32)
                inferencer.infer(dummy_input)
            
            # Measure performance
            inference_times = []
            for image_path in test_images:
                image = cv2.imread(os.path.join(TEST_IMAGES, image_path))
                
                # Measure inference time
                start = time.time()
                prediction = inferencer.infer(image)
                end = time.time()
                
                inference_time = end - start
                inference_times.append(inference_time)
                
                # Save visualization
                visualization = inferencer.visualize(image, prediction)
                vis_path = os.path.join(VISUALIZATION_DIR, f"{config_name}_{image_path}")
                cv2.imwrite(vis_path, visualization)
            
            # Calculate metrics
            avg_inference_time = np.mean(inference_times)
            fps = 1 / avg_inference_time
            memory_usage = self._get_memory_usage()  # Placeholder for actual memory measurement
            
            # Save results
            self.results.append({
                'config': config_name,
                'precision': config['precision'],
                'avg_inference_time': avg_inference_time,
                'fps': fps,
                'memory_usage': memory_usage
            })
            
            print(f"{config_name} results:")
            print(f"Average Inference Time: {avg_inference_time:.4f} s")
            print(f"FPS: {fps:.2f}")
            print(f"Memory Usage: {memory_usage} MB")
            print("-" * 50)
            
    def _get_memory_usage(self):
        """
        Placeholder for memory measurement
        In practice, this would use NVIDIA-smi or similar tool
        """
        return 0  # Replace with actual memory measurement
        
    def save_results(self):
        """
        Save benchmark results to CSV
        """
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(METRICS_DIR, 'benchmark_results.csv'), index=False)
        print(f"Benchmark results saved to {METRICS_DIR}/benchmark_results.csv")

def main():
    # Create necessary directories
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Run benchmark
    benchmark = Benchmark()
    benchmark.create_engines()
    benchmark.measure_performance()
    benchmark.save_results()

if __name__ == "__main__":
    main()
