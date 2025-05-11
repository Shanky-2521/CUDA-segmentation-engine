import torch
import os
from src.model.deeplabv3 import DeepLabV3
from src.optimization.tensorrt_optimizer import TensorRTOptimizer
from src.inference.inference_engine import TensorRTInferenceEngine
import cv2
import numpy as np
from pathlib import Path

def main():
    # Configuration
    NUM_CLASSES = 19
    IMAGE_SIZE = (512, 1024)
    PRECISION = 'fp16'
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Load or train model
    print("\nStep 1: Loading DeepLabV3 model...")
    model = DeepLabV3(num_classes=NUM_CLASSES)
    
    # Step 2: Export to ONNX
    onnx_path = Path('models/deeplabv3.onnx')
    print(f"\nStep 2: Exporting model to ONNX: {onnx_path}")
    model.export_onnx(onnx_path)
    
    # Step 3: Optimize with TensorRT
    trt_path = Path('models/deeplabv3.trt')
    print(f"\nStep 3: Optimizing with TensorRT: {trt_path}")
    optimizer = TensorRTOptimizer(onnx_path, trt_path, precision=PRECISION)
    engine = optimizer.build_engine(batch_size=1, image_size=IMAGE_SIZE)
    
    if engine is None:
        print("Failed to build TensorRT engine")
        return
    
    # Step 4: Create inference engine
    print("\nStep 4: Creating inference engine...")
    inference_engine = TensorRTInferenceEngine(trt_path)
    
    # Step 5: Test inference
    print("\nStep 5: Running inference test...")
    # Create a test image (in practice, you would load a real image)
    test_image = np.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8) * 128
    
    # Run inference
    segmentation = inference_engine.infer(test_image)
    
    # Save visualization
    colors = np.random.randint(0, 255, (NUM_CLASSES, 3), dtype=np.uint8)
    colored_segmentation = colors[segmentation]
    cv2.imwrite('results/test_segmentation.png', colored_segmentation)
    
    # Step 6: Benchmark performance
    print("\nStep 6: Running performance benchmark...")
    benchmark_results = inference_engine.benchmark([test_image] * 10, num_iterations=10)
    
    print("\nPerformance Results:")
    print(f"Mean Latency: {benchmark_results['mean_latency']:.2f} ms")
    print(f"FPS: {benchmark_results['fps']:.2f}")
    print(f"Standard Deviation: {benchmark_results['std_dev']:.2f} ms")

if __name__ == "__main__":
    main()
