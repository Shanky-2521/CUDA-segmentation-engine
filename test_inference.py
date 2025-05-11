import cv2
import numpy as np
from pathlib import Path
from src.inference.simple_inference_engine import SimpleInferenceEngine
from src.config import Config

def test_inference():
    # Initialize config
    config = Config()
    
    # Create inference engine
    engine = SimpleInferenceEngine(
        engine_path='models/trt_engines/pretrained_engine.engine'
    )
    
    # Create a simple test image
    print("Creating test image...")
    image = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8)
    # Create a gradient pattern
    for i in range(config.IMG_HEIGHT):
        image[i, :, :] = [i, i, i]
    
    # Run inference
    print("\nRunning inference...")
    prediction = engine.infer(image)
    
    # Visualize results
    print("\nVisualizing results...")
    overlay = engine.visualize(image, prediction)
    
    # Save results
    output_dir = Path(config.VISUALIZATION_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "test_inference_result.png"
    cv2.imwrite(str(output_path), overlay)
    print(f"Saved visualization to {output_path}")
    
    # Benchmark performance
    print("\nRunning benchmark...")
    latency, fps = engine.benchmark(image, num_runs=100)
    print(f"\nBenchmark Results:")
    print(f"Average Latency: {latency:.2f} ms")
    print(f"Frames per Second: {fps:.2f} FPS")
    
    return latency, fps

if __name__ == '__main__':
    test_inference()
