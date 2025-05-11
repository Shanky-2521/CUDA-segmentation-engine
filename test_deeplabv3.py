import cv2
import numpy as np
from src.inference.deeplabv3_engine import DeepLabV3Engine
import os

def test_inference(engine):
    """Test inference with random image"""
    # Test with a random image
    print("\nTesting inference with random image...")
    test_image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    
    # Run inference
    output = engine.infer(test_image)
    
    # Save results
    cv2.imwrite('output_random.png', output)
    print('Segmentation result saved as output_random.png')
    
    # Run benchmark
    print("\nRunning benchmark...")
    avg_time, fps = engine.benchmark(test_image, num_iterations=100)
    print(f"\nBenchmark Results:")
    print(f"Inference time: {avg_time * 1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Input shape: {engine.input_shape}")
    print(f"Output shape: {engine.output_shape}")

def test_real_image(engine):
    """Test inference with real image"""
    # Load a real image (create a test image if needed)
    test_image_path = 'test_images/test_street.jpg'
    if not os.path.exists(test_image_path):
        # Create a test image with some patterns
        print("Creating test image...")
        img = np.zeros((512, 1024, 3), dtype=np.uint8)
        # Add some patterns
        cv2.rectangle(img, (0, 0), (512, 256), (0, 255, 0), -1)  # green (road)
        cv2.rectangle(img, (0, 256), (512, 512), (255, 0, 0), -1)  # blue (sky)
        cv2.rectangle(img, (512, 0), (1024, 512), (0, 0, 255), -1)  # red (building)
        cv2.imwrite(test_image_path, img)
    
    # Load image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Could not load image from {test_image_path}")
        exit(1)
    
    # Run inference
    print("\nRunning inference on test image...")
    output = engine.infer(image)
    
    # Save results
    cv2.imwrite('output_real_original.png', image)
    cv2.imwrite('output_real_segmentation.png', output)
    print('Original image saved as output_real_original.png')
    print('Segmentation result saved as output_real_segmentation.png')

def main():
    print("\nTesting DeepLabV3 TensorRT Engine")
    print("=================================")
    
    # Create engine once
    engine_path = 'models/trt_engines/deeplabv3_fp16.engine'
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found at {engine_path}")
        exit(1)
    
    engine = DeepLabV3Engine(engine_path)
    
    # Run tests with the same engine instance
    test_inference(engine)
    test_real_image(engine)

if __name__ == '__main__':
    main()
