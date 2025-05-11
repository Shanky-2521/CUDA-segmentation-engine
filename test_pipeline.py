import os
import cv2
import numpy as np
from src.inference.deeplabv3_engine import DeepLabV3Engine
from src.utils.logger import Logger

def test_pipeline():
    logger = Logger('test_pipeline').get_logger()
    
    try:
        # Initialize engine
        logger.info("Initializing engine...")
        engine = DeepLabV3Engine('models/trt_engines/deeplabv3_int8.engine')
        
        # Create test image
        logger.info("Creating test image...")
        test_image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        
        # Run inference
        logger.info("Running inference...")
        output = engine.infer(test_image)
        
        if output is None:
            raise RuntimeError("Inference failed: No output received")
            
        # Verify output shape
        if output.shape != (512, 1024):
            raise ValueError(f"Unexpected output shape: {output.shape}")
            
        # Save output for visualization
        output_path = 'test_output.png'
        cv2.imwrite(output_path, output * 255)
        logger.info(f"Saved output to {output_path}")
        
        # Run benchmark
        logger.info("Running benchmark...")
        benchmark_results = engine.benchmark(test_image, num_iterations=10)
        
        logger.info("Benchmark results:")
        logger.info(f"  FPS: {benchmark_results['fps']:.2f}")
        logger.info(f"  Inference time: {benchmark_results['inference_time_ms']:.2f} ms")
        logger.info(f"  Memory usage: {benchmark_results['memory_usage_mb']:.2f} MB")
        
        logger.info("Pipeline test completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}")
        raise

if __name__ == '__main__':
    test_pipeline()
