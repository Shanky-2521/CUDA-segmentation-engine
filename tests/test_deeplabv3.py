import unittest
import numpy as np
import cv2
from pathlib import Path
from src.inference.deeplabv3_engine import DeepLabV3Engine
from src.utils.logger import Logger

class TestDeepLabV3Engine(unittest.TestCase):
    def setUp(self):
        self.logger = Logger('test_deeplabv3').get_logger()
        self.engine_path = Path('models/trt_engines/deeplabv3_int8.engine')
        self.test_images_dir = Path('tests/data/images')
        self.test_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test images if they don't exist
        if not list(self.test_images_dir.glob('*.png')):
            self._create_test_images()
    
    def _create_test_images(self):
        """Create test images for validation"""
        # Create random images
        for i in range(3):
            img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
            cv2.imwrite(str(self.test_images_dir / f'test_{i}.png'), img)
        
        # Create specific pattern images
        # Solid color image
        solid = np.ones((512, 1024, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(self.test_images_dir / 'solid.png'), solid)
        
        # Gradient image
        gradient = np.zeros((512, 1024, 3), dtype=np.uint8)
        for i in range(512):
            gradient[i] = np.linspace(0, 255, 1024).reshape(1, -1, 1).astype(np.uint8)
        cv2.imwrite(str(self.test_images_dir / 'gradient.png'), gradient)
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        try:
            engine = DeepLabV3Engine(str(self.engine_path))
            self.assertIsNotNone(engine)
            self.logger.info("Engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Engine initialization failed: {str(e)}")
            self.fail(str(e))
    
    def test_inference_random_image(self):
        """Test inference on random images"""
        engine = DeepLabV3Engine(str(self.engine_path))
        
        # Test with random image
        img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        
        try:
            output = engine.infer(img)
            self.assertIsNotNone(output)
            self.assertEqual(output.shape, (512, 1024))
            self.logger.info("Inference on random image successful")
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            self.fail(str(e))
    
    def test_inference_real_images(self):
        """Test inference on real images"""
        engine = DeepLabV3Engine(str(self.engine_path))
        
        for img_path in self.test_images_dir.glob('*.png'):
            try:
                img = cv2.imread(str(img_path))
                output = engine.infer(img)
                
                self.assertIsNotNone(output)
                self.assertEqual(output.shape, (512, 1024))
                
                # Save output for visualization
                output_path = Path('tests/output') / img_path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), output * 255)
                
                self.logger.info(f"Processed {img_path.name} successfully")
            except Exception as e:
                self.logger.error(f"Failed to process {img_path.name}: {str(e)}")
                self.fail(str(e))
    
    def test_batch_inference(self):
        """Test batch inference"""
        engine = DeepLabV3Engine(str(self.engine_path))
        
        # Create batch of images
        batch_size = 4
        batch = []
        for _ in range(batch_size):
            img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
            batch.append(img)
        
        try:
            # Process batch
            outputs = []
            for img in batch:
                output = engine.infer(img)
                outputs.append(output)
                
            self.assertEqual(len(outputs), batch_size)
            for output in outputs:
                self.assertEqual(output.shape, (512, 1024))
            
            self.logger.info("Batch inference successful")
        except Exception as e:
            self.logger.error(f"Batch inference failed: {str(e)}")
            self.fail(str(e))
    
    def test_benchmark(self):
        """Test performance benchmarking"""
        engine = DeepLabV3Engine(str(self.engine_path))
        
        # Create test image
        img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        
        try:
            results = engine.benchmark(img, num_iterations=100)
            self.assertGreater(results['fps'], 0)
            self.assertLess(results['inference_time_ms'], 1000)
            
            self.logger.info(f"Benchmark results: {results}")
        except Exception as e:
            self.logger.error(f"Benchmark failed: {str(e)}")
            self.fail(str(e))

if __name__ == '__main__':
    unittest.main()
