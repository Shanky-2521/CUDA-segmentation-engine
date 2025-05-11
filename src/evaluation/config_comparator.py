import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import *
from src.evaluation.evaluator import Evaluator

class ConfigComparator:
    def __init__(self, test_images, test_masks):
        self.test_images = test_images
        self.test_masks = test_masks
        self.results = []
        
    def compare_configurations(self):
        """
        Compare different optimization configurations
        """
        # List of engine paths to compare
        engine_paths = {
            'FP32': 'models/trt_engines/deeplabv3_fp32.engine',
            'FP16': 'models/trt_engines/deeblabv3_fp16.engine',
            'INT8': 'models/trt_engines/deeblabv3_int8.engine'
        }
        
        print("Comparing optimization configurations...")
        
        for config_name, engine_path in engine_paths.items():
            print(f"\nEvaluating {config_name} configuration...")
            
            # Initialize evaluator
            evaluator = Evaluator(
                engine_path=engine_path,
                class_names=CLASS_NAMES
            )
            
            # Evaluate configuration
            metrics = evaluator.evaluate_model(
                self.test_images,
                self.test_masks
            )
            
            # Store results
            self.results.append({
                'configuration': config_name,
                'pixel_accuracy': metrics['pixel_accuracy'],
                'mean_accuracy': metrics['mean_accuracy'],
                'mean_iou': metrics['mean_iou'],
                'avg_inference_time': metrics['avg_inference_time'],
                'fps': metrics['fps']
            })
        
    def plot_comparison(self):
        """
        Create comparison plots
        """
        results_df = pd.DataFrame(self.results)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy metrics
        sns.barplot(x='configuration', y='pixel_accuracy', data=results_df, ax=axes[0, 0])
        axes[0, 0].set_title('Pixel Accuracy Comparison')
        axes[0, 0].set_ylim(0, 1)
        
        sns.barplot(x='configuration', y='mean_iou', data=results_df, ax=axes[0, 1])
        axes[0, 1].set_title('Mean IoU Comparison')
        axes[0, 1].set_ylim(0, 1)
        
        # Plot performance metrics
        sns.barplot(x='configuration', y='avg_inference_time', data=results_df, ax=axes[1, 0])
        axes[1, 0].set_title('Average Inference Time')
        axes[1, 0].set_ylabel('Time (s)')
        
        sns.barplot(x='configuration', y='fps', data=results_df, ax=axes[1, 1])
        axes[1, 1].set_title('Frames Per Second')
        axes[1, 1].set_ylabel('FPS')
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig('results/visualizations/config_comparison.png')
        plt.close()
        
    def save_comparison(self):
        """
        Save comparison results to CSV
        """
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('results/metrics/config_comparison.csv', index=False)
        print("Comparison results saved to results/metrics/config_comparison.csv")

def main():
    # Get test images and masks
    test_dir = Path('data/cityscapes') / 'leftImg8bit' / 'val'
    test_images = sorted(list(test_dir.glob('*/*_leftImg8bit.png')))
    test_masks = [str(p).replace('leftImg8bit', 'gtFine')
                  .replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                  for p in test_images]
    
    # Initialize and run comparison
    comparator = ConfigComparator(test_images, test_masks)
    comparator.compare_configurations()
    comparator.plot_comparison()
    comparator.save_comparison()
    
    print("Configuration comparison complete!")

if __name__ == '__main__':
    main()
