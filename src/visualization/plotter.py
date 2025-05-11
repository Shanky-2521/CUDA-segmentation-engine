import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import *
import os

class Visualization:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.metrics_dir = os.path.join(results_dir, 'metrics')
        self.vis_dir = os.path.join(results_dir, 'visualizations')
        
    def plot_training_curves(self, train_losses, val_losses):
        """
        Plot training and validation loss curves
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.vis_dir, 'training_curves.png'))
        plt.close()
        
    def plot_benchmark_results(self):
        """
        Plot benchmark results comparing different configurations
        """
        # Load benchmark results
        df = pd.read_csv(os.path.join(self.metrics_dir, 'benchmark_results.csv'))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot inference time
        sns.barplot(x='config', y='avg_inference_time', data=df, ax=axes[0, 0])
        axes[0, 0].set_title('Average Inference Time')
        axes[0, 0].set_ylabel('Time (s)')
        
        # Plot FPS
        sns.barplot(x='config', y='fps', data=df, ax=axes[0, 1])
        axes[0, 1].set_title('Frames Per Second')
        axes[0, 1].set_ylabel('FPS')
        
        # Plot memory usage
        sns.barplot(x='config', y='memory_usage', data=df, ax=axes[1, 0])
        axes[1, 0].set_title('Memory Usage')
        axes[1, 0].set_ylabel('Memory (MB)')
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'benchmark_comparison.png'))
        plt.close()
        
    def plot_class_distribution(self, predictions):
        """
        Plot distribution of predicted classes
        """
        # Flatten predictions and count occurrences
        class_counts = np.zeros(len(CLASS_NAMES))
        for pred in predictions:
            unique, counts = np.unique(pred, return_counts=True)
            class_counts[unique] += counts
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(CLASS_NAMES, class_counts)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution in Predictions')
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'class_distribution.png'))
        plt.close()
        
    def create_comparison_grid(self, images, ground_truths, predictions, num_samples=5):
        """
        Create a grid comparing original images, ground truth, and predictions
        """
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i in range(num_samples):
            # Original image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(ground_truths[i])
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            axes[i, 2].imshow(predictions[i])
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'comparison_grid.png'))
        plt.close()

def main():
    # Create visualization directory
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Initialize visualization
    vis = Visualization()
    
    # Load benchmark results and create plots
    vis.plot_benchmark_results()
    
    # Example usage for training curves
    # train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    # val_losses = [0.6, 0.5, 0.4, 0.3, 0.2]
    # vis.plot_training_curves(train_losses, val_losses)

if __name__ == '__main__':
    main()
