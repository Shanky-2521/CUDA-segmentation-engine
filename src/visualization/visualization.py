import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import Config

class Visualization:
    def __init__(self, config):
        self.config = config
        self.class_colors = self._generate_class_colors()
        self.class_names = self._get_class_names()
        
    def _generate_class_colors(self):
        """Generate random colors for each class"""
        colors = {}
        for i in range(self.config.NUM_CLASSES):
            colors[i] = tuple(np.random.randint(0, 256, 3).tolist())
        return colors
    
    def _get_class_names(self):
        """Get class names from config"""
        return {
            0: 'Road',
            1: 'Sidewalk',
            2: 'Building',
            3: 'Wall',
            4: 'Fence',
            5: 'Pole',
            6: 'Traffic Light',
            7: 'Traffic Sign',
            8: 'Vegetation',
            9: 'Terrain',
            10: 'Sky',
            11: 'Person',
            12: 'Rider',
            13: 'Car',
            14: 'Truck',
            15: 'Bus',
            16: 'Train',
            17: 'Motorcycle',
            18: 'Bicycle',
            19: 'Background'
        }
    
    def visualize_segmentation(self, image, prediction, output_path=None):
        """Visualize segmentation results"""
        # Create colorized prediction
        colorized = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        for class_id, color in self.class_colors.items():
            colorized[prediction == class_id] = color
        
        # Resize back to original size
        original_size = image.shape[:2]
        colorized = cv2.resize(colorized, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Colorized prediction
        ax2.imshow(colorized)
        ax2.set_title('Segmentation Map')
        ax2.axis('off')
        
        # Overlay
        alpha = 0.5
        overlay = cv2.addWeighted(image, alpha, colorized, 1-alpha, 0)
        ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        # Add legend
        legend_elements = []
        for class_id, class_name in self.class_names.items():
            color = [c/255 for c in self.class_colors[class_id]]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color=color, label=class_name))
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_comparison_grid(self, models, image_path, output_dir):
        """Create a grid comparison of different models"""
        # Load image
        image = cv2.imread(str(image_path))
        
        # Create figure
        num_models = len(models)
        fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Process each model
        for i, (model_name, model_path) in enumerate(models.items()):
            # Load model and predict
            engine = InferenceEngine(self.config)
            engine.load_model(model_path)
            prediction = engine.infer(image)
            
            # Create colorized prediction
            colorized = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
            for class_id, color in self.class_colors.items():
                colorized[prediction == class_id] = color
            
            # Resize to original size
            original_size = image.shape[:2]
            colorized = cv2.resize(colorized, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            
            # Create overlay
            alpha = 0.5
            overlay = cv2.addWeighted(image, alpha, colorized, 1-alpha, 0)
            
            # Plot
            axes[i + 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(model_name)
            axes[i + 1].axis('off')
        
        # Save figure
        output_path = Path(output_dir) / 'model_comparison.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_performance_chart(self, evaluation_results, output_dir):
        """Create performance comparison chart"""
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Extract data
        models = list(evaluation_results.keys())
        ious = [results['mean_iou'] for results in evaluation_results.values()]
        latencies = [results['latency_ms'] for results in evaluation_results.values()]
        fps = [results['fps'] for results in evaluation_results.values()]
        
        # IoU bar chart
        ax1.bar(models, ious, color='skyblue')
        ax1.set_title('Mean IoU Comparison')
        ax1.set_ylabel('Mean IoU')
        ax1.set_ylim(0, 1)
        
        # Performance bar chart
        width = 0.35
        x = np.arange(len(models))
        
        rects1 = ax2.bar(x - width/2, latencies, width, label='Latency (ms)')
        rects2 = ax2.bar(x + width/2, fps, width, label='FPS')
        
        ax2.set_title('Performance Comparison')
        ax2.set_ylabel('Value')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        
        # Add labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        # Save figure
        output_path = Path(output_dir) / 'performance_comparison.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path

if __name__ == '__main__':
    # Example usage
    config = Config()
    vis = Visualization(config)
    
    # Visualize single prediction
    image_path = 'path/to/test/image.png'
    prediction = np.random.randint(0, config.NUM_CLASSES, (512, 1024))
    image = cv2.imread(image_path)
    vis.visualize_segmentation(image, prediction)
    
    # Create comparison grid
    models = {
        'ONNX': 'path/to/onnx/model.onnx',
        'TensorRT': 'path/to/tensorrt/model.engine'
    }
    vis.create_comparison_grid(models, image_path, 'output_dir')
