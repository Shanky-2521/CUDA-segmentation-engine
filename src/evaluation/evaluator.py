import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from config import *
from src.inference.inference import SemanticSegmentationInference

class Evaluator:
    def __init__(self, engine_path, class_names):
        self.inferencer = SemanticSegmentationInference(engine_path, class_names)
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def compute_metrics(self, predictions, ground_truths):
        """
        Compute evaluation metrics
        """
        # Initialize metrics
        metrics = {
            'pixel_accuracy': 0,
            'mean_accuracy': 0,
            'mean_iou': 0,
            'class_iou': np.zeros(self.num_classes),
            'class_accuracy': np.zeros(self.num_classes)
        }
        
        # Compute confusion matrix
        cm = np.zeros((self.num_classes, self.num_classes))
        
        for pred, gt in zip(predictions, ground_truths):
            # Flatten predictions and ground truth
            pred = pred.flatten()
            gt = gt.flatten()
            
            # Update confusion matrix
            cm += confusion_matrix(gt, pred, labels=np.arange(self.num_classes))
        
        # Compute metrics
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        
        # Pixel accuracy
        metrics['pixel_accuracy'] = np.sum(tp) / np.sum(cm)
        
        # Class accuracy
        class_acc = tp / (tp + fn)
        metrics['class_accuracy'] = class_acc
        metrics['mean_accuracy'] = np.nanmean(class_acc)
        
        # Class IoU
        iou = tp / (tp + fp + fn)
        metrics['class_iou'] = iou
        metrics['mean_iou'] = np.nanmean(iou)
        
        return metrics
        
    def evaluate_model(self, test_images, test_masks):
        """
        Evaluate model on test set
        """
        predictions = []
        ground_truths = []
        inference_times = []
        
        print("Evaluating model...")
        
        for image_path, mask_path in zip(test_images, test_masks):
            # Load and preprocess image
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Measure inference time
            start = time.time()
            prediction = self.inferencer.infer(image)
            end = time.time()
            
            inference_times.append(end - start)
            
            # Store results
            predictions.append(prediction)
            ground_truths.append(mask)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, ground_truths)
        
        # Add inference time metrics
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['fps'] = 1 / metrics['avg_inference_time']
        
        return metrics
        
    def save_metrics(self, metrics, output_path):
        """
        Save evaluation metrics to a file
        """
        with open(output_path, 'w') as f:
            f.write("Evaluation Metrics:\n")
            f.write(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n")
            f.write(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}\n")
            f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n")
            f.write(f"Average Inference Time: {metrics['avg_inference_time']:.4f}s\n")
            f.write(f"FPS: {metrics['fps']:.2f}\n\n")
            
            f.write("Class-wise Metrics:\n")
            for i in range(self.num_classes):
                f.write(f"{self.class_names[i]}:\n")
                f.write(f"  Accuracy: {metrics['class_accuracy'][i]:.4f}\n")
                f.write(f"  IoU: {metrics['class_iou'][i]:.4f}\n")
                f.write("-" * 50 + "\n")

def main():
    # Initialize evaluator
    evaluator = Evaluator(
        engine_path='models/trt_engines/deeplabv3_fp32.engine',
        class_names=CLASS_NAMES
    )
    
    # Get test images and masks
    test_dir = Path('data/cityscapes') / 'leftImg8bit' / 'val'
    test_images = sorted(list(test_dir.glob('*/*_leftImg8bit.png')))
    test_masks = [str(p).replace('leftImg8bit', 'gtFine')
                  .replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                  for p in test_images]
    
    # Evaluate model
    metrics = evaluator.evaluate_model(test_images, test_masks)
    
    # Save metrics
    evaluator.save_metrics(metrics, 'results/metrics/evaluation_results.txt')
    print("Evaluation complete! Results saved to results/metrics/evaluation_results.txt")

if __name__ == '__main__':
    main()
