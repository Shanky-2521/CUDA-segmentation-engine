import cv2
import numpy as np
from pathlib import Path
import json
from config import *

class DataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.class_names = CLASS_NAMES
        self.class_to_color = self._create_class_to_color_map()
        
    def _create_class_to_color_map(self):
        """
        Create a mapping from class IDs to colors
        """
        colors = plt.cm.get_cmap('tab20')
        return {
            i: (colors(i)[:3] * 255).astype(np.uint8)
            for i in range(len(self.class_names))
        }
        
    def preprocess_image(self, image_path):
        """
        Preprocess a single image
        """
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, INPUT_SIZE)
        return image
        
    def preprocess_mask(self, mask_path):
        """
        Preprocess a single mask
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        return mask
        
    def create_dataset(self, split='train'):
        """
        Create dataset for training/validation
        """
        image_dir = self.data_dir / 'leftImg8bit' / split
        label_dir = self.data_dir / 'gtFine' / split
        
        images = []
        masks = []
        
        for city in image_dir.iterdir():
            for image_path in (city / '*.png').glob():
                # Get corresponding label path
                image_name = image_path.stem
                label_name = f"{image_name}_gtFine_labelIds.png"
                label_path = label_dir / city.name / label_name
                
                if label_path.exists():
                    images.append(str(image_path))
                    masks.append(str(label_path))
        
        return images, masks
        
    def visualize_sample(self, image, mask, prediction=None):
        """
        Visualize an image with its ground truth mask and prediction
        """
        # Create visualization canvas
        height, width = image.shape[:2]
        canvas = np.zeros((height, width * 3, 3), dtype=np.uint8)
        
        # Add original image
        canvas[:, :width] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Add ground truth
        gt_vis = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id in np.unique(mask):
            if class_id in self.class_to_color:
                gt_vis[mask == class_id] = self.class_to_color[class_id]
        canvas[:, width:2*width] = gt_vis
        
        # Add prediction if provided
        if prediction is not None:
            pred_vis = np.zeros((height, width, 3), dtype=np.uint8)
            for class_id in np.unique(prediction):
                if class_id in self.class_to_color:
                    pred_vis[prediction == class_id] = self.class_to_color[class_id]
            canvas[:, 2*width:] = pred_vis
        
        return canvas
        
    def save_preprocessed_data(self, split='train'):
        """
        Save preprocessed data to disk
        """
        output_dir = Path('data') / 'preprocessed' / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images, masks = self.create_dataset(split)
        
        for i, (image_path, mask_path) in enumerate(zip(images, masks)):
            # Load and preprocess
            image = self.preprocess_image(image_path)
            mask = self.preprocess_mask(mask_path)
            
            # Save
            np.save(output_dir / f'image_{i}.npy', image)
            np.save(output_dir / f'mask_{i}.npy', mask)
        
        # Save metadata
        metadata = {
            'num_samples': len(images),
            'image_size': INPUT_SIZE,
            'class_names': CLASS_NAMES
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
