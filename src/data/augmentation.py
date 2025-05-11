import numpy as np
import cv2
from typing import Tuple, Dict
import random
from config import *

class DataAugmentor:
    def __init__(self, augment_config: Dict = None):
        """
        Initialize data augmentation with configuration
        """
        self.config = {
            'horizontal_flip': 0.5,  # Probability of horizontal flip
            'vertical_flip': 0.2,   # Probability of vertical flip
            'brightness': 0.2,      # Brightness adjustment range
            'contrast': 0.2,        # Contrast adjustment range
            'rotation': 10,         # Maximum rotation angle in degrees
            'scale': (0.8, 1.2),    # Scale range
            'shear': 0.2,           # Shear factor
            'translate': 0.2        # Translation factor
        }
        
        if augment_config:
            self.config.update(augment_config)
            
    def _random_flip(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly flip the image and mask horizontally or vertically
        """
        if random.random() < self.config['horizontal_flip']:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        if random.random() < self.config['vertical_flip']:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        return image, mask
        
    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust image brightness
        """
        if random.random() < self.config['brightness']:
            factor = 1.0 + random.uniform(-self.config['brightness'], self.config['brightness'])
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        return image
        
    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust image contrast
        """
        if random.random() < self.config['contrast']:
            factor = 1.0 + random.uniform(-self.config['contrast'], self.config['contrast'])
            mean = np.mean(image)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return image
        
    def _random_rotation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly rotate the image and mask
        """
        angle = random.uniform(-self.config['rotation'], self.config['rotation'])
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        return image, mask
        
    def _random_scale(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly scale the image and mask
        """
        scale = random.uniform(self.config['scale'][0], self.config['scale'][1])
        
        h, w = image.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        
        return image, mask
        
    def _random_shear(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly shear the image and mask
        """
        h, w = image.shape[:2]
        shear_factor = random.uniform(-self.config['shear'], self.config['shear'])
        
        M = np.float32([
            [1, shear_factor, 0],
            [shear_factor, 1, 0]
        ])
        
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        return image, mask
        
    def _random_translation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly translate the image and mask
        """
        h, w = image.shape[:2]
        
        max_dx = int(w * self.config['translate'])
        max_dy = int(h * self.config['translate'])
        
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        return image, mask
        
    def apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all augmentations in sequence
        """
        # Random scale
        image, mask = self._random_scale(image, mask)
        
        # Random rotation
        image, mask = self._random_rotation(image, mask)
        
        # Random shear
        image, mask = self._random_shear(image, mask)
        
        # Random translation
        image, mask = self._random_translation(image, mask)
        
        # Random flips
        image, mask = self._random_flip(image, mask)
        
        # Adjust brightness
        image = self._adjust_brightness(image)
        
        # Adjust contrast
        image = self._adjust_contrast(image)
        
        return image, mask
        
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.apply_augmentation(image, mask)

def get_preprocessing_transforms():
    """
    Get preprocessing transforms for the model
    """
    def transform(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize to input size
        image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        return image, mask
    
    return transform
