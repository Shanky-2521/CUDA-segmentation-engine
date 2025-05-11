# Project Configuration

import sys
import os
import torch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Data configuration
    DATA_DIR: str = "data/cityscapes"
    NUM_CLASSES: int = 19
    
    # Training configuration
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 2
    NUM_WORKERS: int = 4
    NUM_EPOCHS: int = 2
    
    # Model configuration
    MODEL_NAME: str = "deeplabv3_resnet101"
    
    # Paths
    CHECKPOINT_DIR: str = "models"
    ONNX_PATH: str = 'models/deeplabv3.onnx'
    TRT_ENGINE_DIR: str = 'models/trt_engines'
    
    # Image dimensions
    IMG_WIDTH: int = 1024
    IMG_HEIGHT: int = 512
    
    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization Settings
    OPTIMIZATION_CONFIGS = {
        'fp32': {
            'precision': 'fp32',
            'workspace_size': 1 << 30,  # 1GB
            'max_batch_size': 1
        },
        'fp16': {
            'precision': 'fp16',
            'workspace_size': 1 << 30,  # 1GB
            'max_batch_size': 1
        },
        'int8': {
            'precision': 'int8',
            'workspace_size': 1 << 30,  # 1GB
            'max_batch_size': 1
        }
    }

    # Performance Metrics
    METRICS = [
        'inference_time',
        'memory_usage',
        'fps',
        'accuracy'
    ]

    MODELS_DIR: str = "models"
    RESULTS_DIR: str = "results"

TEST_IMAGES = 'data/test_images'

# Performance Metrics
METRICS = [
    'inference_time',
    'memory_usage',
    'fps',
    'accuracy'
]

# Visualization Settings
VISUALIZATION_DIR = 'results/visualizations'
METRICS_DIR = 'results/metrics'

# Paths
ONNX_PATH = 'models/deeplabv3.onnx'
TRT_ENGINE_DIR = 'models/trt_engines'

# Class Names (Cityscapes)
CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]
