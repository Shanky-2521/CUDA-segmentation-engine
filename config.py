# Project Configuration

# Model Settings
MODEL_NAME = 'deeplabv3_resnet101'
NUM_CLASSES = 19  # Cityscapes has 19 classes
INPUT_SIZE = (512, 1024)  # Width, Height
BATCH_SIZE = 1

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

# Dataset Settings
DATA_DIR = 'data/cityscapes'
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
