# Benchmarking Guide

## Overview
This document provides a comprehensive guide for benchmarking different optimization configurations of the semantic segmentation model using TensorRT.

## Setup Instructions

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Prepare Dataset**
- Download Cityscapes dataset from https://www.cityscapes-dataset.com/
- Place test images in `data/test_images` directory

3. **Create Directories**
```bash
mkdir -p data/test_images
mkdir -p results/visualizations
mkdir -p results/metrics
mkdir -p models/trt_engines
```

## Running Benchmark

1. **Create ONNX Model**
```bash
python src/model/deeplabv3.py
```

2. **Run Benchmark**
```bash
python src/evaluation/benchmark.py
```

## Optimization Configurations
The benchmark evaluates three different optimization configurations:

1. **FP32**
   - Full precision (32-bit floating point)
   - Baseline performance

2. **FP16**
   - Half precision (16-bit floating point)
   - Balances performance and accuracy

3. **INT8**
   - Integer precision (8-bit)
   - Maximum performance optimization

## Performance Metrics
The benchmark measures and reports the following metrics:

1. **Inference Time**
   - Average time per inference
   - Measured in seconds

2. **FPS (Frames Per Second)**
   - Number of frames processed per second
   - Key metric for real-time applications

3. **Memory Usage**
   - GPU memory consumption
   - Measured in MB

4. **Accuracy**
   - Model accuracy after optimization
   - Compared against baseline

## Results
Benchmark results are saved in:
- `results/metrics/benchmark_results.csv` - Performance metrics
- `results/visualizations/` - Segmentation visualizations

## Visualization
The benchmark generates visualizations for each configuration, allowing you to compare:
- Inference speed
- Memory usage
- Segmentation quality
- Overall performance
