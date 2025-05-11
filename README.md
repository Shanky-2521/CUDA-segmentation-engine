# NVIDIA Semantic Segmentation Optimization

Author: Shashank Cuppala

This project demonstrates the optimization of a semantic segmentation deep learning model for real-time inference on NVIDIA hardware using TensorRT and ONNX.

## Project Structure
```
.
├── models/              # Model weights and configurations
├── data/               # CUDA Segmentation Engine

This project implements a DeepLabV3 semantic segmentation model optimized with NVIDIA TensorRT for high-performance inference on NVIDIA GPUs.

## Features

- DeepLabV3 model with ResNet101 backbone
- TensorRT optimization with FP16 and INT8 precision
- CUDA-accelerated inference
- Batch processing support
- Comprehensive error handling and logging
- Performance benchmarking
- Unit tests for validation

## Prerequisites

### System Requirements

- NVIDIA GPU with CUDA capability
- CUDA 12.9 or higher
- TensorRT 10.10.0.31 or higher
- Python 3.8+

### Software Dependencies

```bash
pip install -r requirements.txt
```

### Git LFS Setup

This repository uses Git LFS (Large File Storage) for managing large model files. To download the models:

1. Install Git LFS:
```bash
# Windows
winget install Git.LFS

# Linux
apt-get install git-lfs

# macOS
brew install git-lfs
```

2. Initialize Git LFS:
```bash
git lfs install
```

3. Clone the repository:
```bash
git clone https://github.com/Shanky-2521/CUDA-segmentation-engine.git
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure CUDA and TensorRT

1. Install CUDA Toolkit from NVIDIA (https://developer.nvidia.com/cuda-downloads)
2. Install TensorRT from NVIDIA (https://developer.nvidia.com/tensorrt)
3. Set up environment variables:

```bash
# Linux
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/tensorrt/lib
export PATH=$PATH:/usr/local/cuda/bin:/usr/local/tensorrt/bin

# Windows
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
setx PATH "%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
```

### 3. Download Models

The models will be automatically downloaded when you clone the repository using Git LFS. If you need to download them manually:

```bash
git lfs pull
```

## Usage

### Create TensorRT Engine

```bash
python create_deeplabv3_engine.py
```

This will create a TensorRT engine with the following configuration:
- Batch size: 4
- Workspace size: 1GB
- FP16 enabled
- INT8 enabled
- Calibration dataset size: 500

### Run Inference

```bash
python test_deeplabv3.py
```

This script will:
1. Load the TensorRT engine
2. Test inference on random images
3. Run performance benchmarks
4. Process real images
5. Save results for visualization

### Run Tests

```bash
python -m unittest tests/test_deeplabv3.py
```

The tests will validate:
- Engine initialization
- Inference on random images
- Inference on real images
- Batch processing
- Performance benchmarks

## Performance

The optimized model achieves:
- Average inference time: ~129ms
- FPS: ~7.7
- Memory usage: ~133MB

## Directory Structure

```
.
├── models/                 # Model files
│   ├── deeplabv3.onnx     # ONNX model
│   └── trt_engines/       # TensorRT engines
├── src/                   # Source code
│   ├── inference/         # Inference engine
│   └── utils/            # Utility functions
├── tests/                 # Test files
│   ├── data/             # Test data
│   └── output/           # Test output
└── requirements.txt       # Python dependencies
```

## Logging

The project uses comprehensive logging:
- All logs are saved in `logs/` directory
- Detailed error messages
- Performance metrics
- Debug information

## Error Handling

The project includes:
- Custom exception classes
- Comprehensive error messages
- Graceful failure handling
- Resource cleanup

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- NVIDIA TensorRT
- PyTorch
- DeepLabV3 Model
- Git LFS

## CUDA Setup
This project requires CUDA for GPU acceleration. Make sure:
1. Your system has an NVIDIA GPU
2. NVIDIA CUDA Toolkit is installed
3. CUDA is properly configured:
   ```bash
   # Check CUDA installation
   nvcc --version
   
   # Check GPU availability
   nvidia-smi
   ```

## Performance Optimization
The project includes CUDA optimizations:
1. Mixed precision training (FP16)
2. CUDA memory management
3. TensorRT GPU optimizations
4. CUDA streams for parallel processing
5. Memory allocation optimization

## Troubleshooting CUDA
If you encounter CUDA-related errors:
1. Verify CUDA installation
2. Check GPU drivers
3. Ensure CUDA toolkit version matches requirements
4. Check GPU memory availability

## Performance Metrics
- Inference latency
- Memory usage
- Frame rate
- Accuracy comparison

## License
MIT License
