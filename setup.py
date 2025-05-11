from setuptools import setup, find_packages

setup(
    name="deeplabv3-tensorrt",
    version="0.1.0",
    author="Shashank Cuppala",
    author_email="",
    description="DeepLabV3 semantic segmentation with TensorRT optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Shanky-2521/CUDA-segmentation-engine",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=2.0.0',
        'onnx>=1.14.0',
        'tensorrt>=8.6.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'pycuda>=2023.1.0',
        'pytest>=7.4.0',
        'Pillow>=10.0.0',
        'scikit-learn>=1.3.0',
        'tqdm>=4.65.0',
        'logging>=0.5.1.2'
    ],
    entry_points={
        'console_scripts': [
            'deeplabv3-trt=create_deeplabv3_engine:main',
        ],
    },
)
