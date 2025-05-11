import torch
import torch.nn as nn
import torch.onnx
from torchvision import models
import os

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=19):
        super(DeepLabV3, self).__init__()
        # Load pre-trained DeepLabV3 with ResNet backbone
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        
        # Modify the classifier to match our number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    def forward(self, x):
        return self.model(x)['out']

def create_deeplabv3_onnx():
    # Create model
    model = DeepLabV3(num_classes=19)
    model.eval()
    
    # Create dummy input with correct shape (1, 3, 512, 1024)
    dummy_input = torch.randn(1, 3, 512, 1024)
    
    # Export to ONNX
    onnx_path = 'models/deeplabv3.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Created ONNX model at {onnx_path}")
    return onnx_path

if __name__ == '__main__':
    create_deeplabv3_onnx()
