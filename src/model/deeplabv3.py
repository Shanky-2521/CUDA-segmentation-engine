import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=19):
        super(DeepLabV3, self).__init__()
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x):
        return self.model(x)['out']

    def export_onnx(self, onnx_path):
        """
        Export the model to ONNX format
        """
        dummy_input = torch.randn(1, 3, 512, 1024)
        torch.onnx.export(
            self,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                         'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
        )
