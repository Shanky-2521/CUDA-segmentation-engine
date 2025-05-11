import tensorrt as trt
import numpy as np
import cv2
import time
from pathlib import Path
import matplotlib.pyplot as plt

class SemanticSegmentationInference:
    def __init__(self, engine_path, class_names):
        self.engine_path = Path(engine_path)
        self.class_names = class_names
        self.engine = None
        self.context = None
        self.input_shape = None
        self.output_shape = None
        
    def initialize(self):
        """
        Initialize TensorRT engine and context
        """
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.input_shape = tuple(self.engine.get_binding_shape(0))
        self.output_shape = tuple(self.engine.get_binding_shape(1))
        
    def preprocess(self, image):
        """
        Preprocess input image
        """
        # Resize and normalize
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image
        
    def postprocess(self, output):
        """
        Postprocess network output
        """
        # Get predicted class
        pred = np.argmax(output[0], axis=0)
        return pred
        
    def infer(self, image):
        """
        Run inference on a single image
        """
        if self.engine is None:
            self.initialize()
            
        # Preprocess
        input_data = self.preprocess(image)
        
        # Allocate buffers
        inputs = [np.ascontiguousarray(input_data)]
        outputs = [np.empty(self.output_shape, dtype=np.float32)]
        
        # Create stream
        stream = trt.cuda.Stream()
        
        # Transfer input data to device
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        
        # Execute model
        self.context.execute_async_v2(
            bindings=[i.device for i in inputs] + [o.device for o in outputs],
            stream_handle=stream.handle
        )
        
        # Transfer predictions back
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        
        # Synchronize and return results
        stream.synchronize()
        return self.postprocess(outputs[0])
        
    def visualize(self, image, prediction):
        """
        Visualize segmentation results
        """
        # Create color map
        colors = plt.cm.get_cmap('tab20')
        
        # Create colored segmentation mask
        mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        for i in range(len(self.class_names)):
            mask[prediction == i] = (colors(i)[:3] * 255).astype(np.uint8)
        
        # Overlay mask on original image
        overlay = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        return overlay
