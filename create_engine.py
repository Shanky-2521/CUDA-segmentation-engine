import tensorrt as trt
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_engine():
    """
    Create a simple TensorRT engine from ONNX model
    """
    try:
        # Create builder and network
        logger.info("Creating builder...")
        builder = trt.Builder(trt.Logger(trt.Logger.INFO))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Create parser
        logger.info("Creating parser...")
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.INFO))
        
        # Load ONNX model
        logger.info("Loading ONNX model...")
        with open('models/deeplabv3.onnx', 'rb') as model:
            if not parser.parse(model.read()):
                errors = [parser.get_error(i) for i in range(parser.num_errors)]
                for error in errors:
                    logger.error(f"ONNX parser error: {error}")
                raise RuntimeError("Failed to parse ONNX file")
        
        # Create builder config
        logger.info("Creating builder config...")
        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Enable FP16
        if builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 enabled")
        
        # Set optimization profile
        logger.info("Setting optimization profile...")
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, 512, 1024), (4, 3, 512, 1024), (4, 3, 512, 1024))
        builder_config.add_optimization_profile(profile)
        
        # Build engine
        logger.info("Building engine...")
        engine = builder.build_serialized_network(network, builder_config)
        
        if engine is None:
            raise RuntimeError("Failed to build engine")
            
        # Save engine
        engine_path = 'models/trt_engines/deeplabv3_fp16.engine'
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(engine)
        
        logger.info(f"Created engine at {engine_path}")
        return engine_path
        
    except Exception as e:
        logger.error(f"Error creating engine: {str(e)}")
        raise

if __name__ == '__main__':
    create_engine()
