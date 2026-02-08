import torch
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from diffusers import DiffusionPipeline
    # Checking if the specific classes exist or if they are meant to be loaded via DiffusionPipeline
    import diffusers
    logger.info(f"Diffusers version: {diffusers.__version__}")
    
    # Based on the pipeline.py I saw earlier, it expects these specific classes
    try:
        from diffusers import GlmImagePipeline, LTX2ImageToVideoPipeline
        logger.info("Successfully imported GlmImagePipeline and LTX2ImageToVideoPipeline")
    except ImportError:
        logger.warning("Specific pipelines not found in diffusers. Checking available pipelines...")
        # If they are custom/community pipelines, we might need to load via DiffusionPipeline.from_pretrained
except ImportError as e:
    logger.error(f"ImportError: {e}")
    sys.exit(1)

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")