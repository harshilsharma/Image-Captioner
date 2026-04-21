"""
Image processing utilities for the image captioning pipeline.
"""
from PIL import Image

from config.settings import IMAGE_MAX_SIZE

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Converts a PIL Image to RGB format and resizes it to fit within IMAGE_MAX_SIZE
    while preserving the aspect ratio.
    
    Args:
        image (PIL.Image.Image): The input image.
        
    Returns:
        PIL.Image.Image: The preprocessed image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Using Resampling.LANCZOS which is standard starting with Pillow 9.0
    image.thumbnail(IMAGE_MAX_SIZE, resample=Image.Resampling.LANCZOS)
    
    return image
