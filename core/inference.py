"""
Core inference logic for generating image captions.
"""
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from config.settings import DEVICE, MAX_NEW_TOKENS
from utils.image_utils import preprocess_image

def generate_caption(
    image: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration
) -> str:
    """
    Generates a textual caption for a given image using the BLIP model.

    Args:
        image (PIL.Image.Image): The raw input image.
        processor (BlipProcessor): The HuggingFace processor for BLIP.
        model (BlipForConditionalGeneration): The HuggingFace BLIP model.

    Returns:
        str: The generated caption as a plain stripped string.
    """
    # Preprocess the image by ensuring RGB and enforcing maximum bounds
    processed_image = preprocess_image(image)

    # Create model inputs and transfer them to the configured device
    inputs = processor(images=processed_image, return_tensors="pt").to(DEVICE)
    
    # Generate tokenized output in evaluation block without gradient tracking
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        
    # Safely decode ignoring special tokens
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption.strip()
