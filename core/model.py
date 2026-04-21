"""
Core model loading and initialization logic.
"""
from typing import Tuple
from transformers import BlipProcessor, BlipForConditionalGeneration

from config.settings import MODEL_NAME, DEVICE

def load_model() -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """
    Loads the BLIP processor and model based on the configuration settings.
    Moves the model to the configured device and sets it to evaluation mode.

    Returns:
        Tuple[BlipProcessor, BlipForConditionalGeneration]: A tuple containing
            the initialized processor and model.
    """
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    model.to(DEVICE)
    model.eval()
    
    return processor, model
