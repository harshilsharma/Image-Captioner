"""
Entry point for the Image Captioning Gradio web application.
"""
import gradio as gr
from PIL import Image

from core.model import load_model
from core.inference import generate_caption

# Load the processor and model once at module initialization
processor, model = load_model()

def predict(image: Image.Image) -> str:
    """
    Gradio integration function that triggers caption generation securely.

    Args:
        image (PIL.Image.Image): The uploaded PIL Image.

    Returns:
        str: The generated plain text caption.
    """
    if image is None:
        return ""
    
    return generate_caption(image, processor, model)

# Create the standard Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Caption Generator",
    description="Upload an image to generate a natural language caption using BLIP."
)

if __name__ == "__main__":
    # Launch strictly locally adhering to requirements
    interface.launch(share=False, debug=False)
