"""
Settings and configuration constants for the image captioning app.
"""
import torch

MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_NEW_TOKENS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_MAX_SIZE = (1024, 1024)
