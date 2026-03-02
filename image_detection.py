# Using CLIP via HuggingFace
from transformers import pipeline 
from PIL import Image

def get_image_lavels(image_path):
    classifier = pipeline("zero-shot-image-classification",
                          model="openai/clip-vit-base-patch32")
    image = Image.open(image_path)
    