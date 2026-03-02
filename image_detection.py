# Using CLIP via HuggingFace
from transformers import pipeline 
from PIL import Image

image_path = "/Users/matthewtran/nxp-agentic-geoguesser/nxp-agentic-geoguesser/images/taj-mahal.jpg"

classifier = pipeline("zero-shot-image-classification",
                          model="openai/clip-vit-base-patch32")

def get_image_labels(image_path):
    image = Image.open(image_path)
    candidates = ["a famous landmark in India", "a historical monument in Europe","a Middle Eastern mosque", "a Southeast Asian temple", "an American capital building"] 
    results = classifier(image, candidate_labels = candidates)
    return [r['label'] for r in results if r['score'] > 0.15]