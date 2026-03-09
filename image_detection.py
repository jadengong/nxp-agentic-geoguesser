# Using CLIP via HuggingFace for zero-shot image classification
from transformers import pipeline
from PIL import Image

classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
)

# Default labels used when none are provided (e.g. load your own from config/API).
DEFAULT_CANDIDATE_LABELS = [
    "a famous landmark in India",
    "a historical monument in Europe",
    "a Middle Eastern mosque",
    "a Southeast Asian temple",
    "an American capital building",
]


def get_image_labels(
    image_path,
    candidate_labels=None,
    score_threshold=0.15,
):
    """
    Run zero-shot classification on an image.

    Args:
        image_path: Path to the image file.
        candidate_labels: List of text labels to score. If None, uses DEFAULT_CANDIDATE_LABELS.
        score_threshold: Minimum score (0–1) for a label to be included.

    Returns:
        List of label strings that scored above the threshold.
    """
    if candidate_labels is None:
        candidate_labels = DEFAULT_CANDIDATE_LABELS
    image = Image.open(image_path)
    results = classifier(image, candidate_labels=candidate_labels)
    return [r["label"] for r in results if r["score"] > score_threshold]