# Using CLIP via HuggingFace for zero-shot image classification
import json
import os
import time
import urllib.request

from transformers import pipeline
from PIL import Image

classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
)

# Fallback when LABELS_API_URL is not set or the API fails. Keep a small set so the app always works.
DEFAULT_CANDIDATE_LABELS = [
    "a modern city skyline",
    "a European city",
    "a Middle Eastern city or mosque",
    "a beach or coastal scene",
    "mountains or countryside",
]

# Cache for labels from external API: (timestamp, list). Refreshed when LABELS_CACHE_SECONDS has passed.
_labels_cache: tuple[float, list[str]] | None = None
LABELS_CACHE_SECONDS = 300  # 5 minutes


def get_labels_from_api() -> list[str] | None:
    """
    Fetch candidate labels from the external API configured in LABELS_API_URL.
    Expects JSON: either a list of strings or an object with a "labels" key.
    Returns None on failure or if LABELS_API_URL is not set.
    """
    url = os.environ.get("LABELS_API_URL", "").strip()
    if not url:
        return None
    global _labels_cache
    now = time.monotonic()
    if _labels_cache is not None and (now - _labels_cache[0]) < LABELS_CACHE_SECONDS:
        return _labels_cache[1]
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list) and data and isinstance(data[0], str):
            labels = data
        elif isinstance(data, dict) and "labels" in data and isinstance(data["labels"], list):
            labels = [str(x) for x in data["labels"]]
        else:
            return None
        if labels:
            _labels_cache = (now, labels)
            return labels
    except Exception:
        pass
    return _labels_cache[1] if _labels_cache else None


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
        candidate_labels = get_labels_from_api() or DEFAULT_CANDIDATE_LABELS
    image = Image.open(image_path)
    results = classifier(image, candidate_labels=candidate_labels)
    return [r["label"] for r in results if r["score"] > score_threshold]
