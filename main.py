import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

import image_detection

ALLOWED_CONTENT_TYPES = {"image/jpeg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg"}

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/detect")
async def detect_image(
    file: UploadFile = File(...),
    candidate_labels: list[str] | None = None,
    score_threshold: float = 0.15,
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG images are allowed. Content-Type must be image/jpeg.",
        )
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .jpg or .jpeg files are allowed.",
        )

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    try:
        labels = image_detection.get_image_labels(
            temp_path,
            candidate_labels=candidate_labels,
            score_threshold=score_threshold,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"labels": labels}
