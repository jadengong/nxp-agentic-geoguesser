import asyncio
import io
import json
import os
import tempfile

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import ollama

import image_detection

app = FastAPI()
templates = Jinja2Templates(directory="templates")

OLLAMA_PROMPT = (
    "You are a geographic expert. You have been given visual labels that describe what was detected in this image. "
    "Use those labels and what you see to pick ONE exact location. "
    "You must give a specific country and city—your single best guess. Do not say 'might be', 'could be', or 'not identifiable'. "
    "Commit to one place (e.g. Spain, Seville or USA, Los Angeles or Turkey, Istanbul).\n\n"
    "Format your response exactly as:\n"
    "1. Clue analysis: [2-4 short bullet points]\n"
    "2. Most likely location: [Country, City]\n"
    "3. Confidence level: [Low/Medium/High]\n\n"
    "Example for a Mediterranean campus with palm trees and a bell tower: '2. Most likely location: Spain, Valencia' or 'USA, Santa Barbara'."
)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _identify_stream(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    yield _sse({"stage": "clip"})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        clip_labels = await asyncio.to_thread(
            image_detection.get_image_labels, tmp_path
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    labels_text = ", ".join(clip_labels) if clip_labels else "No strong labels"
    user_content = (
        f"Visual labels detected in this image: {labels_text}\n\n"
        f"{OLLAMA_PROMPT}"
    )
    messages = [
        {"role": "user", "content": user_content, "images": [image_bytes]}
    ]

    yield _sse({"stage": "ollama", "labels": clip_labels})

    queue = asyncio.Queue()

    def stream_ollama():
        for chunk in ollama.chat(
            model="llava", messages=messages, stream=True
        ):
            part = (chunk.get("message") or {}).get("content") or ""
            queue.put_nowait(part)
        queue.put_nowait(None)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, stream_ollama)

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield _sse({"chunk": chunk})
    yield _sse({"stage": "done"})


@app.post("/identify_location")
async def identify_location(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # 1) CLIP: get labels from the image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        clip_labels = await asyncio.to_thread(
            image_detection.get_image_labels, tmp_path
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    labels_text = ", ".join(clip_labels) if clip_labels else "No strong labels"
    user_content = (
        f"Visual labels detected in this image: {labels_text}\n\n"
        f"{OLLAMA_PROMPT}"
    )

    # 2) Ollama: use labels + image to deduce location
    response = await asyncio.to_thread(
        ollama.chat,
        model="llava",
        messages=[
            {
                "role": "user",
                "content": user_content,
                "images": [image_bytes],
            }
        ],
    )

    return {"result": response["message"]["content"], "labels": clip_labels}


@app.post("/identify_location_stream")
async def identify_location_stream(file: UploadFile = File(...)):
    return StreamingResponse(
        _identify_stream(file),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})