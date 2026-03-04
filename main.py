import asyncio
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import ollama
import io

app = FastAPI()
templates = Jinja2Templates(directory="templates")

prompt = (
    "You are a geographic expert tasked with identifying the location in this image? "
    "reason step by step and provide a detailed answer about what region of the world this photo was likely taken in. "
    "First consider, climate clues, vegetation, architecture, famous landmarks, textual language (if present), "
    "technological features and anything else. Give short concise responces. Provide an exact location like one country or one city for exact location. "
    "Format your response as:\n"
    "1. Clue analysis:\n"
    "2. Most Likely location:\n"
    "3. Confidence level:\n"
)

@app.post("/identify_location")
async def identify_location(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    response = await asyncio.to_thread(
        ollama.chat,
        model="llava",
        messages=[{"role": "user", "content": prompt, "images": [image_bytes]}]
    )

    return {"result": response["message"]["content"]}

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})