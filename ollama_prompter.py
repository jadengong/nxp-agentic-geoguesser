import ollama
import io
import base64

from typer import prompt

# assuming current_image is your PIL Image

prompt = "You are a geographic expert tasked with identifying the location in this image?" \
"reason step by step and provide a detailed answer about what region of the world this photo was likely taken in. First consider, climate clues, vegatation, architeture, famous landmarks, textual language( if present), technological features and anything else. "\
"Format your responce as :" \
"1. Clue analysis: \n" \
"2. Most Likely location: \n"   
"3. Confidence level: \n" \

def run_ollama(current_image):
    buffer = io.BytesIO()
    current_image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    response = ollama.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_bytes]  # raw bytes work too
            }
        ]
    )

    print(response["message"]["content"])