from fastapi import FastAPI
import image_detection
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
