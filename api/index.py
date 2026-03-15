import sys
import os

# Ensure api/ directory is in path so `services` package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
from services.vision_engine import analyze_facade_async
from services.sdxl_engine import generate_sdxl_facade_async

app = FastAPI(
    title="Facade.ai Backend",
    description="API for handling architectural AI vision and generation workflows.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "Facade.ai API Core"}

@app.post("/api/v1/generate")
async def generate_facade(
    image: UploadFile = File(...),
    palette_id: str = Form(...),
    hex_color: str = Form(None)
):
    contents = await image.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    vision_logic = await analyze_facade_async(base64_image, palette_id, hex_color)
    prompt = vision_logic.get("image_generation_prompt", "")
    print(f"Sending prompt to SDXL: {prompt}")
    generated_image_url = await generate_sdxl_facade_async(base64_image, prompt)
    return {
        "status": "success",
        "vision_analysis": vision_logic,
        "generated_image_url": generated_image_url
    }
