from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
from typing import Optional
from services.vision_engine import analyze_facade_async
from services.sdxl_engine import generate_sdxl_facade_async

app = FastAPI(
    title="Facade.ai Backend",
    description="API for handling architectural AI vision and generation workflows.",
    version="1.0.0"
)

# Configure CORS for Next.js frontend
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
    """
    Core endpoint that:
    1. Receives the original photo and requested palette.
    2. Sends it to the Vision LLM (GPT-4o) to frame the generative prompt.
    3. Runs an asynchronous API call to Replicate SDXL to render the final pixels.
    """
    
    # 1. Read and encode image memory stream
    contents = await image.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    
    # 2. Vision Engine Logic
    # Get the highly structured JSON prompt from OpenAI
    vision_logic = await analyze_facade_async(base64_image, palette_id, hex_color)
    
    # 3. Apply Replicate SDXL Generative Layer
    prompt = vision_logic.get("image_generation_prompt", "")
    print(f"Sending prompt to SDXL: {prompt}")
    
    generated_image_url = await generate_sdxl_facade_async(base64_image, prompt)
    
    # Return exactly what the Next.js frontend needs to display
    return {
        "status": "success",
        "vision_analysis": vision_logic,
        "generated_image_url": generated_image_url
    }
