import base64
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize client safely to allow the backend to run in dev environments without a key
try:
    client = AsyncOpenAI()
    has_api_key = True
except Exception:
    client = None
    has_api_key = False

class FacadeAnalysis(BaseModel):
    masking_instructions: str = Field(
        description="Detailed description of the structural boundaries for segmentation (Walls, Trim, Roof). What should be masked?"
    )
    locked_elements: str = Field(
        description="Elements that must NOT be altered (Windows, Doors, Trees, structural mullions)."
    )
    image_generation_prompt: str = Field(
        description="The final prompt to send to SDXL/ControlNet. Must incorporate the chosen palette, lighting, textures, and camera details."
    )
    architectural_notes: str = Field(
        description="Brief note on the architectural style and any challenges like occlusions."
    )

async def analyze_facade_async(base64_image: str, palette_id: str, hex_color: str = None) -> dict:
    """
    Sends the architectural photo to GPT-4o-mini to return the structured generation prompt.
    """
    
    # Map palette IDs to detailed architectural descriptions
    palette_map = {
        "nordic": "Nordic Noir (Hyper-Minimalist Dark): Primary Vantablack/Matte Onyx. Trim Anodized Aluminum. Accents Soft warm 3000K LED glow.",
        "biophilic": "Biophilic Greens: Primary Sage Shadows (Muted Moss Green). Trim Raw Corten Steel. Accents Natural Matte Teak Wood.",
        "brutalist": "Eco Brutalism: Primary Board-formed Concrete Gray. Trim Charcoal Black (Matte). Accents Slatted natural oak.",
        "monolith": "Desert Monolith: Primary Plastered Terracotta/Adobe Sand. Trim Muted Bronze. Accents Limestone White.",
        "americana": "New Americana: Primary Deep Navy Blue Matte. Trim Alabaster Crisp Off-White. Accents Reclaimed Red Brick.",
        "haveli": "Haveli Sandstone (Rajasthani Heritage): Primary warm golden Dholpur sandstone. Trim ornate hand-carved jali latticework in warm ochre. Accents deep maroon and saffron decorative motifs. Arched doorways and jarokha window overhangs. Architectural photography, rich warm daylight.",
        "kerala": "Kerala Verdant (South Indian Vernacular): Primary deep laterite red walls. Roof steep Mangalore clay tiles in rustic terracotta. Trim polished tropical teak wood. Accents lush green plantation backdrop, whitewashed plinth. Humid tropical light, traditional nalukettu courtyard style.",
        "mughal": "Mughal Marble (Imperial Indo-Islamic): Primary gleaming Makrana white marble facade. Trim inlaid pietra dura floral patterns in lapis and jade. Accents cusped arches with fine chhatri pavilions. Gold and ivory filigree detailing. Architectural photography, golden-hour lighting on white marble.",
    }
    
    # Handle custom hex color
    if palette_id == "custom" and hex_color:
        # Convert hex to RGB for richer prompt context
        hex_clean = hex_color.lstrip('#')
        r, g, b = int(hex_clean[0:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)
        palette_details = (
            f"Custom User Color: Paint the main exterior walls with the exact color hex {hex_color} "
            f"(RGB {r}, {g}, {b}). Preserve and match this color precisely. "
            f"Apply as a smooth matte or satin exterior paint finish. "
            f"Architectural photography, natural daylight to reveal the true color."
        )
    else:
        palette_details = palette_map.get(palette_id, palette_map["nordic"])

    
    system_prompt = f"""
    You are an expert Senior Architectural Prompt Engineer for 'Facade.ai'.
    Your job is to analyze photos of buildings and generate precise masking instructions
    and generation prompts so a downstream image diffusion model can repaint the house without altering its structure.
    
    The user has selected the following 'Modern 2026' palette:
    {palette_details}
    
    Analyze the image and fill out the required JSON fields. Be highly specific about 
    architectural terms.
    """

    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this building and formulate the generation prompt."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            response_format=FacadeAnalysis,
            max_tokens=800
        )
        return response.choices[0].message.parsed.model_dump()
        
    except Exception as e:
        print(f"Vision API Error: {e}")
        # Return a mock output for demonstration if API fails or KEY is missing
        return {
            "masking_instructions": "MOCK: Mask main walls and trim.",
            "locked_elements": "MOCK: Lock windows, doors, and surrounding foliage.",
            "image_generation_prompt": f"MOCK: {palette_details}. Architectural photography, highly detailed, 8k.",
            "architectural_notes": f"MOCK: Failed to reach actual OpenAI API. Ensure OPENAI_API_KEY is set in backend. Selected palette was {palette_id}."
        }
