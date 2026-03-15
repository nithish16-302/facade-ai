import os
import io
import httpx
import asyncio
import base64 as b64
import struct
import zlib

from dotenv import load_dotenv

load_dotenv()

NEGATIVE_PROMPT = (
    "wrong colors, faded, desaturated, washed out, blurry, artifacts, "
    "different building, different structure, deformed architecture, "
    "painting, sketch, cartoon, unrealistic"
)


def extract_canny_edges_pillow(image_bytes: bytes) -> bytes:
    """
    Extracts Canny-like edges using Pillow only (no OpenCV).
    Works on Vercel serverless which doesn't have OpenCV.
    Uses Pillow's FIND_EDGES filter which is a Laplacian-based edge detector.
    Returns edge image as PNG bytes.
    """
    from PIL import Image, ImageFilter, ImageOps, ImageEnhance

    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale

    # Slight blur to reduce noise
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Find edges
    edges = img.filter(ImageFilter.FIND_EDGES)

    # Enhance contrast of edges
    edges = ImageEnhance.Contrast(edges).enhance(3.0)

    # Threshold to make crisp black/white edges
    edges = edges.point(lambda p: 255 if p > 30 else 0)

    # Convert to RGB (ControlNet expects 3-channel image)
    edges_rgb = edges.convert("RGB")

    buffer = io.BytesIO()
    edges_rgb.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


async def generate_sdxl_facade_async(base64_img_str: str, generation_prompt: str) -> str:
    """
    Two-step ControlNet-Canny pipeline:
    1. Extract edges from the original image using Pillow (preserves building structure)
    2. Run SDXL-ControlNet-Canny on Replicate — structure is locked, only colors change
    Falls back to standard SDXL img2img at moderate strength if ControlNet fails.
    """

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("Missing Replicate token.")
        return ""

    print("Initiating Replicate ControlNet pipeline...")
    try:
        if "," in base64_img_str:
            base64_data = base64_img_str.split(",")[1]
        else:
            base64_data = base64_img_str
        image_bytes = b64.b64decode(base64_data)

        # Step 1: Extract edges using Pillow
        print("Extracting structural edges for ControlNet...")
        canny_bytes = extract_canny_edges_pillow(image_bytes)
        canny_data_uri = "data:image/jpeg;base64," + b64.b64encode(canny_bytes).decode("utf-8")

        auth_headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=180.0) as client:
            # Step 2: Run SDXL-ControlNet-Canny
            print("Running SDXL-ControlNet (structure-locked generation)...")
            payload = {
                "version": "06d6fae3b75ab68a28cd2900afa6033166910dd0",
                "input": {
                    "image": canny_data_uri,
                    "prompt": generation_prompt,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "num_inference_steps": 40,
                    "guidance_scale": 8.0,
                    "controlnet_conditioning_scale": 0.85,
                    "scheduler": "DPMSolverMultistep",
                }
            }

            pred_resp = await client.post(
                "https://api.replicate.com/v1/predictions",
                headers={**auth_headers, "Content-Type": "application/json"},
                json=payload
            )

            if pred_resp.status_code == 429:
                return "Error: Replicate API Rate Limit (429). You likely have another generation in progress. Please wait a minute and try again."

            if pred_resp.status_code not in [200, 201]:
                print(f"ControlNet failed with {pred_resp.status_code}: {pred_resp.text[:200]}")
                # Don't fallback if it's a structural error (like 401/403/429)
                if pred_resp.status_code in [401, 403]:
                    return f"API Auth Error: {pred_resp.status_code}"
                return await _sdxl_fallback(image_bytes, generation_prompt, auth_headers, client)

            prediction = pred_resp.json()
            get_url = prediction["urls"]["get"]

            print("Polling for ControlNet generation completion...")
            while True:
                poll_resp = await client.get(get_url, headers=auth_headers)
                
                if poll_resp.status_code == 429:
                    print("  Polling rate limit (429), waiting longer...")
                    await asyncio.sleep(5)
                    continue

                poll_resp.raise_for_status()
                status_data = poll_resp.json()
                status = status_data["status"]
                print(f"  Status: {status}")

                if status == "succeeded":
                    output = status_data["output"]
                    if isinstance(output, list) and len(output) > 0:
                        print("ControlNet generation succeeded!")
                        return output[0]
                    return str(output)

                elif status in ["failed", "canceled"]:
                    error = status_data.get("error", "")
                    print(f"ControlNet failed: {error}. Falling back to SDXL img2img...")
                    return await _sdxl_fallback(image_bytes, generation_prompt, auth_headers, client)

                await asyncio.sleep(3)  # Increased to 3s to reduce poll frequency

    except Exception as e:
        print(f"Error in generate_sdxl_facade_async: {e}")
        return ""


async def _sdxl_fallback(image_bytes: bytes, prompt: str, auth_headers: dict, client: httpx.AsyncClient) -> str:
    """Fallback to SDXL img2img at 0.65 strength if ControlNet fails."""
    try:
        print("Running SDXL img2img fallback...")
        files = {"content": ("facade.jpg", image_bytes, "image/jpeg")}
        file_resp = await client.post(
            "https://api.replicate.com/v1/files",
            headers=auth_headers,
            files=files
        )
        file_resp.raise_for_status()
        hosted_url = file_resp.json()["urls"]["get"]

        payload = {
            "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            "input": {
                "image": hosted_url,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "prompt_strength": 0.65,
                "num_outputs": 1,
                "num_inference_steps": 35,
                "guidance_scale": 8.0,
                "refine": "expert_ensemble_refiner",
                "high_noise_frac": 0.8,
            }
        }
        resp = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers={**auth_headers, "Content-Type": "application/json"},
            json=payload
        )
        resp.raise_for_status()
        poll_url = resp.json()["urls"]["get"]
        while True:
            poll = await client.get(poll_url, headers=auth_headers)
            
            if poll.status_code == 429:
                print("  Fallback polling rate limit (429), waiting longer...")
                await asyncio.sleep(5)
                continue

            pd = poll.json()
            status = pd.get("status")
            print(f"  Fallback status: {status}")
            if status == "succeeded":
                output = pd["output"]
                return output[0] if isinstance(output, list) else str(output)
            elif status in ["failed", "canceled"]:
                print(f"Fallback also failed: {pd.get('error')}")
                return ""
            await asyncio.sleep(3)
    except Exception as e:
        print(f"Fallback error: {e}")
        return ""
