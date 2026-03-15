import os
import httpx
import asyncio
import base64 as b64
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

NEGATIVE_PROMPT = (
    "wrong colors, faded, desaturated, washed out, blurry, artifacts, "
    "different building, different structure, deformed architecture, "
    "painting, sketch, cartoon, unrealistic"
)

def extract_canny_edges(image_bytes: bytes) -> bytes:
    """
    Extracts Canny edges from the input image bytes using OpenCV.
    Returns the edge image as JPEG bytes.
    This is used as the ControlNet conditioning signal to preserve architecture.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Canny edge detection with tuned thresholds for architecture
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Dilate slightly so thin lines carry through
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Convert back to 3-channel image (ControlNet expects RGB)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    _, buffer = cv2.imencode('.jpg', edges_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()


async def generate_sdxl_facade_async(base64_img_str: str, generation_prompt: str) -> str:
    """
    Two-step pipeline:
    1. Extract Canny edges from the original image using OpenCV
    2. Run SDXL-ControlNet-Canny on Replicate — this locks the building structure
       via the edge map while applying the color/style from the prompt.
    """

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("Missing Replicate token.")
        return ""

    print("Initiating Replicate ControlNet pipeline...")
    try:
        # Decode original image
        if "," in base64_img_str:
            base64_data = base64_img_str.split(",")[1]
        else:
            base64_data = base64_img_str
        image_bytes = b64.b64decode(base64_data)

        # Step 1: Extract Canny edges (structural skeleton of the building)
        print("Extracting Canny edges for ControlNet conditioning...")
        canny_bytes = extract_canny_edges(image_bytes)
        canny_b64 = b64.b64encode(canny_bytes).decode('utf-8')
        canny_data_uri = f"data:image/jpeg;base64,{canny_b64}"

        auth_headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=180.0) as client:
            # Step 2: Upload original image for img2img reference
            print("Uploading original image to Replicate...")
            files = {"content": ("original.jpg", image_bytes, "image/jpeg")}
            file_resp = await client.post(
                "https://api.replicate.com/v1/files",
                headers=auth_headers,
                files=files
            )
            file_resp.raise_for_status()
            original_url = file_resp.json()["urls"]["get"]
            print(f"Original hosted at: {original_url}")

            # Step 3: Run SDXL-ControlNet-Canny
            # The canny edges lock the building silhouette/structure
            # The prompt drives the color and material transformation
            print("Running SDXL-ControlNet (structure-locked generation)...")
            payload = {
                "version": "06d6fae3b75ab68a28cd2900afa6033166910dd0",
                "input": {
                    "image": canny_data_uri,           # Canny edges = structural constraint
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
            pred_resp.raise_for_status()
            prediction = pred_resp.json()
            get_url = prediction["urls"]["get"]

            # Step 4: Poll for completion
            print("Polling for ControlNet generation completion...")
            while True:
                poll_resp = await client.get(get_url, headers=auth_headers)
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
                    print(f"ControlNet failed: {error}")
                    # Fallback to plain SDXL if ControlNet fails
                    print("Falling back to SDXL img2img...")
                    return await _sdxl_fallback(original_url, generation_prompt, auth_headers, client)

                await asyncio.sleep(2)

    except Exception as e:
        print(f"Error in generate_sdxl_facade_async: {e}")
        return ""


async def _sdxl_fallback(image_url: str, prompt: str, auth_headers: dict, client: httpx.AsyncClient) -> str:
    """Fallback to standard SDXL img2img at moderate strength if ControlNet fails."""
    payload = {
        "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "input": {
            "image": image_url,
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
    pred = resp.json()
    poll_url = pred["urls"]["get"]
    while True:
        poll = await client.get(poll_url, headers=auth_headers)
        pd = poll.json()
        status = pd.get("status")
        if status == "succeeded":
            output = pd["output"]
            return output[0] if isinstance(output, list) else str(output)
        elif status in ["failed", "canceled"]:
            return ""
        await asyncio.sleep(2)
