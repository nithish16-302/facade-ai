import os
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def generate_sdxl_facade_async(base64_img_str: str, generation_prompt: str) -> str:
    """
    Sends the original base64 image and the GPT-4o-mini generation prompt 
    to Replicate's SDXL image-to-image API.
    1. Uploads image to Replicate's /v1/files endpoint via multipart
    2. Passes the hosted URL + generation prompt to SDXL
    3. Polls until generation succeeds and returns the output image URL
    """
    
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("Missing Replicate token.")
        return ""

    print("Initiating Replicate API Prediction...")
    try:
        # 1. Decode base64 to raw bytes
        if "," in base64_img_str:
            base64_data = base64_img_str.split(",")[1]
        else:
            base64_data = base64_img_str
            
        import base64 as b64
        image_bytes = b64.b64decode(base64_data)
        
        auth_headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Step 1: Upload image to Replicate's file hosting via multipart
            print("Uploading image to Replicate file hosting...")
            files = {"content": ("facade.jpg", image_bytes, "image/jpeg")}
            file_resp = await client.post(
                "https://api.replicate.com/v1/files",
                headers=auth_headers,
                files=files
            )
            file_resp.raise_for_status()
            file_data = file_resp.json()
            
            # The hosted URL is at urls.get
            hosted_url = file_data["urls"]["get"]
            print(f"Image hosted at: {hosted_url}")
            
            # Step 2: Create SDXL image-to-image prediction
            print("Sending prompt to Replicate SDXL...")
            payload = {
                "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                "input": {
                    "image": hosted_url,
                    "prompt": generation_prompt,
                    "prompt_strength": 0.60,
                    "num_outputs": 1,
                    "scheduler": "K_EULER",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "refine": "expert_ensemble_refiner",
                    "high_noise_frac": 0.8
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
            
            # Step 3: Poll until complete
            print("Polling Replicate for generation completion...")
            while True:
                poll_resp = await client.get(get_url, headers=auth_headers)
                poll_resp.raise_for_status()
                status_data = poll_resp.json()
                
                status = status_data["status"]
                print(f"  Generation status: {status}")
                
                if status == "succeeded":
                    output = status_data["output"]
                    if isinstance(output, list) and len(output) > 0:
                        print("Successfully received generated image URL from Replicate.")
                        return output[0]
                    return str(output)
                    
                elif status in ["failed", "canceled"]:
                    print(f"Replicate Generation Failed: {status_data.get('error')}")
                    return ""
                    
                await asyncio.sleep(2)
                
    except Exception as e:
        print(f"Replicate HTTP Error: {e}")
        return ""
