import cv2
import numpy as np
import base64

def apply_tint(base64_img_str: str, palette_name: str) -> str:
    """
    Takes a base64 image and programmatically applies a color space 
    transformation (tint) to mock the Generative AI coloring the house.
    """
    
    # 1. Decode base64 to OpenCV Matrix
    try:
        nparr = np.frombuffer(base64.b64decode(base64_img_str), np.uint8)
        if len(nparr) == 0:
            print("Empty buffer received, skipping tint.")
            return base64_img_str
            
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return base64_img_str

    if img is None:
        print("cv2.imdecode returned None, skipping tint.")
        return base64_img_str # Return original if decoding fails

    # Define color tints (BGR format for OpenCV)
    # Scale from 0 to 255. High values means more tint in that channel
    palette_colors = {
        "nordic": (40, 40, 40),       # Dark gray addition
        "biophilic": (120, 150, 120), # Sage Green addition
        "brutalist": (160, 160, 160), # Flat Light Gray addition
        "monolith": (140, 180, 210),  # Terracotta / warm sand
        "americana": (200, 100, 80)   # Deep Navy Blue
    }

    # Default to slight gray if palette not found
    target_tint = palette_colors.get(palette_name, (200, 200, 200))

    # 2. Convert to HSV to safely adjust saturation and brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create a solid color image of the same size
    tint_layer = np.full(img.shape, target_tint, dtype=np.uint8)
    
    # 3. Blend the images 
    # Use cv2.addWeighted to overlay the tint at 35% opacity
    blended = cv2.addWeighted(img, 0.65, tint_layer, 0.35, 0)
    
    # Optional: Slightly dim the Nordic theme
    if palette_name == "nordic":
       hsv_blend = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)
       hsv_blend[:,:,2] = np.clip(hsv_blend[:,:,2] * 0.7, 0, 255).astype(np.uint8)
       blended = cv2.cvtColor(hsv_blend, cv2.COLOR_HSV2BGR)

    # Optional: Slightly boost saturation for Americana
    if palette_name == "americana":
       hsv_blend = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)
       hsv_blend[:,:,1] = np.clip(hsv_blend[:,:,1] * 1.3, 0, 255).astype(np.uint8)
       blended = cv2.cvtColor(hsv_blend, cv2.COLOR_HSV2BGR)

    # 4. Encode back to base64
    _, buffer = cv2.imencode('.jpg', blended)
    base64_output = base64.b64encode(buffer).decode('utf-8')

    return base64_output
