#!/usr/bin/env python3
import os
import urllib.request
from stable_diffusion_cpp import StableDiffusion

MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "v1-5-pruned-emaonly.safetensors")
MODEL_URL = "https://huggingface.co/sd-legacy/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"

def callback(step: int, steps: int, time: float):
    print(f"Completed step: {step} of {steps}")

# Ensure models folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading Stable Diffusion v1.5 model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")

# Initialize the Stable Diffusion engine
stable_diffusion = StableDiffusion(
    model_path=MODEL_PATH,
    # wtype="default",  # Optional: weight type (e.g. "q8_0", "f16")
)

# Generate an image
output = stable_diffusion.generate_image(
    prompt="a lovely cat",
    width=512,
    height=512,
    progress_callback=callback,
    # seed=1337,  # Uncomment to fix the random seed
)

# Save output image
output[0].save("output.png")
print("Image saved as output.png")

# Display metadata
print("Image info:")
print(output[0].info)
