#!/usr/bin/env python3
import argparse
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration # Changed from PaliGemma
import os

# Default model ID, updated to Gemma3
default_model_id = os.environ.get("GEMMA_VLM_MODEL_ID", "google/gemma-3-4b-it")

# Argument parser
parser = argparse.ArgumentParser(description="Test script for Gemma3 VLM models.")
parser.add_argument("--model_id", type=str, default=default_model_id,
                    help="Hugging Face model ID for the Gemma3 model.")
parser.add_argument("--image_url", type=str, 
                    default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg", 
                    help="URL of the image to process.")
parser.add_argument("--prompt_text", type=str, default="Describe this image in detail.", 
                    help="Text part of the prompt for the image.")
parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.",
                    help="System prompt for the chat model.")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the model on ('cuda' or 'cpu').")
parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
                    help="Data type for model parameters (e.g., 'bfloat16', 'float16', 'float32').")
parser.add_argument("--max_new_tokens", type=int, default=100,
                    help="Maximum number of new tokens to generate.")

def main():
    args = parser.parse_args()

    print(f"Using model: {args.model_id}")
    print(f"Using device: {args.device}")
    print(f"Using dtype: {args.dtype}")

    # Determine torch_dtype
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    try:
        # Load the processor and model
        print(f"Loading processor for {args.model_id}...")
        # Gemma3 uses AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_id)
        print(f"Loading model {args.model_id} to {args.device} with dtype {args.dtype}...")
        # Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            device_map=args.device, # Using device_map for auto placement based on arg
        ).eval()
        print("Model and processor loaded successfully.")

        # Load image from URL
        print(f"Downloading image from: {args.image_url}")
        try:
            # The new script expects a PIL image object for the chat template
            pil_image = Image.open(requests.get(args.image_url, stream=True).raw).convert('RGB')
            print("Image loaded successfully.")
        except Exception as e:
            print(f"Failed to load image from URL: {args.image_url}")
            print(f"Error: {e}")
            return

        # Prepare chat messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": args.system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image}, # Pass the PIL image
                    {"type": "text", "text": args.prompt_text}
                ]
            }
        ]
        
        print(f"Processing inputs with prompt: '{args.prompt_text}'")
        # Apply chat template
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, # Important for Gemma3
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device) # Ensure inputs are on the same device as the model
        
        print("Inputs processed.")
        
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        print("Generating output...")
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            # Slice to get only the generated part
            generated_ids = generation[0][input_len:]
        print("Output generated.")

        # Decode and print the output
        decoded_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        print("\n--- Generated Text ---")
        print(decoded_text)
        print("----------------------")

        if decoded_text:
            print("\nTest PASSED: Model generated a response.")
        else:
            print("\nTest FAILED: Model did not generate a response or response was empty.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nTest FAILED due to an exception.")

if __name__ == "__main__":
    main()
