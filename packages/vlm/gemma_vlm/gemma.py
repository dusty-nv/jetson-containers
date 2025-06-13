#!/usr/bin/env python3
import os
import PIL
import time
import torch
import requests
import argparse

from transformers import AutoProcessor, AutoModelForImageTextToText

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="google/medgemma-4b-it")
parser.add_argument('--token', type=str, default=os.environ.get('HUGGINGFACE_TOKEN'))
parser.add_argument('--image', type=str, default="https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png")
parser.add_argument('--prompt', type=str, default="Describe this X-ray")
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--interactive', action='store_true')
parser.add_argument('--max-new-tokens', type=int, default=200)

args = parser.parse_args()

# load model
model = AutoModelForImageTextToText.from_pretrained(
    args.model, 
    token=args.token,
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained(args.model, token=args.token)

print('Processor:\n', processor)
print('Model:\n', model)

# image Q/A
image, prompt = args.image, args.prompt

while True:
    if prompt is None and args.interactive:
        print("\nEnter prompt or image path/URL:\n")
        entry = input('>> ').strip()
        
        if any([entry.lower().endswith(x) for x in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']]):
            image = entry
            prompt = None
            continue
        else:
            prompt = entry
    
    if prompt is None:
        break
           
    if isinstance(image, str):
        print(f'Loading {image}')
        image = PIL.Image.open(
            requests.get(image, headers={"User-Agent": "example"}, stream=True).raw
            if image.startswith('http') else image
        )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    for i in range(args.runs):
        print(f"Processing prompt:  {prompt}")
        time_begin = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens,
                do_sample=False
            )
        response = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        time_end = time.perf_counter()
        time_elapsed = time_end - time_begin
        print('\n' + response)
        print(f"\nRun {i} - total {time_elapsed:.4f}s ({len(output[0])} tokens, {len(output[0])/time_elapsed:.2f} tokens/sec)\n")      
    
    prompt = None