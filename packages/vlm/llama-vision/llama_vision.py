#!/usr/bin/env python3
import os
import PIL
import time
import torch
import requests
import argparse

from transformers import MllamaForConditionalGeneration, AutoProcessor

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-11B-Vision")
parser.add_argument('--token', type=str, default=os.environ.get('HUGGINGFACE_TOKEN'))
parser.add_argument('--image', type=str, default="https://llava-vl.github.io/static/images/view.jpg")
parser.add_argument('--prompt', type=str, default="If I had to write a haiku for this one")
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--interactive', action='store_true')
parser.add_argument('--max-new-tokens', type=int, default=64)

args = parser.parse_args()

# load model
model = MllamaForConditionalGeneration.from_pretrained(
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
            requests.get(image, stream=True).raw
            if image.startswith('http') else image
        )

    inputs = f"<|image|><|begin_of_text|>{prompt}"
    inputs = processor(text=inputs, images=image, return_tensors="pt").to(model.device)

    for i in range(args.runs):
        print(f"Processing prompt:  {prompt}")
        time_begin = time.perf_counter()
        output = model.generate(
            **inputs, 
            do_sample=False, 
            max_new_tokens=args.max_new_tokens
        )
        response = processor.decode(output[0], skip_special_tokens=True)
        time_end = time.perf_counter()
        time_elapsed = time_end-time_begin
        print('\n' + response)
        print(f"\nRun {i} - total {time_elapsed:.4f}s ({len(output[0])} tokens, {len(output[0])/time_elapsed:.2f} tokens/sec)\n")      
    
    prompt = None
  
