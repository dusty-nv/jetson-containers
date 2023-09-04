#!/usr/bin/env python3
# cli chat client (gets installed to /opt/minigpt4.cpp/minigpt4)
import os
import sys
import time
import argparse
import minigpt4_library

from PIL import Image

parser = argparse.ArgumentParser(description='Test loading minigpt4')

parser.add_argument('model_path', help='Path to model file')
parser.add_argument('llm_model_path', help='Path to llm model file')

parser.add_argument('-p', '--prompt', action='append', nargs='*')
parser.add_argument('-i', '--image', default='/data/images/hoover.jpg', help="Path to the image to test")
parser.add_argument('-r', '--runs', type=int, default=3, help="Number of inferencing runs to do (for timing)")

parser.add_argument('--max-new-tokens', type=int, default=64, help="Limit the length of LLM output")

args = parser.parse_args()

if not args.prompt:
    args.prompt = [
        "What does the sign in the image say?",
        "What kind of environment is it in?"
    ]
    
print(args)

minigpt4_chatbot = minigpt4_library.MiniGPT4ChatBot(args.model_path, args.llm_model_path, verbosity=minigpt4_library.Verbosity.DEBUG) # SILENT, ERR, INFO, DEBUG

print(f"-- opening {args.image}")
image = Image.open(args.image).convert('RGB')

for run in range(args.runs):
    time_begin=time.perf_counter()
    minigpt4_chatbot.upload_image(image)
    time_encoder=time.perf_counter() - time_begin
    print(f"{os.path.basename(args.model_path)} encoder:  {time_encoder:.3f} seconds")
    
    for prompt in args.prompt:
        num_tokens=0
        time_begin=time.perf_counter()
        for token in minigpt4_chatbot.generate(prompt, limit=args.max_new_tokens):
            print(token, end='')
            sys.stdout.flush()
            num_tokens += 1
        print('\n')
        time_elapsed=time.perf_counter() - time_begin
        print(f"{os.path.basename(args.llm_model_path)}:  {time_elapsed:.2f} seconds, {num_tokens} tokens, {num_tokens / time_elapsed:.2f} tokens/sec")
        
    minigpt4_chatbot.reset_chat()