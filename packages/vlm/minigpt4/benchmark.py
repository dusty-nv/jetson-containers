#!/usr/bin/env python3
import os
import sys
import time
import datetime
import resource
import argparse
import socket

import minigpt4_library

from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('model_path', help='Path to model file')
parser.add_argument('llm_model_path', help='Path to llm model file')

parser.add_argument('-p', '--prompt', action='append', nargs='*')
parser.add_argument('-i', '--image', default='/data/images/hoover.jpg', help="Path to the image to test")
parser.add_argument('-r', '--runs', type=int, default=2, help="Number of inferencing runs to do (for timing)")
parser.add_argument('-w', '--warmup', type=int, default=1, help='the number of warmup iterations')
parser.add_argument('-s', '--save', type=str, default='', help='CSV file to save benchmarking results to')
   
parser.add_argument('--max-new-tokens', type=int, default=64, help="Limit the length of LLM output")

args = parser.parse_args()

if not args.prompt:
    args.prompt = [
        "What does the sign in the image say?",
        "How far is the exit?",
        "What kind of environment is it in?",
        "Does it look like it's going to rain?",
    ]
else:
    args.prompt = [x[0] for x in args.prompt]
    
print(args)

def get_max_rss():  # peak memory usage in MB (max RSS - https://stackoverflow.com/a/7669482)
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  
    
minigpt4_chatbot = minigpt4_library.MiniGPT4ChatBot(args.model_path, args.llm_model_path, verbosity=minigpt4_library.Verbosity.DEBUG) # SILENT, ERR, INFO, DEBUG

model_name=f"{os.path.basename(args.model_path)}+{os.path.basename(args.llm_model_path)}"

print(f"-- opening {args.image}")
image = Image.open(args.image).convert('RGB')

avg_encoder=0
avg_latency=0
avg_tokens_sec=0

for run in range(args.runs + args.warmup):
    time_begin=time.perf_counter()
    minigpt4_chatbot.upload_image(image)
    time_encoder=time.perf_counter() - time_begin
    
    print(f"{model_name} encoder:  {time_encoder:.3f} seconds\n")
    
    if run >= args.warmup:
        avg_encoder += time_encoder
        
    for prompt in args.prompt:
        print(prompt)
        num_tokens=0
        time_begin=time.perf_counter()
        for token in minigpt4_chatbot.generate(prompt, limit=args.max_new_tokens):
            if num_tokens == 0:
                time_first_token=time.perf_counter()
                latency=time_first_token - time_begin
                time_begin=time_first_token
            print(token, end='')
            sys.stdout.flush()
            num_tokens += 1
        print('\n')
        time_elapsed=time.perf_counter() - time_begin
        tokens_sec=(num_tokens-1) / time_elapsed
        print(f"{model_name}:  {num_tokens} tokens in {time_elapsed:.2f} sec, {tokens_sec:.2f} tokens/sec, latency {latency:.2f} sec\n")
        if run >= args.warmup:
            avg_latency += latency
            avg_tokens_sec += tokens_sec
        
    minigpt4_chatbot.reset_chat()
    
avg_encoder /= args.runs
avg_latency /= args.runs * len(args.prompt)
avg_tokens_sec /= args.runs * len(args.prompt)

memory_usage=get_max_rss()

print(f"AVERAGE of {args.runs} runs:")
print(f"{model_name}:  encoder {avg_encoder:.2f} sec, {avg_tokens_sec:.2f} tokens/sec, latency {avg_latency:.2f} sec, memory {memory_usage:.2f} MB")

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, model, encoder, tokens/sec, latency, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
        file.write(f"minigpt4.cpp, {model_name}, {avg_encoder}, {avg_tokens_sec}, {avg_latency}, {memory_usage}\n")
        
        