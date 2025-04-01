#!/usr/bin/env python3
# benchmark a quantized GGML model with llama_cpp_python API
import os
import time
import datetime
import argparse
import resource
import socket
import pprint

from llama_cpp import Llama


# parse command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str, default='', required=True, help="path to the GGML .bin model")
parser.add_argument('-p', '--prompt', type=str, default='Once upon a time,')

parser.add_argument('-n', '--n-predict', type=int, default=128, help='number of output tokens to generate, including the input prompt')
parser.add_argument('-c', '--ctx-size', type=int, default=512, help='size of the prompt context (default: 512)')
parser.add_argument('-b', '--batch-size', type=int, default=512, help='batch size for prompt processing (default: 512)')
parser.add_argument('-t', '--threads', type=int, default=6, help='number of threads to use during computation (default: 6)')

parser.add_argument('-ngl', '--n-gpu-layers', type=int, default=999, help='number of layers to store in VRAM (default: 999)')
parser.add_argument('-gqa', '--gqa', type=int, default=1, help='grouped-query attention factor (TEMP!!! use 8 for LLaMAv2 70B) (default: 1)')

parser.add_argument('--top-k', type=int, default=40, help='top-k sampling (default: 40, 0 = disabled)')
parser.add_argument('--top-p', type=float, default=0.95, help='top-p sampling (default: 0.95, 1.0 = disabled)')

parser.add_argument('--use-prompt-cache', action='store_true', help='store the model eval results of past runs')
parser.add_argument('--profile-tokenization', action='store_true', help='include the time to tokenize/detokenize in perf measurements')

parser.add_argument('--runs', type=int, default=2, help='the number of benchmark timing iterations')
parser.add_argument('--warmup', type=int, default=2, help='the number of warmup iterations')
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')
   
args = parser.parse_args()

print(args)


def get_max_rss():
    """
    Return the peak memory usage in MB (max RSS - https://stackoverflow.com/a/7669482)
    """
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  
    

model = Llama(model_path=args.model, 
              n_ctx=args.ctx_size, 
              n_batch=args.batch_size, 
              n_gpu_layers=args.n_gpu_layers,
              n_gqa=args.gqa,
              n_threads=args.threads)
              
input_tokens = model.tokenize(args.prompt.encode('utf-8'))

print(f"input_tokens ({len(input_tokens)})", input_tokens)
print(f"system RAM used: {get_max_rss():.2f} MB")

time_avg = 0.0

for run in range(args.runs + args.warmup):
    if not args.use_prompt_cache:
        model.reset()
        
    output_tokens = []
    
    time_begin = time.perf_counter()
    
    if args.profile_tokenization:
        output = model(args.prompt, max_tokens=args.n_predict, top_k=args.top_k, top_p=args.top_p, echo=True)
    else:   
        for token in model.generate(input_tokens, top_k=args.top_k, top_p=args.top_p):
            output_tokens.append(token)
            if len(output_tokens) >= args.n_predict:
                break

    time_elapsed = (time.perf_counter() - time_begin)
    
    if run >= args.warmup:
        time_avg += time_elapsed
        
    if not args.profile_tokenization:
        output = model.detokenize(output_tokens).decode('utf-8', errors='ignore')
            
    print('\n')
    pprint.pprint(output)
    
    print(f"\n{'WARMUP' if run < args.warmup else 'RUN'} {run} = {time_elapsed:.4f} seconds, {args.n_predict/time_elapsed:.1f} tokens/sec")
 
# compute statistics
time_avg /= args.runs  
tokens_sec = args.n_predict / time_avg 
memory_usage = get_max_rss()

print(f"\nAVG = {time_avg:.4f} seconds, {tokens_sec:.1f} tokens/sec  memory={memory_usage:.2f} MB\n")  
print(args)

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, model, tokens, tokens/sec, latency, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
        file.write(f"{os.path.basename(args.model)}, {args.n_predict}, {tokens_sec}, {time_avg}, {memory_usage}\n")
        