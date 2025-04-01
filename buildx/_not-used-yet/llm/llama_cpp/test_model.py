#!/usr/bin/env python3
import argparse
import pprint

from llama_cpp import Llama

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
parser.add_argument('-m', '--model', type=str, default='', required=True, help="path to the GGML .bin model")
parser.add_argument('-p', '--prompt', type=str, default='Once upon a time,')

parser.add_argument('-n', '--n-predict', type=int, default=128, help='number of output tokens to generate, including the input prompt')
parser.add_argument('-c', '--ctx-size', type=int, default=512, help='size of the prompt context (default: 512)')
parser.add_argument('-b', '--batch-size', type=int, default=512, help='batch size for prompt processing (default: 512)')
parser.add_argument('-t', '--threads', type=int, default=6, help='number of threads to use during computation (default: 6)')

parser.add_argument('-ngl', '--n-gpu-layers', type=int, default=999, help='number of layers to store in VRAM (default: 999)')
parser.add_argument('-gqa', '--gqa', type=int, default=1, help='grouped-query attention factor (TEMP!!! use 8 for LLaMAv2 70B) (default: 1)')

args = parser.parse_args()
print(args)

model = Llama(model_path=args.model, 
              n_ctx=args.ctx_size, 
              n_batch=args.batch_size, 
              n_gpu_layers=args.n_gpu_layers,
              n_gqa=args.gqa,
              n_threads=args.threads)
              
print(f"\nPROMPT: {args.prompt}\n")
pprint.pprint(model(args.prompt, max_tokens=args.n_predict, echo=False))

print("\nllama.cpp OK")