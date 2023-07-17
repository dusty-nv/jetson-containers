#!/usr/bin/env python3
# benchmark a text-generation model with huggingface transformers library
import os
import time
import datetime
import argparse
import resource
import socket
import torch

import transformers
print('transformers version:', transformers.__version__)

from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='distilgpt2')
parser.add_argument('--prompt', type=str, default='California is in which country?')
parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'fp4', 'int8'])
parser.add_argument('--tokens', type=int, nargs='+', default=[20], help='number of output tokens to generate, including the input prompt')
parser.add_argument('--runs', type=int, default=5, help='the number of benchmark timing iterations')
parser.add_argument('--warmup', type=int, default=1, help='the number of warmup iterations')
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')

args = parser.parse_args()
print(args)

# select compute device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Running on device {device}')

# end the prompt with a newline
args.prompt += '\n'

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

# setup precision args
kwargs = {}

if args.precision == 'int8':
    kwargs['load_in_8bit'] = True
    #kwargs['int8_threshold'] = 0   # https://github.com/TimDettmers/bitsandbytes/issues/6#issuecomment-1225990890
elif args.precision == 'fp4':
    kwargs['load_in_4bit'] = True
elif args.precision == 'fp16':
    kwargs['torch_dtype'] = torch.float16
elif args.precision == 'fp32':
    kwargs['torch_dtype'] = torch.float32
    
# load model
print(f'Loading model {args.model}')
    
model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)

if args.precision == 'fp32' or args.precision == 'fp16':
    model = model.to(device)   # int8/int4 already sets the device
    
# run inference
for num_tokens in args.tokens:
    print(f"Generating {num_tokens} tokens with {args.model} {args.precision} on prompt:  {args.prompt}")

    time_avg = 0

    for run in range(args.runs + args.warmup):
        time_begin = time.perf_counter()
        generated_ids = model.generate(input_ids, do_sample=False, min_length=num_tokens, max_length=num_tokens)  # greedy generation of fixed # of tokens   #max_new_tokens=args.max_new_tokens
        time_elapsed = (time.perf_counter() - time_begin)
        
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        
        if run >= args.warmup:
            time_avg += time_elapsed
            
        print(f"\n{'WARMUP' if run < args.warmup else 'RUN'} {run} = {time_elapsed:.4f} seconds, {num_tokens/time_elapsed:.1f} tokens/sec ({args.precision})")
      
    # compute statistics
    time_avg /= args.runs  
    tokens_sec = num_tokens / time_avg
    memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  # https://stackoverflow.com/a/7669482
    memory_info_gpu = torch.cuda.mem_get_info()

    print(f"AVG = {time_avg:.4f} seconds, {tokens_sec:.1f} tokens/sec  memory={memory_usage:.2f} MB  (--model={args.model} --precision={args.precision} --tokens={num_tokens})\n")

    if args.save:
        if not os.path.isfile(args.save):  # csv header
            with open(args.save, 'w') as file:
                file.write(f"timestamp, hostname, model, precision, tokens, tokens/sec, latency, memory\n")
        with open(args.save, 'a') as file:
            file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
            file.write(f"{args.model}, {args.precision}, {num_tokens}, {tokens_sec}, {time_avg}, {memory_usage}\n")
