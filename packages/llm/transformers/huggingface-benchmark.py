#!/usr/bin/env python3
# benchmark a text-generation model (CausalLM) with huggingface transformers library
import os
import time
import datetime
import argparse
import resource
import socket
import pprint

import torch
import huggingface_hub

from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='distilgpt2')
parser.add_argument('--prompt', type=str, default='Once upon a time,')
parser.add_argument('--precision', type=str, default=None, choices=['fp32', 'fp16', 'fp4', 'int8'])
parser.add_argument('--tokens', type=int, nargs='+', default=[128], help='number of output tokens to generate (not including the input prompt)')
parser.add_argument('--token', type=str, default=os.environ.get('HUGGINGFACE_TOKEN', ''), help="HuggingFace account login token from https://huggingface.co/docs/hub/security-tokens (defaults to $HUGGINGFACE_TOKEN)")
parser.add_argument('--runs', type=int, default=2, help='the number of benchmark timing iterations')
parser.add_argument('--warmup', type=int, default=2, help='the number of warmup iterations')
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')

args = parser.parse_args()
print(args)

# select compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on device {device}')

# log into huggingface hub
if args.token:
    print("Logging into HuggingFace Hub...")
    huggingface_hub.login(token=args.token)
  
# detect the type of model it is
model_info = huggingface_hub.model_info(args.model)
model_type = model_info.transformersInfo['auto_model']

if model_type != 'AutoModelForCausalLM':
    raise ValueError(f"text-generation benchmark only supports CausalLM models (GPT,llama,ect) - {args.model} is {model_type}")

# end the prompt with a newline
#args.prompt += '\n'

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

print('Input tokens:', input_ids, 'shape:', input_ids.shape)

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
print(f'Loading model {args.model} ({args.precision})')

model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device, **kwargs) #AutoModelForCausalLM.from_pretrained(args.model, **kwargs)

#if args.precision == 'fp32' or args.precision == 'fp16':
#    model = model.to(device)   # int8/int4 already sets the device
    
# run inference
for num_tokens in args.tokens:
    print(f"Generating {num_tokens} tokens with {args.model} on prompt:  {args.prompt}")

    time_avg = 0

    for run in range(args.runs + args.warmup):
        time_begin = time.perf_counter()
        generated_ids = model.generate(input_ids, do_sample=False, min_length=num_tokens+input_ids.shape[1], max_length=num_tokens+input_ids.shape[1]) #min_new_tokens=num_tokens, max_new_tokens=num_tokens)  # greedy generation of fixed # of tokens
        time_elapsed = (time.perf_counter() - time_begin)
        
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        
        if run >= args.warmup:
            time_avg += time_elapsed
            
        print(f"\n{'WARMUP' if run < args.warmup else 'RUN'} {run} = {time_elapsed:.4f} seconds, {num_tokens/time_elapsed:.1f} tokens/sec ({args.precision})")
      
    # compute statistics
    time_avg /= args.runs  
    tokens_sec = num_tokens / time_avg
    memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  # https://stackoverflow.com/a/7669482

    print(f"AVG = {time_avg:.4f} seconds, {tokens_sec:.1f} tokens/sec  memory={memory_usage:.2f} MB  (--model={args.model} --precision={args.precision} --tokens={num_tokens})\n")

    if args.save:
        if not os.path.isfile(args.save):  # csv header
            with open(args.save, 'w') as file:
                file.write(f"timestamp, hostname, model, precision, tokens, tokens/sec, latency, memory\n")
        with open(args.save, 'a') as file:
            file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
            file.write(f"{args.model}, {args.precision}, {num_tokens}, {tokens_sec}, {time_avg}, {memory_usage}\n")
