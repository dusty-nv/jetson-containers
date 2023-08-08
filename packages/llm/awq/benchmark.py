#!/usr/bin/env python3
# benchmark a quantized AWQ model
import os
import time
import datetime
import argparse
import resource
import socket
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from awq.quantize.quantizer import real_quantize_model_weight


# parse command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', required=True, help="name or path of the huggingface model")
parser.add_argument('--quant', type=str, default='', required=True, help="path to the real AWQ quantized model checkpoint")
parser.add_argument('--prompt', type=str, default='Once upon a time,')

# benchmarking options
parser.add_argument('--tokens', type=int, nargs='+', default=[128], help='number of output tokens to generate, including the input prompt')
parser.add_argument('--runs', type=int, default=10, help='the number of benchmark timing iterations')
parser.add_argument('--warmup', type=int, default=5, help='the number of warmup iterations')
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')

# quantization options
parser.add_argument('--w_bit', type=int, default=4)
parser.add_argument('--q_group_size', type=int, default=128)
parser.add_argument('--no_zero_point', action='store_true',help="disable zero_point")
                    
args = parser.parse_args()

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}

print("Quantization config:", q_config)

# select compute device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Running on device {device}")
print(f"Loading model {args.model} with quantized weights from {args.quant}")

# load huggingface model, without the weights (just need the model structure)
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    
# prepare model to apply quantized weights
real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config, init_only=True)

# load quantized weights
model = load_checkpoint_and_dispatch(
    model, args.quant, device_map='balanced', 
    no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer"]
)

model.eval()
                  
# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

# benchmark inference
for num_tokens in args.tokens:
    print(f"Generating {num_tokens} tokens with {args.model} fp{args.w_bit} ({args.quant}) on prompt:  {args.prompt}")

    time_avg = 0

    for run in range(args.runs + args.warmup):
        time_begin = time.perf_counter()
        generated_ids = model.generate(input_ids, do_sample=False, min_length=num_tokens, max_length=num_tokens)  # greedy generation of fixed # of tokens   #max_new_tokens=args.max_new_tokens
        time_elapsed = (time.perf_counter() - time_begin)
        
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        
        if run >= args.warmup:
            time_avg += time_elapsed
            
        print(f"\n{'WARMUP' if run < args.warmup else 'RUN'} {run} = {time_elapsed:.4f} seconds, {num_tokens/time_elapsed:.1f} tokens/sec (fp{args.w_bit})")
      
    # compute statistics
    time_avg /= args.runs  
    tokens_sec = num_tokens / time_avg
    memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  # https://stackoverflow.com/a/7669482
    memory_info_gpu = torch.cuda.mem_get_info()

    print(f"AVG = {time_avg:.4f} seconds, {tokens_sec:.1f} tokens/sec  memory={memory_usage:.2f} MB  (--model={args.model} --quant={args.quant} --w_bit={args.w_bit} --tokens={num_tokens})\n")

    if args.save:
        if not os.path.isfile(args.save):  # csv header
            with open(args.save, 'w') as file:
                file.write(f"timestamp, hostname, model, precision, tokens, tokens/sec, latency, memory\n")
        with open(args.save, 'a') as file:
            file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
            file.write(f"{args.quant}, fp{args.w_bit}, {num_tokens}, {tokens_sec}, {time_avg}, {memory_usage}\n")
            