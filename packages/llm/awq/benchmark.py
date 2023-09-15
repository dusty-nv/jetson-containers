#!/usr/bin/env python3
# benchmark a quantized AWQ model
import os
import sys
import time
import datetime
import argparse
import resource
import socket
import threading
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from awq.quantize.quantizer import real_quantize_model_weight

from tinychat.demo import gen_params, stream_output
from tinychat.stream_generators import StreamGenerator
from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp
from tinychat.utils.prompt_templates import get_prompter

# parse command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', required=True, help="name or path of the huggingface model")
parser.add_argument('--quant', type=str, default='', required=True, help="path to the real AWQ quantized model checkpoint")
parser.add_argument('--prompt', type=str, default='Once upon a time,')

# benchmarking options
parser.add_argument('--tokens', type=int, default=128, help='number of output tokens to generate, including the input prompt')
parser.add_argument('--runs', type=int, default=2, help='the number of benchmark timing iterations')
parser.add_argument('--warmup', type=int, default=2, help='the number of warmup iterations')
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')

# quantization options
parser.add_argument('--w_bit', type=int, default=4)
parser.add_argument('--q_group_size', type=int, default=128)
parser.add_argument('--no_zero_point', action='store_true', help="disable zero_point")
parser.add_argument('--no_tinychat_kernels', action='store_true', help="disable tinychat kernels")
parser.add_argument('--no_tinychat_infer', action='store_true', help="disable tinychat inference")
parser.add_argument('--no_streaming', action='store_true', help="disable streaming mode")
parser.add_argument('--no_quant', action='store_true', help="disable quantization and use FP16 through transformers")
parser.add_argument('--do_sample', action='store_true')

args = parser.parse_args()

#args.prompt="Once upon a time, there was a young man named Jack who lived in a small village nestled in the rolling hills of the countryside. Jack was a curious and adventurous soul, always eager to explore the world beyond his village. One day, while wandering through the nearby forest, he stumbled upon a hidden path that he had never seen before. The path was overgrown with weeds and vines, and it looked as though it had been untouched for many years. Jack's curiosity was piqued, and he decided to follow the path to see where it led"
#args.prompt="Once upon a time, there was a young man named Jack who lived in a small village nestled in the rolling hills of the countryside. Jack was a curious and adventurous soul, always eager to explore the world beyond his village. One day, while wandering through the nearby forest, he stumbled upon a hidden path that he had never seen before. The path was overgrown with weeds and vines, and it looked as though it had been untouched for many years. Jack's curiosity was piqued, and he decided to follow the path to see where it led. As he walked down the path, the trees grew taller and the air grew colder. Jack could feel a strange energy emanating from the forest, as if it were alive and watching him. He quickened his pace, eager to reach the end of the path and discover its secrets. After a while, the path opened up into a clearing, and Jack found himself standing in front of a massive stone structure. The building was unlike anything he had ever seen before, with intricate carvings and symbols etched into its walls. Jack felt a sense of awe and wonder as he approached the"

print(args)

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    
    if len(model_paths) > 2:
        return model_paths[-2] + '/' + model_paths[-1]
    else:
        return model_paths[-1]
        
model_name = get_model_name_from_path(args.quant)
       
# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}

print("Quantization config:", q_config)

# select compute device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Running on device {device}")


if args.no_quant:
    precision = "fp16"
    print(f"Loading model {args.model} without quantization ({precision})")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
else:
    print(f"Loading model {args.model} with quantized weights from {args.quant}")
    
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)

    print(model)
    print(f"model device: {model.device}")
         
    # prepare model to apply quantized weights
    real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config, init_only=True)

    # load quantized weights
    model = load_checkpoint_and_dispatch(
        model, args.quant, device_map='balanced', 
        no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer"]
    )
    
    precision = f"int{args.w_bit}"

    print(model)
    print(f"model device: {model.device}")
     
    if not args.no_tinychat_kernels:
        make_quant_attn(model, device)
        make_quant_norm(model)
        make_fused_mlp(model)

    print(model)
    print(f"model device: {model.device}")

#for name, param in model.named_parameters():
#    print(f"{name} {param}")
    
model.eval()
                  
# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

# benchmark inference
avg_latency=0
avg_tokens_sec=0

for run in range(args.runs + args.warmup):
    if args.no_streaming:
        time_begin = time.perf_counter()
        generated_ids = model.generate(input_ids, do_sample=args.do_sample, min_new_tokens=args.tokens, max_new_tokens=args.tokens)
        time_elapsed = time.perf_counter() - time_begin
        
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=False))
        
        num_tokens=len(generated_ids[0])
        tokens_sec=num_tokens / time_elapsed
        latency=time_elapsed
    else:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        
        def generate():
            with torch.inference_mode():
                model.generate(
                    inputs=input_ids,
                    do_sample=args.do_sample,
                    min_new_tokens=args.tokens, 
                    max_new_tokens=args.tokens,
                    streamer=streamer
                )
        
        thread = threading.Thread(target=generate)
        thread.start()
        
        print(f"Prompt:  {args.prompt}")
        
        new_tokens = ''
        num_tokens = 0
        time_begin = time.perf_counter()
       
        for token in streamer:
            print(token, end='')
            sys.stdout.flush()

            if num_tokens == 0:
                time_first_token=time.perf_counter()
                latency=time_first_token - time_begin
                time_begin=time_first_token
                
            new_tokens += token
            num_tokens += 1
    
        time_elapsed=time.perf_counter() - time_begin
        tokens_sec=(num_tokens-1) / time_elapsed
        
    print(f"\n{model_name}:  {num_tokens} tokens in {time_elapsed:.2f} sec, {tokens_sec:.2f} tokens/sec, latency {latency:.2f} sec  ({precision})\n")
            
    if run >= args.warmup:
        avg_latency += latency
        avg_tokens_sec += tokens_sec
  
# compute statistics
avg_latency /= args.runs
avg_tokens_sec /= args.runs

memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  # https://stackoverflow.com/a/7669482

print(f"AVERAGE of {args.runs} runs:")
print(f"{model_name}:  {avg_tokens_sec:.2f} tokens/sec, latency {avg_latency:.2f} sec, memory={memory_usage:.2f} MB  ({precision})\n")

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, model, precision, tokens, tokens/sec, latency, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, {'tinychat' if args.tiny_chat else 'awq'}, ")
        file.write(f"{model_name}, {precision}, {args.tokens}, {avg_tokens_sec}, {avg_latency}, {memory_usage}\n")
            