#!/usr/bin/env python3
# benchmark a quantized AWQ model
import os
import sys
import time
import json
import datetime
import argparse
import resource
import socket
import threading
import torch
import tinychat

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
parser.add_argument("--prompt", action='append', nargs='*')

# benchmarking options
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--max-num-prompts", type=int, default=None)
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')

# quantization options
parser.add_argument('--w_bit', type=int, default=4)
parser.add_argument('--q_group_size', type=int, default=128)
parser.add_argument('--no_zero_point', action='store_true', help="disable zero_point")
parser.add_argument('--no_tinychat_kernels', action='store_true', help="disable tinychat kernels")
#parser.add_argument('--no_tinychat_infer', action='store_true', help="disable tinychat inference")
parser.add_argument('--no_quant', action='store_true', help="disable quantization and use FP16 through transformers")
parser.add_argument('--do_sample', action='store_true')

args = parser.parse_args()

if not args.prompt:
    if args.chat:  # https://modal.com/docs/guide/ex/vllm_inference
        args.prompt = [
            "What is the meaning of life?",
            "How many points did you list out?",
            "What is the weather forecast today?",
            "What is the fable involving a fox and grapes?",
            "What's a good recipe for making tabouli?",
            "What is the product of 9 and 8?",
            "If a train travels 120 miles in 2 hours, what is its average speed?",
        ]
    else:
        args.prompt = [
            "Once upon a time,",
            "A great place to live is",
            "In a world where dreams are shared,",
            "The weather forecast today is",
            "Large language models are",
            "Space exploration is exciting",
            "The history of the Hoover Dam is",
            "San Fransisco is a city in",
            "To train for running a marathon,",
            "A recipe for making tabouli is"
        ]
else:
    args.prompt = [x[0] for x in args.prompt]
    
print(args)

def load_prompts(prompts):
    """
    Load prompts from a list of txt or json files
    (or if these are strings, just return the strings)
    """
    prompt_list = []
    
    for prompt in prompts:
        ext = os.path.splitext(prompt)[1]
        
        if ext == '.json':
            with open(prompt) as file:
                json_prompts = json.load(file)
            for json_prompt in json_prompts:
                if isinstance(json_prompt, dict):
                    prompt_list.append(json_prompt)  # json_prompt['text']
                elif ifinstance(json_prompt, str):
                    prompt_list.append(json_prompt)
                else:
                    raise TypeError(f"{type(json_prompt)}")
        elif ext == '.txt':
            with open(prompt) as file:
                prompt_list.append(file.read())
        else:
            prompt_list.append(prompt)
            
    return prompt_list
    
args.prompt = load_prompts(args.prompt)

if args.max_num_prompts:
    args.prompt = args.prompt[:args.max_num_prompts]
    
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
    
    precision = f"W{args.w_bit}A16"

    print(model)
    print(f"model device: {model.device}")
     
    if not args.no_tinychat_kernels:
        tinychat.utils.constants.max_seq_len = model.config.max_position_embeddings
        make_quant_attn(model, device)
        make_quant_norm(model)
        make_fused_mlp(model)
        print("TinyChat model:\n", model)
        print(f"Model max context length: {model.config.max_position_embeddings}")
        
#for name, param in model.named_parameters():
#    print(f"{name} {param}")
    
model.eval()
                  
# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

# benchmark inference
avg_prefill_time = 0
avg_prefill_rate = 0
avg_decode_time = 0
avg_decode_rate = 0

for i, prompt in enumerate(args.prompt):
    if isinstance(prompt, dict):
        prompt = prompt['text']
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    num_input_tokens = input_ids.shape[-1]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    time_begin = 0
    
    def generate():
        global time_begin
        time_begin = time.perf_counter()
        with torch.inference_mode():
            model.generate(
                inputs=input_ids,
                do_sample=args.do_sample,
                min_new_tokens=args.max_new_tokens, 
                max_new_tokens=args.max_new_tokens,
                streamer=streamer
            )
    
    thread = threading.Thread(target=generate)
    thread.start()
    
    print(f"Prompt:  {prompt}\n")
    
    new_tokens = ''
    num_tokens = 0

    for token in streamer:
        print(token, end='', flush=True)

        if num_tokens == 0:
            time_first_token=time.perf_counter()
            prefill_time=time_first_token - time_begin
            time_begin=time_first_token
            
        new_tokens += token
        num_tokens += 1

    decode_time=time.perf_counter() - time_begin
    decode_rate=(args.max_new_tokens-1) / decode_time
    prefill_rate=num_input_tokens / prefill_time

    print(f"\n\n{model_name}:  input={num_input_tokens} output={num_tokens} prefill_time {prefill_time:.3f} sec, prefill_rate {prefill_rate:.1f} tokens/sec, decode_time {decode_time:.3f} sec, decode_rate {decode_rate:.1f} tokens/sec\n")
            
    if i > 0:
        avg_factor = 1.0 / (len(args.prompt) - 1)
        avg_prefill_time += prefill_time * avg_factor
        avg_prefill_rate += prefill_rate * avg_factor
        avg_decode_time += decode_time * avg_factor
        avg_decode_rate += decode_rate * avg_factor
  
print(f"AVERAGE OVER {len(args.prompt) - 1} RUNS, input={num_input_tokens}, output={args.max_new_tokens}, precision={precision}")
print(f"{model_name}:  prefill_time {avg_prefill_time:.3f} sec, prefill_rate {avg_prefill_rate:.1f} tokens/sec, decode_time {avg_decode_time:.3f} sec, decode_rate {avg_decode_rate:.1f} tokens/sec\n")

memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  # https://stackoverflow.com/a/7669482
print(f"Peak memory usage:  {memory_usage:.2f} MB")

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, model, precision, input_tokens, output_tokens, prefill_time, prefill_rate, decode_time, decode_rate, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, {'tinychat' if not args.no_tinychat_kernels else 'awq'}, ")
        file.write(f"{model_name}, {precision}, {num_input_tokens}, {args.max_new_tokens}, ")
        file.write(f"{avg_prefill_time}, {avg_prefill_rate}, {avg_decode_time}, {avg_decode_rate}, {memory_usage}\n")
            