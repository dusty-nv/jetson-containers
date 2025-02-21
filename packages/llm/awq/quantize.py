#!/usr/bin/env python3
# quantize a HuggingFace CausalLM .pt model with AWQ
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True, help="model name or path to model directory (should be a .pt model)")
parser.add_argument('--output', type=str, default='', help="directory to output the quantized model to (default is /data/models/awq/$MODEL)")
parser.add_argument('--load_awq', type=str, default='', help="load a model that's already had AWQ search performed on it (i.e. from the AWQ Model Zoo), and skip the search step")

parser.add_argument('--w_bit', type=int, default=4, choices=[3,4], help="the number of bits (3 or 4)")
parser.add_argument('--q_group_size', type=int, default=128, help="the group size (default 128)")
parser.add_argument('--no_cache', action='store_true', help="dump the quantized AWQ weights even if the file already exists")
parser.add_argument('--skip_eval', action='store_true', help="evaluate the real quantized model on wikitext")
parser.add_argument('--simulate', action='store_true', help="print out the commands without actually running them")

args = parser.parse_args()

if not args.output:
    args.output = f"/data/models/awq/{os.path.basename(args.model)}"

print(args)

os.makedirs(args.output, exist_ok=True)

prefix_awq = f"w{args.w_bit}-g{args.q_group_size}"

model_search = os.path.join(args.output, f"{prefix_awq}.pt") if not args.load_awq else args.load_awq
model_quant = os.path.join(args.output, f"{prefix_awq}-awq-v2.pt")

def run_cmd(cmd):
    print("\nRunning command:\n")
    print(cmd, "\n")
    if not args.simulate:
        subprocess.run(cmd, executable='/bin/bash', shell=True, check=True) 
    
cmd_prefix = f"python3 -m awq.entry --model_path {args.model} --w_bit {args.w_bit} --q_group_size {args.q_group_size}"

# Perform AWQ search
if not args.load_awq:
    run_cmd(f"{cmd_prefix} --run_awq --dump_awq {model_search}")

# Evaluate the AWQ quantized model on WikiText-2 (simulated pseudo quantization)
if not args.load_awq and not args.skip_eval:
    run_cmd(f"{cmd_prefix} --tasks wikitext --load_awq {model_search} --q_backend fake")

# Generate real quantized weights (INT4)
if args.no_cache or not os.path.isfile(model_quant):
    run_cmd(f"{cmd_prefix} --load_awq {model_search} --q_backend real --dump_quant {model_quant}")

# Load and evaluate the real quantized model
if not args.skip_eval:
    run_cmd(f"{cmd_prefix} --tasks wikitext --load_quant {model_quant}")
