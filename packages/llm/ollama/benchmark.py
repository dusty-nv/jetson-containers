#!/usr/bin/env python3
# benchmark a quantized GGML model with ollama API
import time
import argparse
import json
import requests
from pprint import pp

# Small LLM: tinyllama

DEFAULT_PROMPT = {
  "model": "tinyllama",
  "prompt": "Why is the sky blue?",
  "options": {
    "seed": 123,
    "temperature": 0
  },
  "format": "json",
  "stream": False,
}

# parse command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str, default='tinyllama', required=True, help="name of model to run")
parser.add_argument('-p', '--prompt', type=str, default=DEFAULT_PROMPT.get("prompt"))

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

parser.add_argument('--OLLAMA_PID', type=str, default="", required=True, help='the pid of the ollama process')

args = parser.parse_args()

print(args)

data = DEFAULT_PROMPT.copy()
data['prompt'] = args.prompt

def get_max_memory_usage(PID: str = "self") -> None:
    ''' Maximum memory usage in bytes '''
    with open(f'/proc/{PID}/status', encoding='utf-8') as f:
        memusage = f.read().split('VmPeak:')[1].split('\n')[0][:-3]

    return int(memusage.strip()) / 1024

# Syntax for requests.post():
# url = 'https://www.w3schools.com/python/demopage.php'
# myobj = {'somekey': 'somevalue'}
# x = requests.post(url, json = myobj)
def send_test_prompt(json_data: dict, url:str ="127.0.0.1:11434") -> requests.Response:
    ''' send a test prompt to local ollama container '''
    return requests.post(url, json=json.dumps(json_data))

a = requests.Response()
time_avg = 0.0

for run in range(args.runs):
    time_begin = time.perf_counter()
    response = send_test_prompt(json_data=data)
    time_elapsed = (time.perf_counter() - time_begin)
    
    if not response.ok:
        pp(f'received error code from api service: {response.status_code}')
        continue
    pp(f'[+] run #{run}')
    pp(f'[-] model: {data["model"]}, prompt: {data["prompt"]}')
    pp(f'[-] response: {json.loads(response.text)["response"].strip()}, elapsed time: {time_elapsed}')
    time_avg += float(time_elapsed)
    
pp(f'[+] peak ram used: {get_max_memory_usage(args.OLLAMA_PID) / 1024 / 1024} MB')
if args.runs > 0:
    pp(f'[+] average time: {time_avg / args.runs:.2f}')