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
parser.add_argument('--runs', type=int, default=2, help='the number of benchmark timing iterations')
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

def send_test_prompt(json_data: dict, url:str = "") -> requests.Response:
    ''' send a test prompt to local ollama container '''
    if not url:
        url = "127.0.0.1:11434"
    return requests.post(url, json=json.dumps(json_data))

def run_benchmark(runs: int, json_data: dict, test_url: str = "") -> None:
    ''' run the benchmark '''
    time_avg = 0.0
    for run in range(runs):
        time_begin = time.perf_counter()
        response = send_test_prompt(json_data, test_url)
        time_elapsed = (time.perf_counter() - time_begin)

        if not response.ok:
            pp(f'received error code from api service: {response.status_code}')
            continue
        pp(f'[+] run #{run}')
        pp(f'[-] model: {json_data["model"]}, prompt: {json_data["prompt"]}')
        pp(f'[-] response: {json.loads(response.text)["response"].strip()}, elapsed time: {time_elapsed}')
        time_avg += float(time_elapsed)

    pp(f'[+] peak ram used: {get_max_memory_usage(args.OLLAMA_PID) / 1024 / 1024} MB')
    if runs > 0:
        pp(f'[+] average time: {time_avg / runs:.2f}')

run_benchmark(args.runs, data, "")
