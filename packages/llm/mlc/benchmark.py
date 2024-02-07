#!/usr/bin/env python3
import os
import json
import socket
import datetime
import argparse
import resource

from mlc_chat import ChatModule, ChatConfig
from mlc_chat.callback import StreamToStdout


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="Llama-2-7b-chat-hf-q4f16_1")
parser.add_argument('--model-lib-path', type=str, default=None)
parser.add_argument("--prompt", action='append', nargs='*')
parser.add_argument("--chat", action="store_true")
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--max-num-prompts", type=int, default=None)
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')

args = parser.parse_args()

#if 'chat' in args.model.lower() and not args.chat:
#    args.chat = True
    
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
    
print(f"-- loading {args.model}")

#conv_config = ConvConfig(system='Please show as much happiness as you can when talking to me.')
#chat_config = ChatConfig(max_gen_len=256, conv_config=conv_config)

#conv_config = ConvConfig(system='Please show as much sadness as you can when talking to me.')
#chat_config = ChatConfig(max_gen_len=128, conv_config=conv_config)
#cm.reset_chat(chat_config)

cfg = ChatConfig(max_gen_len=args.max_new_tokens)

if not args.chat:
    cfg.conv_template = 'LM'
    
cm = ChatModule(model=args.model, model_lib_path=args.model_lib_path, chat_config=cfg)

avg_prefill_rate = 0
avg_prefill_time = 0
avg_decode_rate = 0
avg_decode_time = 0

for i, prompt in enumerate(args.prompt):
    if isinstance(prompt, dict):
        num_input_tokens = prompt['num_tokens']
        prompt = prompt['text']
    else:
        num_input_tokens = cm.embed_text(prompt).shape[1]
        cm.reset_chat()
        
    print(f"\nPROMPT:  {prompt}\n")
    
    if args.streaming:
        output = cm.generate(
            prompt=prompt,
            progress_callback=StreamToStdout(callback_interval=2),
        )
    else:
        print(cm.benchmark_generate(prompt=prompt, generate_length=args.max_new_tokens).strip())

    stats_str = cm.stats()
    stats_split = stats_str.split(' ')
    prefill_rate = float(stats_split[1])
    decode_rate = float(stats_split[4])
    
    prefill_time = num_input_tokens / prefill_rate
    decode_time = args.max_new_tokens / decode_rate
    
    if i > 0:
        avg_factor = 1.0 / (len(args.prompt) - 1)
        avg_prefill_rate += prefill_rate * avg_factor
        avg_prefill_time += prefill_time * avg_factor
        avg_decode_rate += decode_rate * avg_factor
        avg_decode_time += decode_time * avg_factor
        
    print(f"\n{args.model}:  input={num_input_tokens} output={args.max_new_tokens} prefill_time {prefill_time:.3f} sec, prefill_rate {prefill_rate:.1f} tokens/sec, decode_time {decode_time:.3f} sec, decode_rate {decode_rate:.1f} tokens/sec\n")

    if not args.streaming or not args.chat:
        cm.reset_chat()

print(f"AVERAGE OVER {len(args.prompt) - 1} RUNS, input={num_input_tokens}, output={args.max_new_tokens}")
print(f"{args.model}:  prefill_time {avg_prefill_time:.3f} sec, prefill_rate {avg_prefill_rate:.1f} tokens/sec, decode_time {avg_decode_time:.3f} sec, decode_rate {avg_decode_rate:.1f} tokens/sec\n")

memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  # https://stackoverflow.com/a/7669482

print(f"Peak memory usage:  {memory_usage:.2f} MB")

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, model, precision, input_tokens, output_tokens, prefill_time, prefill_rate, decode_time, decode_rate, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, mlc, ")
        file.write(f"{args.model}, {args.model.split('-')[-1]}, {num_input_tokens}, {args.max_new_tokens}, ")
        file.write(f"{avg_prefill_time}, {avg_prefill_rate}, {avg_decode_time}, {avg_decode_rate}, {memory_usage}\n")
        
