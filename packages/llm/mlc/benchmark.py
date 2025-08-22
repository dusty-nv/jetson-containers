#!/usr/bin/env python3
import os
import sys
import json
import socket
import datetime
import argparse
import resource

import tvm
import mlc_llm

from packaging.version import Version

# earlier builds of MLC didn't expose the version
TVM_VERSION = Version(tvm.__version__)

try:
    MLC_VERSION = Version(mlc_llm.__version__)
except Exception as error:
    print(f"failed to get MLC version ({error})")
    if TVM_VERSION == Version('0.15.0'):
        MLC_VERSION = Version('0.1.0')
    elif TVM_VERSION == Version('0.16.0'):
        MLC_VERSION = Version('0.1.1')
    else:
        raise ImportError(f"failed to get MLC version ({error}) and unknown TVM version ({TVM_VERSION})")
    print(f"found TVM version {TVM_VERSION} -> MLC version {MLC_VERSION}\n")
    
print(f"TVM version:  {TVM_VERSION}")
print(f"MLC version:  {MLC_VERSION}\n")

print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))

# handle API changes across MLC
try:
    from mlc_llm import MLCEngine
    from mlc_llm.serve import EngineConfig
    USE_MLC_CHAT = False
except Exception:
    try:
        from mlc_llm import ChatModule, ChatConfig
        from mlc_llm.callback import StreamToStdout
        USE_MLC_CHAT = True
    except Exception:
        from mlc_chat import ChatModule, ChatConfig
        from mlc_chat.callback import StreamToStdout
        USE_MLC_CHAT = True

# parse model arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="Llama-2-7b-chat-hf-q4f16_1")
parser.add_argument('--model-lib-path', type=str, default=None)
parser.add_argument("--prompt", action='append', nargs='*')
parser.add_argument("--chat", action="store_true")
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--max-num-prompts", type=int, default=None)
parser.add_argument("--max-context-len", type=int, default=None)
parser.add_argument("--prefill-chunk-size", type=int, default=None)
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')

args = parser.parse_args()

# assign default prompts
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
                elif isinstance(json_prompt, str):
                    prompt_list.append(json_prompt)
                else:
                    raise TypeError(f"{type(json_prompt)}")
        elif ext == '.txt':
            with open(prompt) as file:
                prompt_list.append(file.read())
        else:
            prompt_list.append(prompt)
            
    return prompt_list
    
# load prompts if given a txt/json file
args.prompt = load_prompts(args.prompt)

if args.max_num_prompts:
    args.prompt = args.prompt[:args.max_num_prompts]
    
# load the model
print(f"-- loading {args.model}")

if USE_MLC_CHAT:
    cfg = ChatConfig(max_gen_len=args.max_new_tokens)

    if not args.chat:
        cfg.conv_template = 'LM'
        
    model = ChatModule(model=args.model, model_lib_path=args.model_lib_path, chat_config=cfg)
else:
    cfg = EngineConfig(
        max_num_sequence=1, 
        max_single_sequence_length=args.max_context_len, 
        prefill_chunk_size=args.prefill_chunk_size
    )
    model = MLCEngine(args.model, model_lib=args.model_lib_path, mode='interactive', engine_config=cfg)

def generate(prompt, stats):
    # MLC <= 0.1.1
    if args.streaming:
        output = model.generate(
            prompt=prompt,
            progress_callback=StreamToStdout(callback_interval=2),
        )
    else:
        print(model.benchmark_generate(
            prompt=prompt, 
            generate_length=args.max_new_tokens
        ).strip())

    stats_str = model.stats()
    stats_split = stats_str.split(' ')
    
    stats['prefill_rate'] = float(stats_split[1])
    stats['decode_rate'] = float(stats_split[4])
    stats['output_tokens'] = args.max_new_tokens
  
    if not args.streaming or not args.chat:
        model.reset_chat()
        
def generate_v2(prompt, stats):
    # MLC >= 0.1.2
    for response in model.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=args.model,
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=args.max_new_tokens,
        logit_bias={
            128001: -100,
            128008: -100,
            128009: -100,
        }
    ):
        if response.usage is not None:
            usage = response.usage.extra
            continue
            
        for choice in response.choices:
            print(choice.delta.content, end="", flush=True)

    stats['input_tokens'] = usage['prefill_tokens']
    stats['output_tokens'] = usage['completion_tokens']
    
    stats['prefill_rate'] = usage['prefill_tokens_per_s']
    stats['decode_rate'] = usage.get('decode_tokens_per_s', -1) 
      
# benchmark inference
avg_stats = {}

for i, prompt in enumerate(args.prompt):
    stats = {}
    
    if isinstance(prompt, dict):
        stats['input_tokens'] = prompt['num_tokens']
        prompt = prompt['text']
    elif USE_MLC_CHAT:
        stats['input_tokens'] = model.embed_text(prompt).shape[1]
        model.reset_chat()
        
    print(f"\nPROMPT:  {prompt}\n")
    
    if USE_MLC_CHAT:
        generate(prompt, stats)
    else:
        while True:  # sometimes stops generation early (on EOS)
            generate_v2(prompt, stats)
            if stats['output_tokens'] >= args.max_new_tokens * 0.5:
                break
            print(f"\nShort generation ({stats['output_tokens']} of {args.max_new_tokens} tokens) - retrying...")
                
    stats['prefill_time'] = stats['input_tokens'] / stats['prefill_rate']
    stats['decode_time'] = stats['output_tokens'] / stats['decode_rate']
    
    print(f"\n{args.model}:  input={stats['input_tokens']} output={stats['output_tokens']} prefill_time {stats['prefill_time']:.3f} sec, prefill_rate {stats['prefill_rate']:.1f} tokens/sec, decode_time {stats['decode_time']:.3f} sec, decode_rate {stats['decode_rate']:.1f} tokens/sec\n")

    if i > 0:
        for key in stats:
            avg_stats[key] = avg_stats.get(key, 0) + stats[key] * (1.0 / (len(args.prompt) - 1))

avg_stats['input_tokens'] = int(round(avg_stats['input_tokens']))
avg_stats['output_tokens'] = int(round(avg_stats['output_tokens']))

print(f"AVERAGE OVER {len(args.prompt) - 1} RUNS  (input_tokens={avg_stats['input_tokens']}, output_tokens={avg_stats['output_tokens']})")
print(f"{args.model}:  prefill_time {avg_stats['prefill_time']:.3f} sec, prefill_rate {avg_stats['prefill_rate']:.1f} tokens/sec, decode_time {avg_stats['decode_time']:.3f} sec, decode_rate {avg_stats['decode_rate']:.1f} tokens/sec\n")

memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  # https://stackoverflow.com/a/7669482

print(f"Peak memory usage:  {memory_usage:.2f} MB")

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, model, precision, input_tokens, output_tokens, prefill_time, prefill_rate, decode_time, decode_rate, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, mlc, ")
        file.write(f"{args.model}, {args.model.split('-')[-1]}, {avg_stats['input_tokens']}, {avg_stats['output_tokens']}, ")
        file.write(f"{avg_stats['prefill_time']}, {avg_stats['prefill_rate']}, {avg_stats['decode_time']}, {avg_stats['decode_rate']}, {memory_usage}\n")
    print(f"Saved results to:   {args.save}")

del model  # MLC >= 0.1.2 hangs on exit unless the model is released
        
