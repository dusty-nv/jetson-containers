#!/usr/bin/env python3
import os
import argparse
import resource

from mlc_chat import ChatModule, ChatConfig
from mlc_chat.callback import StreamToStdout


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="Llama-2-7b-chat-hf-q4f16_1")
parser.add_argument("--prompt", action='append', nargs='*')
parser.add_argument("--chat", action="store_true")
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--max-new-tokens", type=int, default=128)

args = parser.parse_args()

if 'chat' in args.model.lower() and not args.chat:
    args.chat = True
    
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
print(f"-- loading {args.model}")

#conv_config = ConvConfig(system='Please show as much happiness as you can when talking to me.')
#chat_config = ChatConfig(max_gen_len=256, conv_config=conv_config)

#conv_config = ConvConfig(system='Please show as much sadness as you can when talking to me.')
#chat_config = ChatConfig(max_gen_len=128, conv_config=conv_config)
#cm.reset_chat(chat_config)

cfg = ChatConfig(max_gen_len=args.max_new_tokens)

if not args.chat:
    cfg.conv_template = 'LM'
    
cm = ChatModule(model=args.model, chat_config=cfg)

for prompt in args.prompt:
    print(f"\nPROMPT:  {prompt}\n")
    
    if args.streaming:
        output = cm.generate(
            prompt=prompt,
            progress_callback=StreamToStdout(callback_interval=2),
        )
    else:
        print(cm.benchmark_generate(prompt=prompt, generate_length=args.max_new_tokens).strip())

    print(f"\n{args.model}:  {cm.stats()}\n")

    if not args.streaming or not args.chat:
        cm.reset_chat()

memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  # https://stackoverflow.com/a/7669482

print(f"Peak memory usage:  {memory_usage:.2f} MB")