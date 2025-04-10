#!/usr/bin/env python3
import os
import sys
import time
import signal
import logging
import numpy as np

from termcolor import cprint

from local_llm import LocalLM, StopTokens
from local_llm.utils import ArgParser, KeyboardInterrupt, load_prompts, print_table 

# see utils/args.py for options
parser = ArgParser()
parser.add_argument("--no-streaming", action="store_true", help="wait to output entire reply instead of token by token")
args = parser.parse_args()

prompts = load_prompts(args.prompt)
interrupt = KeyboardInterrupt()

# load model
model = LocalLM.from_pretrained(
    args.model, 
    quant=args.quant, 
    api=args.api
)

while True: 
    # get the next prompt from the list, or from the user interactivey
    if isinstance(prompts, list):
        if len(prompts) > 0:
            user_prompt = prompts.pop(0)
            print(user_prompt, end='', flush=True)
        else:
            break
    else:
        cprint('>> PROMPT: ', 'blue', end='', flush=True)
        user_prompt = sys.stdin.readline().strip()
    
    # generate bot reply
    reply = model.generate(
        user_prompt, 
        streaming=not args.no_streaming, 
        stop_tokens=StopTokens,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
    )
        
    if args.no_streaming:
        print(reply)
    else:
        for token in reply:
            print(token, end='', flush=True)
            if interrupt:
                reply.stop()
                interrupt.reset()
                break
            
    print('\n')
    print_table(model.stats)
    print('')
