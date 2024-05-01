#!/usr/bin/env python3
import os
import sys
import time
import signal
import logging
import numpy as np

from termcolor import cprint

from local_llm import LocalLM, ChatHistory, ChatTemplates
from local_llm.utils import ImageExtensions, ArgParser, KeyboardInterrupt, load_prompts, print_table 

# see utils/args.py for options
parser = ArgParser()

parser.add_argument("--prompt-color", type=str, default='blue', help="color to print user prompts (see https://github.com/termcolor/termcolor)")
parser.add_argument("--reply-color", type=str, default='green', help="color to print user prompts (see https://github.com/termcolor/termcolor)")

parser.add_argument("--disable-streaming", action="store_true", help="wait to output entire reply instead of token by token")
parser.add_argument("--disable-stats", action="store_true", help="suppress the printing of generation performance stats")

args = parser.parse_args()

prompts = load_prompts(args.prompt)
interrupt = KeyboardInterrupt()

# load model
model = LocalLM.from_pretrained(
    args.model, 
    quant=args.quant, 
    api=args.api,
    max_context_len=args.max_context_len,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling, 
)

# create the chat history
chat_history = ChatHistory(model, args.chat_template, args.system_prompt)


while True: 
    # get the next prompt from the list, or from the user interactivey
    if isinstance(prompts, list):
        if len(prompts) > 0:
            user_prompt = prompts.pop(0)
            cprint(f'>> PROMPT: {user_prompt}', args.prompt_color)
        else:
            break
    else:
        cprint('>> PROMPT: ', args.prompt_color, end='', flush=True)
        user_prompt = sys.stdin.readline().strip()
    
    print('')
    
    # special commands:  load prompts from file
    # 'reset' or 'clear' resets the chat history
    if user_prompt.lower().endswith(('.txt', '.json')):
        user_prompt = ' '.join(load_prompts(user_prompt))
    elif user_prompt.lower() == 'reset' or user_prompt.lower() == 'clear':
        logging.info("resetting chat history")
        chat_history.reset()
        continue

    # add the latest user prompt to the chat history
    entry = chat_history.append(role='user', msg=user_prompt)

    # images should be followed by text prompts
    if 'image' in entry and 'text' not in entry:
        logging.debug("image message, waiting for user prompt")
        continue
        
    # get the latest embeddings (or tokens) from the chat
    embedding, position = chat_history.embed_chat(
        max_tokens=model.config.max_length - args.max_new_tokens,
        wrap_tokens=args.wrap_tokens,
        return_tokens=not model.has_embed,
    )
    
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"adding embedding shape={embedding.shape} position={position}")

    # generate bot reply
    reply = model.generate(
        embedding, 
        streaming=not args.disable_streaming, 
        kv_cache=chat_history.kv_cache,
        stop_tokens=chat_history.template.stop,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
    )
        
    bot_reply = chat_history.append(role='bot', text='') # placeholder
    
    if args.disable_streaming:
        bot_reply.text = reply
        cprint(reply, args.reply_color)
    else:
        for token in reply:
            bot_reply.text += token
            cprint(token, args.reply_color, end='', flush=True)
            if interrupt:
                reply.stop()
                interrupt.reset()
                break
            
    print('\n')
    
    if not args.disable_stats:
        print_table(model.stats)
        print('')
    
    chat_history.kv_cache = reply.kv_cache   # save the kv_cache 
    bot_reply.text = reply.output_text  # sync the text once more
