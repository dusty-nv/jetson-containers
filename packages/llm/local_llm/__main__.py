#!/usr/bin/env python3
import os
import sys
import time
import signal
import argparse
import numpy as np

from termcolor import cprint

from local_llm import LocalLM, ChatHistory, ChatTemplates, CLIPModel, load_image, load_prompts, print_table, ImageExtensions


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", type=str, required=True, help="path to the model, or repository on HuggingFace Hub")
parser.add_argument("--quant", type=str, default=None, help="path to the quantized weights (AWQ uses this)")
parser.add_argument("--api", type=str, default=None, choices=['auto_gptq', 'awq', 'hf', 'mlc'], help="specify the API to use (otherwise inferred)")

parser.add_argument("--vision-model", type=str, default=None, help="for VLMs, manually select the CLIP vision model to use (e.g. openai/clip-vit-large-patch14-336 for higher-res)")

parser.add_argument("--prompt", action='append', nargs='*', help="add a prompt (can be prompt text or path to .txt, .json, or image file)")
parser.add_argument("--system", action='append', nargs='*', help="set the system prompt instruction")
parser.add_argument("--chat", action="store_true", help="enabled chat mode (automatically enabled if 'chat' in model name)")
parser.add_argument("--chat-template", type=str, default=None, choices=list(ChatTemplates.keys()), help="manually select the chat template")

parser.add_argument("--no-streaming", action="store_true", help="disable streaming output (text output will appear all at once)")
parser.add_argument("--max-new-tokens", type=int, default=128, help="the maximum number of new tokens to generate, in addition to the prompt")

args = parser.parse_args()

if 'chat' in args.model:
    args.chat = True
    
# populate default prompts - https://modal.com/docs/guide/ex/vllm_inference
if args.prompt:
    args.prompt = [x[0] for x in args.prompt]
    if args.prompt[0] == 'default' or args.prompt[0] == 'defaults':
        if args.chat:
            args.prompt = [
                "What is the weather forecast today?",
                "What is the fable involving a fox and grapes?",
                "What's a good recipe for making tabouli?",
                "How do I allocate memory in C?",
                "Implement a Python function to compute the Fibonacci numbers.",
                "What is the product of 9 and 8?",
                "Is Pluto really a planet or not?",
                "When was the Hoover Dam built?",
                "What's a training plan to run a marathon?",
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

if args.system:
    args.system = [x[0] for x in args.system]
    
print(args)

# load .txt or .json files
prompts = load_prompts(args.prompt) if args.prompt else None
system_prompt = ' '.join(load_prompts(args.system)) if args.system else None

# load model
model = LocalLM.from_pretrained(
    args.model, 
    quant=args.quant, 
    api=args.api,
    vision_model=args.vision_model
)

print_table(model.config)

# create the chat history
chat_history = ChatHistory(model, args.chat_template, system_prompt)

# make an interrupt handler for muting the bot output
last_interrupt = 0.0
interrupt_chat = False

def on_interrupt(signum, frame):
    """
    Ctrl+C handler - if done once, interrupts the LLM
    If done twice in succession, exits the program
    """
    global last_interrupt
    global interrupt_chat
    
    curr_time = time.perf_counter()
    time_diff = curr_time - last_interrupt
    last_interrupt = curr_time
    
    if time_diff > 2.0:
        print("\n-- Ctrl+C:  interrupting chatbot")
        interrupt_chat = True
    else:
        while True:
            print("\n-- Ctrl+C:  exiting...")
            sys.exit(0)
            time.sleep(0.5)
               
signal.signal(signal.SIGINT, on_interrupt)


while True: 
    # get the next prompt from the list, or from the user interactivey
    if isinstance(prompts, list):
        if len(prompts) > 0:
            user_prompt = prompts.pop(0)
            cprint(f'>> PROMPT: {user_prompt}', 'blue')
        else:
            break
    else:
        cprint('>> PROMPT: ', 'blue', end='', flush=True)
        user_prompt = sys.stdin.readline().strip()
    
    # special commands:  load prompts from file
    # 'reset' or 'clear' resets the chat history
    if user_prompt.lower().endswith(('.txt', '.json')):
        user_prompt = ' '.join(load_prompts(user_prompt))
    elif user_prompt.lower() == 'reset' or user_prompt.lower() == 'clear':
            print('-- resetting chat history')
            chat_history = []
            kv_cache = None
            continue

    # add the latest user prompt to the chat history
    entry = chat_history.add_entry(role='user', input=user_prompt)

    print(f"chat entry {len(chat_history)}:  {entry}")

    # images should be followed by text prompts
    if 'image' in entry and 'text' not in entry:
        print('-- image message, waiting for user prompt')
        continue
        
    # get the latest embeddings from the chat
    embedding, position = chat_history.embed_chat()
    
    print('adding embedding', embedding.shape, 'position', position)
    
    # generate bot reply
    output = model.generate(
        embedding, 
        streaming=not args.no_streaming, 
        max_new_tokens=args.max_new_tokens,
        kv_cache=chat_history.kv_cache
    )
        
    bot_reply = chat_history.add_entry(role='bot', text='') # placeholder
    
    if args.no_streaming:
        bot_reply.text = output
        cprint(output, 'green')
    else:
        for token in output:
            bot_reply.text += token
            cprint(token, 'green', end='', flush=True)
            if interrupt_chat:
                output.stop()
                interrupt_chat = False
                break
            
    print('')
    print_table(model.stats)

    chat_history.kv_cache = output.kv_cache   # save the kv_cache 
    bot_reply.text = output.output_text  # sync the text once more
 
print('exiting...')
