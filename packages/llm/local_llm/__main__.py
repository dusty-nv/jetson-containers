#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np

from termcolor import cprint

from local_llm import LocalLM, ChatHistory, CLIPModel, load_image, load_prompts, print_table


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", type=str, required=True, help="path to the model, or repository on HuggingFace Hub")
parser.add_argument("--quant", type=str, default=None, help="path to the quantized weights (AWQ uses this)")
parser.add_argument("--api", type=str, default=None, choices=['auto_gptq', 'awq', 'hf', 'mlc'], help="specify the API to use (otherwise inferred)")
parser.add_argument("--prompt", action='append', nargs='*')
parser.add_argument("--chat", action="store_true")
parser.add_argument("--chat-template", type=str, default=None)
parser.add_argument("--no-streaming", action="store_true")
parser.add_argument("--max-new-tokens", type=int, default=128, help="the maximum number of new tokens to generate, in addition to the prompt")
parser.add_argument("--image", type=str, default=None)

args = parser.parse_args()

if 'chat' in args.model:
    args.chat = True
    
# populate default prompts - https://modal.com/docs/guide/ex/vllm_inference
if args.prompt == 'default' or args.prompt == 'defaults':
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
elif args.prompt:
    args.prompt = [x[0] for x in args.prompt]
    
print(args)

# load .txt or .json files
prompts = load_prompts(args.prompt) if args.prompt else None

# load model
model = LocalLM.from_pretrained(args.model, quant=args.quant, api=args.api)

print(model.model)
print_table(model.config)

# load image
"""
clip = None
image_embedding = None

def get_image_embedding(image):
    global clip
    
    if clip is None:
        clip = CLIPModel()
        print_table(clip.config)
        
    if isinstance(image, str):
        image = load_image(image)
        
    embedding = clip.embed_image(image)
    print_table(clip.stats)
    #print('image_embedding', embedding.shape, embedding.dtype)
    return embedding
    
if args.image:
    image_embedding = get_image_embedding(args.image)



chat_templates = {
    'llama-2': {
        'system': '[INST] <<SYS>>\n${SYSTEM_PROMPT}\n<</SYS>>\n\n',
        'turn': '${USER_MESSAGE} [/INST] ${BOT_MESSAGE} </s><s>[INST] '
    }
}
        
chat_template = chat_templates['llama-2']
"""

# system prompt
'''
system_prompt="""[INST] <<SYS>>
You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
<</SYS>>

"""
'''
"""
if 'llava' in model.config.name.lower():
    system_prompt = 'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.'
else:
    system_prompt = 'Answer the questions.'
    
def replace_text(text, dict):
    for key, value in dict.items():
        text = text.replace(key, value)
    return text
    
system_prompt = replace_text(
    chat_template['system'],
    {'${SYSTEM_PROMPT}': system_prompt}
)

print(f"system_prompt:\n```\n{system_prompt}```")
system_embedding = model.embed_text(system_prompt)
print('system_embedding', system_embedding.shape, system_embedding.dtype)
"""

if args.chat:
    if not args.chat_template:
        if 'llama-2' in model.config.name.lower():
            if 'llava' in model.config.name.lower():
                args.chat_template = 'llava-2'
            else:
                args.chat_template = 'llama-2'
                
    chat_history = ChatHistory(model, template=args.chat_template)
    

while True: 
    if isinstance(prompts, list):
        if len(prompts) > 0:
            user_prompt = prompts.pop()
        else:
            break
    else:
        cprint('>> PROMPT: ', 'blue', end='', flush=True)
        user_prompt = sys.stdin.readline().strip()
    
    """
    if user_prompt.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image_embedding = get_image_embedding(prompt)
        continue
    """
    
    """
    chat_history.append([user_prompt, ''])
    chat_text = ""
    
    for i, turn in enumerate(chat_history):
        if not turn[1]: # latest user prompt
            text = replace_text(
                chat_template['turn'].split('${BOT_MESSAGE}')[0].rstrip(' '),
                {'${USER_MESSAGE}': turn[0]}
            )
        else: # previous query/response turn
            text = replace_text(
                chat_template['turn'],
                {'${USER_MESSAGE}': turn[0], '${BOT_MESSAGE}': turn[1]}
             )
        chat_text += text
        
    print(f"chat_text:\n```{chat_text}```")
    
    chat_embedding = model.embed_text(chat_text)
    
    #prompt = '\n' + prompt + ' [/INST]'

    #prompt_embedding = model.embed_text(prompt)

    if image_embedding is not None:
        embedding = (system_embedding, image_embedding, chat_embedding)
    else:
        embedding = (system_embedding, chat_embedding)
      
    embedding = np.concatenate(embedding, axis=1)
    """
    if args.chat:
        chat_history.add_entry(role='user', text=user_prompt)
    
        embedding = chat_history.embed_chat()
        
        output = model.generate(embedding, streaming=not args.no_streaming, max_new_tokens=args.max_new_tokens)
            
        bot_reply = chat_history.add_entry(role='bot', text='')
        
        if args.no_streaming:
            bot_reply.text = output
            print(output)
        else:
            for token in output:
                bot_reply.text += token
                print(token, end='', flush=True)
                
        print('')
        print_table(model.stats)

    
"""
for prompt in prompts:
    cprint(prompt + ' ', 'blue', end='', flush=True)

    output = model.generate(prompt, image=image_features, streaming=args.streaming, max_new_tokens=args.max_new_tokens)
    
    if args.streaming:
        for token in output:
            print(token, end='', flush=True)
    else:
        print(output)
        
    print('')
    print_table(model.stats)
"""   
print('exiting...')
