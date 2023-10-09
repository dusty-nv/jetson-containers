#!/usr/bin/env python3
import os
import sys
import time
import signal
import argparse
import numpy as np

from termcolor import cprint

from local_llm import LocalLM, ChatHistory, CLIPModel, load_image, load_prompts, print_table


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", type=str, required=True, help="path to the model, or repository on HuggingFace Hub")
parser.add_argument("--quant", type=str, default=None, help="path to the quantized weights (AWQ uses this)")
parser.add_argument("--api", type=str, default=None, choices=['auto_gptq', 'awq', 'hf', 'mlc'], help="specify the API to use (otherwise inferred)")
parser.add_argument("--prompt", action='append', nargs='*')
parser.add_argument("--system", action='append', nargs='*')
parser.add_argument("--chat", action="store_true")
parser.add_argument("--chat-template", type=str, default=None)
parser.add_argument("--no-streaming", action="store_true")
parser.add_argument("--max-new-tokens", type=int, default=128, help="the maximum number of new tokens to generate, in addition to the prompt")
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--use-embeddings", action="store_true")

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
    
if args.system:
    args.system = [x[0] for x in args.system]
    
print(args)

# load .txt or .json files
prompts = load_prompts(args.prompt) if args.prompt else None
system_prompts = load_prompts(args.system) if args.system else None

print('prompts', prompts)
print('system_prompts', system_prompts)

# load model
model = LocalLM.from_pretrained(args.model, quant=args.quant, api=args.api)

#print(model.model)
print_table(model.config)

# load image
if not args.use_embeddings:
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
            'system': '<s>[INST] <<SYS>>\n${SYSTEM_PROMPT}\n<</SYS>>\n\n',
            'turn': '${USER_MESSAGE} [/INST] ${BOT_MESSAGE} </s><s>[INST] '
        }
    }
            
    chat_template = chat_templates['llama-2']

    # system prompt
    '''
    system_prompt="""[INST] <<SYS>>
    You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
    <</SYS>>
    '''
    if system_prompts is None:
        if 'llava' in model.config.name.lower():
            system_prompt = 'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.'
        else:
            system_prompt = 'Answer the questions.'
    else:
        system_prompt = ' '.join(system_prompts)
        
    def replace_text(text, dict):
        for key, value in dict.items():
            text = text.replace(key, value)
        return text
        
    system_prompt = replace_text(
        chat_template['system'],
        {'${SYSTEM_PROMPT}': system_prompt}
    )

    print(f"system_prompt:\n```\n{system_prompt}```")
    system_embedding = model.embed_text(system_prompt, use_cache=True)
    print('system_embedding', system_embedding.shape, system_embedding.dtype)

    chat_history = []
    kv_cache = None
else: #args.chat:
    if not args.chat_template:
        if 'llama-2' in model.config.name.lower():
            if 'llava' in model.config.name.lower():
                args.chat_template = 'llava-2'
            else:
                args.chat_template = 'llama-2'
                
    chat_history = ChatHistory(model, template=args.chat_template)
    
    
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
    if isinstance(prompts, list):
        if len(prompts) > 0:
            user_prompt = prompts.pop(0)
        else:
            break
    else:
        cprint('>> PROMPT: ', 'blue', end='', flush=True)
        user_prompt = sys.stdin.readline().strip()
    
    if not args.use_embeddings:
        if user_prompt.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_embedding = get_image_embedding(user_prompt)
            continue
        elif user_prompt.lower().endswith(('.txt', '.json')):
            user_prompt = ' '.join(load_prompts(user_prompt))
        elif user_prompt.lower() == 'reset' or user_prompt.lower() == 'clear':
            print('-- resetting chat history')
            chat_history = []
            kv_cache = None
            continue
           
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
        
        print('system_embedding', system_embedding.shape, system_embedding.dtype, type(system_embedding))
        print('chat_embedding', chat_embedding.shape, chat_embedding.dtype, type(chat_embedding))
        #prompt = '\n' + prompt + ' [/INST]'

        #prompt_embedding = model.embed_text(prompt)

        if image_embedding is not None:
            embedding = (system_embedding, image_embedding, chat_embedding)
        else:
            embedding = (system_embedding, chat_embedding)
          
        print('embedding', len(embedding))
        
        embedding = np.concatenate(embedding, axis=1)
        """
        
        if len(chat_history) == 0:
            text = f"{user_prompt} [/INST]"  #'${USER_MESSAGE} [/INST] ${BOT_MESSAGE} </s><s>[INST] '
            embedding = (system_embedding, model.embed_text(text))
            embedding = np.concatenate(embedding, axis=1)
        else:
            text = f"<s>[INST] {user_prompt} [/INST]"
            
            if not chat_history[-1][1].strip().endswith('</s>'):
                print(f"-- adding EOS to bot reply")
                text = "</s>" + text
                
            embedding = model.embed_text(text)
        
        print('prompt', text)
        
        output = model.generate(
            embedding, 
            streaming=not args.no_streaming, 
            max_new_tokens=args.max_new_tokens,
            kv_cache=kv_cache
        )
        
        if args.no_streaming:
            #chat_history[-1][1] = output
            print(output)
        else:
            for token in output:
                #chat_history[-1][1] += token
                print(token, end='', flush=True)
                if interrupt_chat:
                    output.stop()
                    interrupt_chat = False
                    break
                
        print('')
        print_table(model.stats)
        
        kv_cache = output.kv_cache
        chat_history.append([user_prompt, output.output_text])
        
    else: #args.chat:
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
