#!/usr/bin/env python3
import os
import sys
import time
import datetime
import resource
import requests
import argparse
import threading
import socket
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from io import BytesIO

from transformers import TextIteratorStreamer

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--model-name", type=str, default=None)
parser.add_argument("--image-file", type=str, default="/data/images/hoover.jpg")

parser.add_argument("--prompt", action='append', nargs='*')

parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=64)

parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")

parser.add_argument('--runs', type=int, default=2, help="Number of inferencing runs to do (for timing)")
parser.add_argument('--warmup', type=int, default=1, help='the number of warmup iterations')
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')

args = parser.parse_args()

if not args.prompt:
    args.prompt = [
        "What does the sign in the image say?",
        "How far is the exit?",
        "What kind of environment is it in?",
        "Does it look like it's going to rain?",
    ]
    
print(args)    


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_max_rss():  # peak memory usage in MB (max RSS - https://stackoverflow.com/a/7669482)
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  
    
    
disable_torch_init()

model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

if 'llama-2' in model_name.lower():
    conv_mode = "llava_llama_2"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

if args.conv_mode is not None and conv_mode != args.conv_mode:
    print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
else:
    args.conv_mode = conv_mode

print(image_processor)

avg_encoder=0
avg_latency=0
avg_tokens_sec=0

for run in range(args.runs + args.warmup):
    conv = conv_templates[args.conv_mode].copy()
    
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    
    image = load_image(args.image_file)
    
    time_begin=time.perf_counter()
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    time_encoder=time.perf_counter() - time_begin
        
    print(f"{image_processor.feature_extractor_type} encoder:  {time_encoder:.3f} seconds\n")
        
    if run >= args.warmup:
        avg_encoder += time_encoder
        
    for inp in args.prompt:
        print(f"{roles[0]}: ", inp)
        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        #tokenizer_begin=time.perf_counter()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        #print(f"tokenizer:  {time.perf_counter()-tokenizer_begin:.3f} seconds")
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        def generate():
            with torch.inference_mode():
                model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
            
        thread = threading.Thread(target=generate)
        thread.start()

        new_tokens = ''
        num_tokens = 0
        time_begin = time.perf_counter()
        
        for token in streamer:
            print(token, end='')
            sys.stdout.flush()
            
            if num_tokens == 0:
                time_first_token=time.perf_counter()
                latency=time_first_token - time_begin
                time_begin=time_first_token
                    
            new_tokens += token
            num_tokens += 1

        print('\n')

        conv.messages[-1][-1] = new_tokens
        
        #outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        #print(outputs)
        #conv.messages[-1][-1] = outputs
        
        time_elapsed=time.perf_counter() - time_begin
        tokens_sec=(num_tokens-1) / time_elapsed
        print(f"{model_name}:  {num_tokens} tokens in {time_elapsed:.2f} sec, {tokens_sec:.2f} tokens/sec, latency {latency:.2f} sec\n")
            
        if run >= args.warmup:
            avg_latency += latency
            avg_tokens_sec += tokens_sec

avg_encoder /= args.runs
avg_latency /= args.runs * len(args.prompt)
avg_tokens_sec /= args.runs * len(args.prompt)

memory_usage=get_max_rss()

print(f"AVERAGE of {args.runs} runs:")
print(f"{model_name}:  encoder {avg_encoder:.3f} sec, {avg_tokens_sec:.2f} tokens/sec, latency {avg_latency:.2f} sec, memory {memory_usage:.2f} MB")

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, model, encoder, tokens/sec, latency, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
        file.write(f"llava, {model_name}, {avg_encoder}, {avg_tokens_sec}, {avg_latency}, {memory_usage}\n")
        