#!/usr/bin/env python3
print('testing bitsandbytes...')

import bitsandbytes

print('bitsandbytes OK\n')

import sys
import os.path
if not os.path.isfile('/data/models/huggingface/token'):
    print("###################################")
    print("### ATTENTION: TEST DID NOT RUN ###")
    print("###################################")
    print("--> This test requires HuggingFace token")
    print("--> Check if you have saved /data/models/huggingface/token")
    sys.exit()

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# HF   meta-llama/Llama-2-7b-chat-hf
# GPTQ TheBloke/Llama-2-7B-Chat-GPTQ
# AWQ  TheBloke/Llama-2-7B-Chat-AWQ
#
# HF   TinyLlama/TinyLlama-1.1B-Chat-v1.0
# GPTQ TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ
# AWQ  TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ
model_name='meta-llama/Llama-2-7b-chat-hf'

model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextIteratorStreamer(tokenizer)

prompt = [{'role': 'user', 'content': 'Can I get a recipe for French Onion soup?'}]
inputs = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
).to(model.device)

Thread(target=lambda: model.generate(inputs, max_new_tokens=256, streamer=streamer)).start()

for text in streamer:
    print(text, end='', flush=True)
