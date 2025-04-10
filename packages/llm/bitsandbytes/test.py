#!/usr/bin/env python3
import bitsandbytes
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

print('transformers version:', transformers.__version__)
#print('bitsandbytes version:', bitsandbytes.__version__)

# HF   meta-llama/Llama-2-7b-chat-hf
# GPTQ TheBloke/Llama-2-7B-Chat-GPTQ
# AWQ  TheBloke/Llama-2-7B-Chat-AWQ
#
# HF   TinyLlama/TinyLlama-1.1B-Chat-v1.0
# GPTQ TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ
# AWQ  TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ
#
# microsoft/Phi-3-mini-128k-instruct
model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
print(f'loading {model_name} with bitsandbytes (8-bit)')

model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', load_in_8bit=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextIteratorStreamer(tokenizer)

prompt = [{'role': 'user', 'content': 'Can I get a recipe for French Onion soup?'}]

if hasattr(tokenizer, 'apply_chat_template'):
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors='pt'
    ).to(model.device)
else:
    inputs = tokenizer(
        "Once upon a time, in a land far far away, ", 
        return_tensors='pt'
    ).input_ids.to(model.device)
    
Thread(target=lambda: model.generate(inputs, max_new_tokens=64, streamer=streamer)).start()

for text in streamer:
    print(text, end='', flush=True)
    
print(f'\n\ndone testing bitsandbytes with {model_name}')
