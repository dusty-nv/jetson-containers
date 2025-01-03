#!/usr/bin/env python3

import os
import time
import shutil
import pprint
import argparse
import requests
import numpy as np

from packaging.version import Version

print('testing onnxruntime-genai...')
import onnxruntime_genai as og
print('onnxruntime-genai version: ' + str(og.__version__))

model = og.Model('cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4')
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options['max_length'] = 2048
search_options['batch_size'] = 1

chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

text = "Input: Hello, I'm in jetson-containers"
if not text:
    print("Error, input cannot be empty")
    exit

prompt = f'{chat_template.format(input=text)}'

input_tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(**search_options)
generator = og.Generator(model, params)

print("Output: ", end='', flush=True)

try:
    generator.append_tokens(input_tokens)
    while not generator.is_done():
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end='', flush=True)
except KeyboardInterrupt:
    print("  --control+c pressed, aborting generation--")

print()
del generator

print("\nonnxruntime-genai OK\n")
