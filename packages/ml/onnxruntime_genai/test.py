#!/usr/bin/env python3

import os
import time
import shutil
import pprint
import argparse
import requests
import numpy as np

from packaging.version import Version

import onnxruntime as ort
print('onnxruntime version: ' + str(ort.__version__))

ort_version = Version(ort.__version__)

if ort_version > Version('1.10'):
    print(ort.get_build_info())

# verify execution providers
providers = ort.get_available_providers()

print(f'execution providers:  {providers}')

print('testing onnxruntime_genai...')
import onnxruntime_genai as og
print('onnxruntime_genai version: ' + str(og.__version__))

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

print("\n onnxruntime_genai OK\n")
