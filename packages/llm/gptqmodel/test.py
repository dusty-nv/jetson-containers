#!/usr/bin/env python3
print('testing GPTQModel...')

from gptqmodel import GPTQModel
# load Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 from modelscope
model = GPTQModel.load("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4")
result = model.generate("Uncovering deep insights begins with")[0] # tokens
print(model.tokenizer.decode(result))

print('GPTQModel OK\n')
