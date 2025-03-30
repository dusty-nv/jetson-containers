#!/usr/bin/env python3
import argparse

from llama_cpp import Llama

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
parser.add_argument('-m', '--model', type=str, default='', required=True, help="path to the GGML .bin model")
parser.add_argument('-p', '--prompt', action='append', nargs='*')
#parser.add_argument('--add-bos', action='store_true')

args = parser.parse_args()

if args.prompt:
    args.prompt = [x[0] for x in args.prompt]
else:
    args.prompt = []
    
print(args)

llm = Llama(model_path=args.model, 
        n_gpu_layers=999,
        vocab_only=True,
        verbose=True)

def test_tokenizer(text, add_bos=False, checks=[]):
    print(f"\nPrompt:\n{text}")
    tokens = llm.tokenize(text.encode('utf-8'), add_bos=add_bos)
    print(f"\nTokens:\n{tokens}")
    detokenized = llm.detokenize(tokens).decode('utf-8', errors='ignore')
    print(f"\nDetokenized:\n{detokenized}")
    
    for check in checks:
        if tokens[check[0]] != check[1]:
            raise RuntimeError(f"token {check[0]} has unexpected value (actual={tokens[check[0]]} expected={check[1]})")
            
    if not add_bos and detokenized != text:
        raise RuntimeError(f"prompt doesn't match detokenized string")
    

for prompt in args.prompt:
    test_tokenizer(text)
    
test_tokenizer("Once upon a time,", add_bos=True)

test_tokenizer(" <s>", checks=[[0,29871], [1,1]])
test_tokenizer("<s>", checks=[[0,1]])
test_tokenizer("<s></s>", checks=[[0,1], [-1,2]])
test_tokenizer("<s> </s>", checks=[[0,1], [-1,2]])
test_tokenizer("<unk>", checks=[[0,0]])

prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden! What should I do? [/INST]"""

test_tokenizer(prompt, checks=[[0,1]])

prompt += " I'm not sure, that's quite the conundrum! </s>"

test_tokenizer(prompt, checks=[[0,1], [-1,2]])

prompt += "<s>[INST] It's eating all the plant's in my garden! [/INST]"

test_tokenizer(prompt, checks=[[0,1]])

prompt = """<s>[INST] <<SYS>>
Answer the questions.
<</SYS>>

Hello, how are you today? [/INST] Hello! I'm doing well, thank you for asking! How about you? Is there anything on your mind that you would like to talk about or ask me? I'm here to help with any questions you may have. </s><s>[INST] Wanna play 20 questions? [/INST]
"""

test_tokenizer(prompt, checks=[[0,1]])

print("\nllama.cpp tokenizer OK")