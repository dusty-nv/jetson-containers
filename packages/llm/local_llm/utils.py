#!/usr/bin/env python3
import os
import time
import json
import requests
import numpy as np

from PIL import Image
from io import BytesIO

from termcolor import cprint
from tabulate import tabulate

from huggingface_hub import snapshot_download, hf_hub_download, login


def load_image(path):
    """
    Load an image from a local path or URL
    """
    if path.startswith('http') or path.startswith('https'):
        print(f'-- downloading {path}')
        response = requests.get(path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        print(f'-- loading {path}')
        image = Image.open(path).convert('RGB')
        
    return image
    
    
def load_prompts(prompts):
    """
    Load prompts from a list of txt or json files
    (or if these are strings, just return the strings)
    """
    if isinstance(prompts, str):
        prompts = [prompts]
        
    prompt_list = []
    
    for prompt in prompts:
        ext = os.path.splitext(prompt)[1]
        
        if ext == '.json':
            with open(prompt) as file:
                json_prompts = json.load(file)
            for json_prompt in json_prompts:
                if isinstance(json_prompt, dict):
                    prompt_list.append(json_prompt['text'])
                elif isinstance(json_prompt, str):
                    prompt_list.append(json_prompt)
                else:
                    raise TypeError(f"{type(json_prompt)}")
        elif ext == '.txt':
            with open(prompt) as file:
                prompt_list.append(file.read())
        else:
            prompt_list.append(prompt)
            
    return prompt_list
    

def download_model(model, type='model', cache_dir='$TRANSFORMERS_CACHE'):
    """
    Get the local path to a cached model or file in the cache_dir, or download it from HuggingFace Hub if needed.
    If the asset is private and authentication is required, set the HUGGINGFACE_TOKEN environment variable.
    cache_dir is where the model gets downloaded to - by default, set to $TRANSFORMERS_CACHE (/data/models/huggingface)
    """
    token = os.environ.get('HUGGINGFACE_TOKEN', os.environ.get('HUGGING_FACE_HUB_TOKEN'))
    
    if token:
        login(token=token)
       
    if not cache_dir or cache_dir == '$TRANSFORMERS_CACHE':
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/root/.cache/huggingface')
        
    # handle either "org/repo" or individual "org/repo/file"
    # the former has 0-1 slashes, while the later has 2.
    num_slashes = 0
    
    for c in model:
        if c == '/':
            num_slashes += 1
            
    if num_slashes >= 2:  
        slash_count = 0
        
        for idx, i in enumerate(model):
            if i == '/':
                slash_count += 1
                if slash_count == 2:
                    break
                    
        repo_id = model[:idx]
        filename = model[idx+1:]
        
        repo_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=type, cache_dir=cache_dir, resume_download=True)
    else:
        repo_path = snapshot_download(repo_id=model, repo_type=type, cache_dir=cache_dir, resume_download=True)
        
    return repo_path
    
    
def default_model_api(model_path, quant_path=None):
    """
    Given the local path to a model, determine the type of API to use to load it.
    TODO check the actual model files / configs instead of just parsing the paths
    """
    if quant_path:
        quant_api = default_model_api(quant_path)
        
        if quant_api != 'hf':
            return quant_api

    model_path = model_path.lower()

    if 'ggml' in model_path or 'ggml' in model_path:
        return 'llama.cpp'
    elif 'gptq' in model_path:
        return 'auto_gptq'  # 'exllama'
    elif 'awq' in model_path:
        return 'awq'
    elif 'mlc' in model_path:
        return 'mlc'
    else:
        return 'hf'
        
        
def print_table(rows, header=None, footer=None, color='green'):
    """
    Print a table from a list[list] of rows/columns, or a 2-column dict 
    where the keys are column 1, and the values are column 2.
    
    Header is a list of columns or rows that are inserted at the top.
    Footer is a list of columns or rows that are added to the end.
    """
    if isinstance(rows, dict):
        rows = [[key,value] for key, value in rows.items()]    

    if header:
        if not isinstance(header[0], list):
            header = [header]
        rows = header + rows
        
    if footer:
        if not isinstance(footer[0], list):
            footer = [footer]
        rows = rows + footer
        
    cprint(tabulate(rows, tablefmt='simple_grid', numalign='center'), color)


def replace_text(text, dict):
    """
    Replace instances of each of the keys in dict in the text string with the values in dict
    """
    for key, value in dict.items():
        text = text.replace(key, value)
    return text    
    
    
class AttrDict(dict):
    """
    A dict where keys are available as attributes
    https://stackoverflow.com/a/14620633
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
                
  
class cudaArrayInterface():
    """
    Exposes __cuda_array_interface__ - typically used as a temporary view into a larger buffer
    https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    """
    def __init__(self, data, shape, dtype=np.float32):
        if dtype == np.float32:
            typestr = 'f4'
        elif dtype == np.float64:
            typestr = 'f8'
        elif dtype == np.float16:
            typestr = 'f2'
        else:
            raise RuntimeError(f"unsupported dtype:  {dtype}")
            
        self.__cuda_array_interface__ = {
            'data': (data, False),  # R/W
            'shape': shape,
            'typestr': typestr,
            'version': 3,
        }  