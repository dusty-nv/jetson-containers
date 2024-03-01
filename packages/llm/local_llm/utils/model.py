#!/usr/bin/env python3
import os
from glob import glob

from huggingface_hub import snapshot_download, hf_hub_download, login


def download_model(model, type='model', cache_dir='$TRANSFORMERS_CACHE', use_safetensors=False, **kwargs):
    """
    Get the local path to a cached model or file in the cache_dir, or download it from HuggingFace Hub if needed.
    If the asset is private and authentication is required, set the HUGGINGFACE_TOKEN environment variable.
    cache_dir is where the model gets downloaded to - by default, set to $TRANSFORMERS_CACHE (/data/models/huggingface)
    By default, the PyTorch .bin weights will be downloaded instead of the .safetensors (use_safetensors=False)
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
        
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=type, cache_dir=cache_dir, resume_download=True)
    else:
        repo_path = snapshot_download(repo_id=model, repo_type=type, cache_dir=cache_dir, resume_download=True, ignore_patterns=['*.safetensors', '*.gguf'])
                                      
        if glob(os.path.join(repo_path, '*model*.pt')) or glob(os.path.join(repo_path, '*model*.bin')):
            return repo_path
            
        return snapshot_download(repo_id=model, repo_type=type, cache_dir=cache_dir, resume_download=True, ignore_patterns=['*.gguf'])
    
    
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