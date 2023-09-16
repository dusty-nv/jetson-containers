#!/usr/bin/env python3
import os
import time
import json
import tabulate

from termcolor import cprint
from tabulate import tabulate

from huggingface_hub import snapshot_download, hf_hub_download, login


class LocalLM():
    """
    Base class for local LLM APIs. It defines common Huggingface-like interfaces for
    model loading, text generation, chat, tokenization/detokenization, and streaming.
    
    Supported API backends include: llama.cpp, exllama2, AutoGPTQ, AWQ, and MLC
    
    Use LocalLM.from_pretrained() rather than instantiating this class directly.
    """
    @staticmethod
    def from_pretrained(model, quant=None, api=None, **kwargs):
        """
        Load a model from the given path or download it from HuggingFace Hub.
        If the API isn't specified, it will be inferred from the type of model.
        """
        if os.path.isdir(model) or os.path.isfile(model):
            model_path = model
            model_name = os.path.basename(model_path)
        else:
            model_path = LocalLM.download(model)
            model_name = model
            
        if not api:
            api = LocalLM.determine_model_api(model_path)
            
        print(f"-- loading {model_path} with {api}")
        load_begin = time.perf_counter()
        
        if api == 'auto_gptq':
            from .auto_gptq import AutoGPTQModel
            model = AutoGPTQModel(model_path, **kwargs)
        elif api == 'awq':
            from .awq import AWQModel
            model = AWQModel(model_path, quant, **kwargs)
        elif api == 'mlc':
            from .mlc import MLCModel
            model = MLCModel(model_path, **kwargs)
        else:
            raise ValueError(f"invalid API: {api}")
        
        if 'name' not in model.config:
            model.config.name = model_name
            
        model.config.api = api
        
        model.print_config(extras=['load_time', f"{time.perf_counter()-load_begin:.2f} sec"])
        
        return model
     
    def generate(self, inputs, streaming=True, **kwargs):
        """
        Generate output from input tokens or text.
        
        Parameters:
          inputs (str|list[int]|torch.Tensor) -- the prompt string or tokens
          streaming (bool) -- if true (default), an iterator will be returned that outputs
                              one token at a time.  Otherwise, return the full response.
          kwargs -- see https://huggingface.co/docs/transformers/main/en/main_classes/text_generation  
          
        Returns:
          If streaming is true, an iterator is returned that provides one decoded token string at a time.
          Otherwise, a string containing the full reply is returned after it's been completed.
        """
        raise NotImplementedError("use LLM.from_pretrained() as opposed to instantiating an LLM object directly")
    
    def print_config(self, extras=None):
        """
        Print the model config in a table.
        """
        table = []
        
        for key, value in self.config.items():
            table += [[key, value]]
            
        if extras:
            if isinstance(extras[0], list):
                table.extend(extras)
            else:
                table.append(extras)
            
        cprint(tabulate(table, tablefmt='simple_grid', numalign='center'), 'green')
     
    def print_stats(self, extras=None):
        """
        Print generation and performance stats.
        """
        table = []
        
        for key, value in self.stats.items():
            table += [[key, value]]
            
        if extras:
            if isinstance(extras[0], list):
                table.extend(extras)
            else:
                table.append(extras)
            
        cprint(tabulate(table, tablefmt='simple_grid', numalign='center'), 'green')
        
    @staticmethod
    def download(model, type='model', cache_dir=None):
        """
        Download a model or file from Huggingface Hub, returning the local path.
        If the asset is private and authentication is required, set the HUGGINGFACE_TOKEN environment variable.
        """
        token = os.environ.get('HUGGINGFACE_TOKEN', os.environ.get('HUGGING_FACE_HUB_TOKEN'))
        
        if token:
            login(token=token)
           
        if not cache_dir:
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
    
    @staticmethod
    def determine_model_api(model_path):
        """
        Given the local path to a model, determine the type of API to use to load it.
        TODO check the actual model files / configs instead of just parsing the paths
        """
        model_path = model_path.lower()
        
        if 'ggml' in model_path or 'ggml' in model_path:
            return 'llama.cpp'
        elif 'gptq' in model_path:
            return 'auto_gptq'  # 'exllama'
        elif 'awq' in model_path:
            return 'awq'
        elif 'mlc' in model_path:
            return 'mlc'
    
    def __init__(self):
        """
        @internal this is down here because it should only be used by inherited classes.
        """
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self
        
        self.config = AttrDict()
        self.stats = AttrDict()
        
        self.config.name = ''
        self.config.api = ''
        

def load_prompts(prompts):
    """
    Load prompts from a list of txt or json files
    (or if these are strings, just return the strings)
    """
    prompt_list = []
    
    for prompt in prompts:
        ext = os.path.splitext(prompt)[1]
        
        if ext == '.json':
            with open(prompt) as file:
                json_prompts = json.load(file)
            for json_prompt in json_prompts:
                if isinstance(json_prompt, dict):
                    prompt_list.append(json_prompt['text'])
                elif ifinstance(json_prompt, str):
                    prompt_list.append(json_prompt)
                else:
                    raise TypeError(f"{type(json_prompt)}")
        elif ext == '.txt':
            with open(prompt) as file:
                prompt_list.append(file.read())
        else:
            prompt_list.append(prompt)
            
    return prompt_list
    
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--model", type=str, required=True, help="path to the model, or repository on HuggingFace Hub")
    parser.add_argument("--quant", type=str, default=None, help="path to the quantized weights (AWQ uses this)")
    parser.add_argument("--api", type=str, default=None, choices=['llama.cpp', 'exllama', 'auto_gptq', 'awq', 'mlc'], help="specify the API to use (otherwise inferred)")
    parser.add_argument("--prompt", action='append', nargs='*')
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="the maximum number of new tokens to generate, in addition to the prompt")
    
    args = parser.parse_args()
    
    if not args.prompt:
        if args.chat:  # https://modal.com/docs/guide/ex/vllm_inference
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
    else:
        args.prompt = [x[0] for x in args.prompt]
        
    print(args)
    
    prompts = load_prompts(args.prompt)
    
    model = LocalLM.from_pretrained(args.model, quant=args.quant, api=args.api)
    
    for prompt in prompts:
        cprint(prompt + ' ', 'blue', end='', flush=True)

        output = model.generate(prompt, streaming=args.streaming, max_new_tokens=args.max_new_tokens)
        
        if args.streaming:
            for token in output:
                print(token, end='', flush=True)
        else:
            print(output)
            
        print('')
        model.print_stats()
    