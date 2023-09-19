#!/usr/bin/env python3
import os
import time
import json

from .utils import download_model, default_model_api, AttrDict

class LocalLM():
    """
    Base class for local LLM APIs. It defines common Huggingface-like interfaces for
    model loading, text generation, chat, tokenization/detokenization, and streaming.
    
    Supported API backends include: AutoGPTQ, AWQ, MLC (TODO llama.cpp, exllama2)
    
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
            model_path = download_model(model)
            model_name = model
            
        if not api:
            api = default_model_api(model_path)
            
        print(f"-- loading {model_path} with {api.upper()}")
        load_begin = time.perf_counter()
        
        # doing this imports here avoid circular import, and makes it so these
        # dependencies are only needed if they are actually used to load a model
        if api == 'auto_gptq':
            from .auto_gptq import AutoGPTQModel
            model = AutoGPTQModel(model_path, **kwargs)
        elif api == 'awq':
            from .awq import AWQModel
            model = AWQModel(model_path, quant, **kwargs)
        elif api == 'mlc':
            from .mlc import MLCModel
            model = MLCModel(model_path, **kwargs)
        elif api == 'hf':
            from .hf import HFModel
            model = HFModel(model_path, **kwargs)
        else:
            raise ValueError(f"invalid API: {api}")
        
        if 'name' not in model.config:
            model.config.name = model_name
            
        model.config.api = api
        model.config.load_time = time.perf_counter() - load_begin
        
        return model
     
    def generate(self, inputs, streaming=True, **kwargs):
        """
        Generate output from input text or an embedding.
        
        Parameters:
          inputs (str|list[int]|torch.Tensor|np.ndarray) -- the prompt string or embedding
          streaming (bool) -- if true (default), an iterator will be returned that outputs
                              one token at a time.  Otherwise, return the full response.
          kwargs -- see https://huggingface.co/docs/transformers/main/en/main_classes/text_generation  
          
        Returns:
          If streaming is true, an iterator is returned that provides one decoded token string at a time.
          Otherwise, a string containing the full reply is returned after it's been completed.
        """
        raise NotImplementedError("use LLM.from_pretrained() as opposed to instantiating an LLM object directly")

    def __init__(self):
        """
        @internal this is down here because it should only be used by inherited classes.
        """
        self.config = AttrDict()
        self.stats = AttrDict()
        
        self.config.name = ''
        self.config.api = ''
