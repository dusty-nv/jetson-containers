#!/usr/bin/env python3
import os
import re
import json
import torch
import logging
import safetensors

from transformers import AutoConfig

from local_llm.utils import download_model


class MMProjector():
    """
    Multimodal projector MLP used by Llava and other Vision-Language Models
    to map from CLIP vision embedding space to the LLM's word embedding space.
    """
    @staticmethod
    def from_pretrained(model, dtype=torch.float16):
        """
        Load the projector from the HuggingFace Transformers model (Llava)
        
        If the model directory doesn't already have mm_projector.bin, its
        weights will be extracted from the main model (and saved there)
        
        Parameters:
        
          model (str) -- either the path to the model, or HuggingFace model repo/name
                         (e.g. liuhaotian/llava-v1.5-13b)
                         
          dtype (dtype) -- use either torch.float32 or torch.float16 weights
        """
        from local_llm import LocalLM
        
        if isinstance(model, LocalLM):
            return MMProjector(model.model_path, model.config, dtype)
        elif isinstance(model, str):
            if not os.path.isdir(model):
                model = download_model(model)
            return MMProjector(model)
        else:
            raise ValueError(f"model should either be a string containing the path or name of the HuggingFace model, or a LocalLM model instance")
            
    def __init__(self, model_path, config=None, dtype=torch.float16):
        """
        Create the mm_projector network and load its weights
        """
        if config:
            self.config = config
        else:
            self.config = AutoConfig.from_pretrained(model_path)
            
        self.model_path = model_path
        self.type = 'linear'
        self.dtype = dtype
        
        if hasattr(self.config, 'mm_projector_type'):
            self.type = self.config.mm_projector_type

        # either a variable-depth MLP, or single linear layer
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.type)
        
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(torch.nn.GELU())
                modules.append(torch.nn.Linear(self.config.hidden_size, self.config.hidden_size))
            self.model = torch.nn.Sequential(*modules)
        elif self.type == 'linear':
            self.model = torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)
        else:
            raise RuntimeError(f"Unknown vision mm_projector type: {self.type}")
            
        # load projector weights, extracting from the original model if needed
        self.weights_path = os.path.join(self.model_path, 'mm_projector.bin')
        
        if not os.path.isfile(self.weights_path):
            weight_indexes = [
                os.path.join(self.model_path, 'pytorch_model.bin.index.json'),
                os.path.join(self.model_path, 'model.safetensors.index.json'),
            ]
            
            for weight_index in weight_indexes:
                if os.path.isfile(weight_index):
                    break

            if not os.path.isfile(weight_index):
                raise ValueError(f"could not find model weight map at any of these locations:  {weight_indexes}")
                
            with open(weight_index, 'r') as file:
                weight_map = json.load(file)['weight_map']
                
            for key, value in weight_map.items():
                if 'mm_projector' in key:
                    weights_path = os.path.join(self.model_path, value)
                    break
                    
            logging.debug(f"extracting mm_projector weights from {weights_path}")
            
            if 'safetensors' in weight_index:
                weights = safetensors.torch.load_file(weights_path, device='cpu')
            else:
                weights = torch.load(weights_path, map_location='cpu')
                
            weights = {k : v for k, v in weights.items() if 'mm_projector' in k}
            
            logging.debug(f"saving mm_projector weights to {self.weights_path}")
            torch.save(weights, self.weights_path)
          
        logging.info(f"loading mm_projector weights from {self.weights_path}")
        
        mm_projector_weights = torch.load(self.weights_path, map_location='cpu')
        mm_projector_weights = {k.replace('model.mm_projector.', ''):v for k,v in mm_projector_weights.items()}  
        
        self.model.load_state_dict(mm_projector_weights)
        self.model.to(dtype=self.dtype, device='cuda:0').eval()
        
        print("mm_projector", self.model)
    
    def __call__(self, *args, **kwargs):
        """
        Forward-pass call to the model
        """
        with torch.inference_mode():
            return self.model(*args, **kwargs)
