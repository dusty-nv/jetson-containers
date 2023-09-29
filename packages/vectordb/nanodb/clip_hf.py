#!/usr/bin/env python3
import os
import time
import PIL
import torch
import numpy as np

from transformers import CLIPImageProcessor, CLIPVisionModel
from .utils import AttrDict, load_image, download_model, print_table

_clip_model_cache = dict(image={}, text={})

class CLIPImageEmbedding():
    """
    CLIP feature extractor and projector for generating image embeddings.
    """
    @staticmethod
    def from_pretrained(model, dtype=np.float32, use_cache=True, **kwargs):
        global _clip_model_cache
        
        if use_cache and model in _clip_model_cache['image']:
            return _clip_model_cache['image'][model]
            
        inst = CLIPImageEmbedding(model, dtype=dtype, **kwargs)
        
        if use_cache:
            _clip_model_cache['image'][model] = inst
            
        return inst
    
    def __init__(self, model="openai/clip-vit-large-patch14-336", dtype=np.float32, **kwargs):
        self.stats = AttrDict()
        self.config = AttrDict()
        
        self.config.name = model
        self.extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.stream = None
        
        dtype = np.dtype(dtype)
        
        if dtype == np.float32:
            self.dtype = torch.float32
        elif dtype == np.float16:
            self.dtype = torch.float16
        else:
            raise ValueError(f"unsupported datatype:  {dtype}")

        print(f'-- loading {model}')
        
        self.preprocessor = CLIPImageProcessor.from_pretrained(model, torch_dtype=self.dtype)#.to(self.device)
        self.model = CLIPVisionModel.from_pretrained(model, torch_dtype=self.dtype).to(self.device)
        
        print('CLIPImageProcessor', self.preprocessor)
        print('CLIPVisionModel', self.model)

        print(f'-- {self.config.name} warmup')
        self.config.input_shape = (self.model.config.image_size, self.model.config.image_size)
        self.embed(PIL.Image.new('RGB', self.config.input_shape, (255,255,255)))
        print_table(self.config)
        
    def embed(self, image, crop=False, hidden_state=None, return_tensors='pt', **kwargs):
        """
        TODO:  return 'pooled', 'hidden', 'projected' in a dict
        """
        if isinstance(image, str):
            image = load_image(image)

        time_begin_pre = time.perf_counter()
        
        image_size = image.size
        
        if not crop:
            image = image.resize(self.config.input_shape, PIL.Image.BILINEAR)
            
        with torch.cuda.StreamContext(self.stream), torch.inference_mode():
            image = self.preprocessor(image, do_center_crop=crop, do_resize=crop, return_tensors='pt')['pixel_values']  # 
            image = image.to(self.device, dtype=self.dtype)
            
            time_begin_enc = time.perf_counter()
            
            outputs = self.model(image, output_hidden_states=hidden_state is not None)   #.pooler_output  .last_hidden_state
            
            if hidden_state is not None:
                features = outputs.hidden_states[hidden_state].to(self.device, dtype=self.dtype)
            else:
                features = outputs.pooler_output.to(dtype=self.dtype)
                
            self.config.output_shape = features.shape
            
            time_end_enc = time.perf_counter()
        
        self.stats.clip_time = time_end_enc - time_begin_pre
        self.stats.clip_rate = 1.0 / self.stats.clip_time
        self.stats.preprocess_time = time_begin_enc - time_begin_pre
        self.stats.encode_time = time_end_enc - time_begin_enc
        self.stats.input_shape = f"{image_size[0]}x{image_size[1]} -> {self.model.config.image_size}x{self.model.config.image_size}"
        self.stats.output_shape = self.config.output_shape
        
        #print('input: ', image.shape, image.dtype, image.device)
        #print('output:', image_features.shape, image_features.dtype, image_features.device)
        
        if return_tensors == 'np':
            return features.detach().cpu().numpy()  # .squeeze
        elif return_tensors == 'pt':
            return features
        else:
            raise ValueError(f"return_tensors should be 'np' or 'pt' (was '{return_tensors}')")
        