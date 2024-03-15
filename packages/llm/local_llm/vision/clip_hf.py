#!/usr/bin/env python3
import os
import time
import logging

import cv2
import PIL
import torch
import numpy as np

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor, SiglipVisionModel
from ..utils import AttributeDict, load_image, torch_image, image_size, convert_tensor, download_model, print_table

_clip_model_cache = dict(image={}, text={})

class CLIPImageEmbedding():
    """
    CLIP feature extractor and projector for generating image embeddings.
    """
    @staticmethod
    def from_pretrained(model="openai/clip-vit-large-patch14-336", dtype=torch.float32, use_cache=True, **kwargs):
        global _clip_model_cache
        
        if use_cache and model in _clip_model_cache['image']:
            return _clip_model_cache['image'][model]
            
        inst = CLIPImageEmbedding(model, dtype=dtype, **kwargs)
        
        if use_cache:
            _clip_model_cache['image'][model] = inst
            
        return inst
    
    def __init__(self, model, dtype=torch.float32, **kwargs):
        self.stats = AttributeDict()
        self.config = AttributeDict()
        
        self.config.name = model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.stream = None
        self.dtype = dtype
        
        self.model_types = {
            'clip':  dict(preprocessor=CLIPImageProcessor, model=CLIPVisionModelWithProjection),
            'siglip': dict(preprocessor=SiglipImageProcessor, model=SiglipVisionModel),
        }
        
        for key, model_type in self.model_types.items():
            if key in model.lower():
                self.model_type = key
                break
                
        if not hasattr(self, 'model_type'):
            raise ValueError(f"tried loading vision model {model} - supported model types are CLIP and SigLIP")
            
        logging.info(f'loading {self.model_type} vision model {model}')

        self.preprocessor = model_type['preprocessor'].from_pretrained(model, torch_dtype=self.dtype)#.to(self.device)
        self.model = model_type['model'].from_pretrained(model, torch_dtype=self.dtype).to(self.device).eval()

        print(type(self.preprocessor), model, self.preprocessor)
        print(type(self.model), model, self.model)

        logging.debug(f'{self.config.name} warmup')
        self.config.input_shape = (self.model.config.image_size, self.model.config.image_size)
        self(PIL.Image.new('RGB', self.config.input_shape, (255,255,255)))
        print_table(self.config)
        
    def embed_image(self, image, crop=False, hidden_state=None, return_tensors='pt', return_dict=False, stream=None, **kwargs):
        """
        Return the encoded features from the given image in the embedding (or whatever the model output is)
        TODO:  return 'pooled', 'hidden', 'projected' in a dict
        """
        if isinstance(image, str):
            image = load_image(image)
        else:
            image = torch_image(image)
        
        time_begin_pre = time.perf_counter()

        if not crop:
            logging.debug(f"resizing image from {image.shape if hasattr(image, 'shape') else image.size} -> {self.config.input_shape}")
            if isinstance(image, PIL.Image.Image):
                image = image.resize(self.config.input_shape, PIL.Image.BILINEAR) # PIL.Image.BICUBIC
            elif isinstance(image, np.ndarray):
                image = cv2.resize(image, self.config.input_shape)
            else:
                raise TypeError(f"expected either PIL.Image or np.ndarray (was {type(image)})")
        else:
            logging.debug(f"cropping image from {image.shape if hasattr(image, 'shape') else image.size} -> {self.config.input_shape}")
            
        output = AttributeDict() if return_dict else None
        
        with torch.cuda.StreamContext(stream), torch.inference_mode():
            image = self.preprocessor(image, do_center_crop=crop, do_resize=crop, return_tensors='pt')['pixel_values']
            image = image.to(self.device, dtype=self.dtype)
            
            time_begin_enc = time.perf_counter()
            model_output = self.model(image, output_hidden_states=hidden_state is not None)   #.pooler_output  .last_hidden_state

            if hidden_state is not None:
                hidden_tensor = convert_tensor(model_output.hidden_states[hidden_state], return_tensors=return_tensors, device=self.device, dtype=self.dtype)
                if return_dict:
                    output.hidden_state = hidden_tensor
                else:
                    output = hidden_tensor
                self.config.output_shape = hidden_tensor.shape
            else:
                self.config.output_shape = model_output.image_embeds.shape
                
            if return_dict:
                #output.pooler_output = convert_tensor(model_output.pooler_output, return_tensors=return_tensors, device=self.device, dtype=self.dtype) 
                output.image_embeds = convert_tensor(model_output.image_embeds, return_tensors=return_tensors, device=self.device, dtype=self.dtype) 
            elif hidden_state is None:
                output = convert_tensor(model_output.image_embeds, return_tensors=return_tensors, device=self.device, dtype=self.dtype) 

        time_end_enc = time.perf_counter()
        
        self.stats.clip_time = time_end_enc - time_begin_pre
        self.stats.clip_rate = 1.0 / self.stats.clip_time
        self.stats.preprocess_time = time_begin_enc - time_begin_pre
        self.stats.encode_time = time_end_enc - time_begin_enc
        self.stats.input_shape = f"{image_size(image)} -> {self.model.config.image_size}x{self.model.config.image_size}"
        self.stats.output_shape = self.config.output_shape

        return output
        
    def __call__(self, image, crop=False, hidden_state=None, return_tensors='pt', **kwargs):
        return self.embed_image(image, crop=crop, hidden_state=hidden_state, return_tensors='pt', **kwargs)
        