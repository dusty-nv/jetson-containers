#!/usr/bin/env python3
import os
import time
import PIL
import torch
import logging

from transformers import CLIPImageProcessor, CLIPVisionModel, SiglipImageProcessor, SiglipVisionModel
from ..utils import AttributeDict, load_image, torch_image, image_size, download_model, print_table

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
            'clip':  dict(preprocessor=CLIPImageProcessor, model=CLIPVisionModel),
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
        
    def embed_image(self, image, crop=False, hidden_state=None, return_tensors='pt', stream=None, **kwargs):
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
            logging.debug(f"resizing image from {image.size} -> {self.config.input_shape}")
            image = image.resize(self.config.input_shape, PIL.Image.BILINEAR) # PIL.Image.BICUBIC
        else:
            logging.debug(f"cropping image from {image.size} -> {self.config.input_shape}")
            
        with torch.cuda.StreamContext(stream), torch.inference_mode():
            image = self.preprocessor(image, do_center_crop=crop, do_resize=crop, return_tensors='pt')['pixel_values']
            image = image.to(self.device, dtype=self.dtype)
            
            time_begin_enc = time.perf_counter()
            
            outputs = self.model(image, output_hidden_states=hidden_state is not None)   #.pooler_output  .last_hidden_state
            
            if hidden_state is not None:
                output = outputs.hidden_states[hidden_state].to(self.device, dtype=self.dtype)
            else:
                output = outputs.pooler_output.to(dtype=self.dtype)
                
            self.config.output_shape = output.shape
            
            time_end_enc = time.perf_counter()
        
        self.stats.clip_time = time_end_enc - time_begin_pre
        self.stats.clip_rate = 1.0 / self.stats.clip_time
        self.stats.preprocess_time = time_begin_enc - time_begin_pre
        self.stats.encode_time = time_end_enc - time_begin_enc
        self.stats.input_shape = f"{image_size(image)} -> {self.model.config.image_size}x{self.model.config.image_size}"
        self.stats.output_shape = self.config.output_shape

        if return_tensors == 'pt':
            return output
        elif return_tensors == 'np':
            return output.detach().cpu().numpy()
        else:
            raise ValueError(f"return_tensors should be 'np' or 'pt' (was '{return_tensors}')")
        
    def __call__(self, image, crop=False, hidden_state=None, return_tensors='pt', **kwargs):
        return self.embed_image(image, crop=crop, hidden_state=hidden_state, return_tensors='pt', **kwargs)
        