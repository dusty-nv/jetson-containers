#!/usr/bin/env python3
import os
import time
import PIL
import clip
import torch
import numpy as np

from .utils import AttrDict, load_image, download_model, print_table


class CLIPEmbedding():
    """
    CLIP feature extractor and projector for generating image embeddings.
    """
    def __init__(self, model='ViT-L/14@336px', dtype=np.float32, jit=False, **kwargs):
        """
        Parameters:
        
          model (str) -- name or path to CLIP model, one of:
                         'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        """                
        self.config = AttrDict(name=model)
        self.image_stats = AttrDict()
        self.text_stats = AttrDict()
        self.extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.stream = None
        
        print(f'-- loading CLIP {model}')
        
        self.model, self.preprocessor = clip.load(
            model, 
            device=self.device, 
            jit=jit, 
            download_root='/data/models/clip'
        )
        
        dtype = np.dtype(dtype)
        
        if dtype == np.float32:
            self.dtype = torch.float32
        elif dtype == np.float16:
            self.dtype = torch.float16
        else:
            raise ValueError(f"unsupported datatype:  {dtype}")

        #if self.dtype == torch.float16:
        #    self.model = self.model.half()
            
        self.model = self.model.eval()
        
        print(self.model)
        
        print(f'-- {self.config.name} warmup')
        self.config.input_shape = (self.model.visual.input_resolution, self.model.visual.input_resolution) if not jit else (336,336) 
        self.embed_image(PIL.Image.new('RGB', self.config.input_shape, (255,255,255)))
        print_table(self.config)
        
    def embed_image(self, image, crop=False, return_tensors='pt', **kwargs):
        """
        TODO:  return 'pooled', 'hidden', 'projected' in a dict
        """
        if isinstance(image, str):
            image = load_image(image)

        time_begin_pre = time.perf_counter()
        
        image_size = image.size
        
        #if not crop:
        #    image = image.resize(self.config.input_shape, PIL.Image.BILINEAR)
            
        with torch.cuda.StreamContext(self.stream), torch.inference_mode():
            image = self.preprocessor(image).unsqueeze(0).to(self.device, dtype=self.dtype)
            time_begin_enc = time.perf_counter()
            features = self.model.encode_image(image)
            time_end_enc = time.perf_counter()
            self.config.output_shape = features.shape

        self.image_stats.clip_time = time_end_enc - time_begin_pre
        self.image_stats.clip_rate = 1.0 / self.image_stats.clip_time
        self.image_stats.preprocess_time = time_begin_enc - time_begin_pre
        self.image_stats.encode_time = time_end_enc - time_begin_enc
        self.image_stats.input_shape = f"({image_size[0]},{image_size[1]}) -> {self.config.input_shape}"
        self.image_stats.output_shape = self.config.output_shape
        
        #print('input: ', image.shape, image.dtype, image.device)
        #print('output:', image_features.shape, image_features.dtype, image_features.device)
        
        if return_tensors == 'np':
            return features.detach().cpu().numpy()  # .squeeze
        elif return_tensors == 'pt':
            return features
        else:
            raise ValueError(f"return_tensors should be 'np' or 'pt' (was '{return_tensors}')")
        