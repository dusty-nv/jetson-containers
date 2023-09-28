#!/usr/bin/env python3
import os
import time
import PIL
import torch
import numpy as np

from transformers import CLIPImageProcessor, CLIPVisionModel
from .utils import AttrDict, load_image, download_model, print_table

_clip_model_cache = {}

class CLIPEmbedding():
    """
    CLIP feature extractor and projector for generating image embeddings.
    """
    @staticmethod
    def from_pretrained(model, dtype=np.float32, use_cache=True, **kwargs):
        global _clip_model_cache
        
        if use_cache and model in _clip_model_cache:
            return _clip_model_cache[model]
            
        return CLIPEmbedding(model, dtype=dtype, **kwargs)
    
    def __init__(self, model="openai/clip-vit-large-patch14-336", dtype=np.float32, **kwargs):
        self.stats = AttrDict()
        self.config = AttrDict()
        self.config.name = model
        self.extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if dtype == np.float32:
            self.dtype = torch.float32
        elif dtype == np.float16:
            self.dtype = torch.float16
        else:
            raise ValueError(f"unsupported datatype:  {self.dtype}")

        print(f'-- loading {model}')
        
        self.preprocessor = CLIPImageProcessor.from_pretrained(model, torch_dtype=self.dtype)#.to(self.device)
        self.model = CLIPVisionModel.from_pretrained(model, torch_dtype=self.dtype).to(self.device)
        
        print('CLIPImageProcessor', self.preprocessor)
        print('CLIPVisionModel', self.model)

        self.config.projector_name = os.path.join('liuhaotian/llava-llama-2-13b-chat-lightning-preview', 'mm_projector.bin')
        self.config.projector_shape = (1024, 4096)
        
        print(f'-- loading {self.config.projector_name}')
        
        projector_path = download_model(self.config.projector_name)
        projector_ckpt = torch.load(projector_path)
        
        self.mm_projector = torch.nn.Linear(*self.config.projector_shape)
        self.mm_projector.weight = torch.nn.Parameter(projector_ckpt['model.mm_projector.weight'].to(dtype=self.dtype), False)
        self.mm_projector.bias = torch.nn.Parameter(projector_ckpt['model.mm_projector.bias'].to(dtype=self.dtype), False)
        self.mm_projector = self.mm_projector.to(self.device)
        
        print('mm_projector', self.mm_projector)
        
        print(f'-- {self.config.name} warmup')
        self.config.input_shape = (self.model.config.image_size, self.model.config.image_size)
        self.embed(PIL.Image.new('RGB', self.config.input_shape, (255,255,255)))
        print_table(self.config)
        
    def embed(self, image, crop=False, do_projection=False, return_tensors='pt', **kwargs):
        """
        TODO:  return 'pooled', 'hidden', 'projected' in a dict
        """
        if isinstance(image, str):
            image = load_image(image)

        if not crop:
            image = image.resize(self.config.input_shape, PIL.Image.BICUBIC)
            
        image_size = image.size
        
        time_begin_pre = time.perf_counter()
        
        image = self.preprocessor(image, do_center_crop=crop, do_resize=crop, return_tensors='pt')['pixel_values']  # 
        image = image.to(self.device, dtype=self.dtype)
        
        time_begin_enc = time.perf_counter()
        
        with torch.inference_mode():
            image_forward_outs = self.model(image, output_hidden_states=do_projection)   #.pooler_output  .last_hidden_state
            
            if do_projection:
                image_features = self.mm_projector(
                    image_forward_outs.hidden_states[-2][:, 1:].to(self.device, dtype=self.dtype)
                )
            else:
                image_features = image_forward_outs.pooler_output.to(dtype=self.dtype)
            self.config.output_shape = image_features.shape
            
        time_end_enc = time.perf_counter()
        
        self.stats.clip_time = time_end_enc - time_begin_pre
        self.stats.clip_rate = 1.0 / self.stats.clip_time
        self.stats.preprocess_time = time_begin_enc - time_begin_pre
        self.stats.encode_time = time_end_enc - time_begin_enc
        self.stats.input_size = f"{image_size[0]}x{image_size[1]} -> {self.model.config.image_size}x{self.model.config.image_size}"
        self.stats.output_tokens = image_features.shape[1]
        
        #print('input: ', image.shape, image.dtype, image.device)
        #print('output:', image_features.shape, image_features.dtype, image_features.device)
        
        if return_tensors == 'np':
            return image_features.detach().cpu().numpy()  # .squeeze
        elif return_tensors == 'pt':
            return image_features
        else:
            raise ValueError(f"return_tensors should be 'np' or 'pt' (was '{return_tensors}')")
        