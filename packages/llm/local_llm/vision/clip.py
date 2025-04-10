#!/usr/bin/env python3
import os
import time
import torch
import PIL

from transformers import CLIPImageProcessor, CLIPVisionModel
from .utils import AttributeDict, load_image, download_model, print_table

_clip_model_cache = {}

class CLIPModel():
    """
    CLIP feature extractor and projector for generating image embeddings.
    """
    @staticmethod
    def from_pretrained(model="openai/clip-vit-large-patch14-336", use_cache=True, **kwargs):
        global _clip_model_cache
        
        if use_cache and model in _clip_model_cache:
            return _clip_model_cache[model]
            
        instance = CLIPModel(model, **kwargs)
        
        if use_cache:
            _clip_model_cache[model] = instance
            
        return instance
    
    def __init__(self, model="openai/clip-vit-large-patch14-336", **kwargs):
        self.stats = AttributeDict()
        self.config = AttributeDict()
        
        self.config.name = model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16
        
        print(f'-- loading {model}')
        
        self.preprocessor = CLIPImageProcessor.from_pretrained(model, torch_dtype=self.dtype)#.to(self.device)
        self.model = CLIPVisionModel.from_pretrained(model, torch_dtype=self.dtype).to(self.device)
        
        print('CLIPImageProcessor', self.preprocessor)
        print('CLIPVisionModel', self.model)

        self.config.projector_name = os.path.join('liuhaotian/llava-llama-2-13b-chat-lightning-preview', 'mm_projector.bin')
        self.config.projector_shape = (1024, 5120)  # 4096 for 7B
        
        print(f'-- loading {self.config.projector_name}')
        
        projector_path = download_model(self.config.projector_name)
        projector_ckpt = torch.load(projector_path)
        
        self.mm_projector = torch.nn.Linear(*self.config.projector_shape)
        self.mm_projector.weight = torch.nn.Parameter(projector_ckpt['model.mm_projector.weight'].to(dtype=self.dtype), False)
        self.mm_projector.bias = torch.nn.Parameter(projector_ckpt['model.mm_projector.bias'].to(dtype=self.dtype), False)
        self.mm_projector = self.mm_projector.to(self.device)
        
        print('mm_projector', self.mm_projector)
        
        print(f'-- {self.config.name} warmup')
        self.embed_image(PIL.Image.new('RGB', (self.model.config.image_size, self.model.config.image_size), (255,255,255)))
        print_table(self.config)
        
    def embed_image(self, image):
        if isinstance(image, str):
            image = load_image(image)

        image = image.resize((336,336), PIL.Image.BICUBIC)
        image_size = image.size
        
        time_begin_pre = time.perf_counter()
        
        image = self.preprocessor(image, do_center_crop=False, do_resize=False, return_tensors='pt')['pixel_values']
        image = image.to(self.device, dtype=self.dtype)
        
        time_begin_enc = time.perf_counter()
        
        with torch.inference_mode():
            image_forward_outs = self.model(image, output_hidden_states=True)
            select_hidden_state = image_forward_outs.hidden_states[-2]
            image_features = select_hidden_state[:, 1:].to(self.device, dtype=self.dtype)
            image_features = self.mm_projector(image_features)
            
        time_end_enc = time.perf_counter()
        
        self.stats.clip_time = time_end_enc - time_begin_pre
        self.stats.clip_rate = 1.0 / self.stats.clip_time
        self.stats.preprocess_time = time_begin_enc - time_begin_pre
        self.stats.encode_time = time_end_enc - time_begin_enc
        self.stats.input_size = f"{image_size[0]}x{image_size[1]} -> {self.model.config.image_size}x{self.model.config.image_size}"
        self.stats.output_tokens = image_features.shape[1]
        
        #print('input: ', image.shape, image.dtype, image.device)
        #print('output:', image_features.shape, image_features.dtype, image_features.device)
        
        return image_features.detach().cpu().numpy()  # .squeeze
        