#!/usr/bin/env python3
import os
import numpy as np

from .clip import CLIPEmbedding
from .utils import print_table


class AutoEmbedding():
    """
    Multi-modal embedding table that routes incoming media (text, tokens, images, ect)
    to their respective emedding models and returns the generated embeddings.
    It can also scan directories of files and images and parse/extract embeddings from those.
    """
    def __init__(self, text=None, image="openai/clip-vit-large-patch14-336", dtype=np.float32, **kwargs):
        self.embeddings = {}
        self.extensions = []
        
        if image is not None:
            self.embeddings['image'] = CLIPEmbedding.from_pretrained(image, dtype=dtype)

        for key, value in self.embeddings.items():
            if hasattr(value, 'extensions'):
                self.extensions.extend(value.extensions)
                
    def embed(self, data, type):
        if type not in self.embeddings:
            raise RuntimeError(f"AutoEmbedding was not loading with embedding model for type '{type}'")
            
    def embed_text(self, text):
        return self.embed(text, type='text')
        
    def embed_tokens(self, tokens):
        return self.embed(tokens, type='tokens')
        
    def embed_image(self, image):
        return self.embed(image, type='image')
    
    
        