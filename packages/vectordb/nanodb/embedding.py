#!/usr/bin/env python3
import os
import PIL
import numpy as np

from .clip import CLIPImageEmbedding
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
        self._stream = None
        
        if image is not None:
            self.embeddings['image'] = CLIPImageEmbedding.from_pretrained(image, dtype=dtype)

        for key, value in self.embeddings.items():
            if hasattr(value, 'extensions'):
                self.extensions.extend(value.extensions)
                
    def embed(self, data, type=None, **kwargs):
        if type is None:
            type = self.embedding_type(data)
            print(f"-- generating embedding for {data} with {self.embeddings[type].__class__}")
                
        if type not in self.embeddings:
            raise RuntimeError(f"AutoEmbedding was not loading with embedding model for type '{type}'")
            
        embedding = self.embeddings[type].embed(data, **kwargs)
        
        if hasattr(self.embeddings[type], 'stats'):
            print_table(self.embeddings[type].stats)
            
        return embedding
        
    def embed_text(self, text, **kwargs):
        return self.embed(text, type='text', **kwargs)
        
    def embed_tokens(self, tokens, **kwargs):
        return self.embed(tokens, type='tokens', **kwargs)
        
    def embed_image(self, image, **kwargs):
        return self.embed(image, type='image', **kwargs)
        
    def embedding_type(self, data):
        if isinstance(data, str):
            ext = os.path.splitext(data)[1].lower()
            if ext in self.extensions:
                for key, embedder in self.embeddings.items():
                    if hasattr(embedder, 'extensions') and ext in embedder.extensions:
                        return key
            elif len(ext) > 0:
                print(f"-- warning:  couldn't find embedder for file {data} with extension {ext} (treating as text)")
            return "text" 
        elif isinstance(data, PIL.Image):
            return 'image'
        else:
            raise ValueError(f"couldn't find type of embedding for {type(data)}, please specify the 'type' argument")
    
    @property
    def stream(self):
        return self._stream
        
    @stream.setter
    def stream(self, stream):
        self._stream = stream
        for embedder in self.embeddings.values():
            embedder.stream = stream
            
        