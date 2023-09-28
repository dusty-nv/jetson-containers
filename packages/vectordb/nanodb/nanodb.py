#!/usr/bin/env python3
import os
import time
import numpy as np

from .clip import CLIPEmbedding
from .vector_index import cudaVectorIndex, DistanceMetrics
from .utils import print_table

class NanoDB:
    def __init__(self, path=None, model='ViT-L/14@336px', dtype=np.float32, reserve=1024, metric='cosine', max_search_queries=1):
        self.path = path
        self.metadata = []
        self.img_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
        
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)
            
        self.model = CLIPEmbedding(model, dtype=dtype) #AutoEmbedding(dtype=dtype) if model is None else model
        dim = self.model.config.output_shape[-1]
        self.index = cudaVectorIndex(dim, dtype, reserve, metric, max_search_queries)
        self.model.stream = self.index.torch_stream
        
    def __len__(self):
        return len(self.index)
        
    def search(self, query, k=4):
        """
        Queries can be text (str or list[str]), tokens (list[int], ndarray[int] or torch.Tensor[int])
        or images (filename or list of filenames, PIL image or a list of PIL images)
        """
        embedding = self.model.embed(query)
        indexes, distances = self.index.search(embedding, k=k)
        return indices, distances
        
    def scan(self, path, max_items=None, **kwargs):
        time_begin = time.perf_counter()
        
        if os.path.isfile(path):
            files = [path]
        elif os.path.isdir(path):
            if max_items is None:
                max_items = self.index.reserved - self.index.shape[0]
                
            entries = sorted([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn])

            if len(entries) > max_items:
                entries = entries[:max_items]
            
            files = []
            
            for entry in entries:
                if not os.path.isfile(entry):
                    continue
                if os.path.splitext(entry)[1].lower() in self.img_extensions:
                    files.append(entry)
         
        indexes = []

        for file in files:
            embedding = self.embed(file, **kwargs)
            index = self.index.add(embedding, sync=False)
            self.metadata.insert(index, file)
            indexes.append(index)
        
        time_elapsed = time.perf_counter() - time_begin
        print(f"-- added {len(indexes)} items to the index in from {path} ({time_elapsed:.1f} sec, {len(indexes)/time_elapsed:.1f} items/sec)")
        
        return indexes

    def embed(self, data, type=None, **kwargs):
        if type is None:
            type = self.embedding_type(data)
            print(f"-- generating embedding for {data} with type={type}")
                
        if type == 'image':
            embedding = self.model.embed_image(data)
            print_table(self.model.image_stats)
        elif type == 'text':
            embedding = self.model.embed_text(data)
            print_table(self.model.text_stats)
        else:
            raise ValueError(f"invalid embedding type '{type}' (should be 'image' or 'text')")

        return embedding
     
    def embedding_type(self, data):
        if isinstance(data, str):
            ext = os.path.splitext(data)[1].lower()
            if ext in self.img_extensions:
                return 'image'
            elif len(ext) > 0:
                raise ValueError(f"-- file {str} has unsupported extension for embeddings")
                
                for key, embedder in self.embeddings.items():
                    if hasattr(embedder, 'extensions') and ext in embedder.extensions:
                        return key
            else:
                return "text" 
        elif isinstance(data, PIL.Image):
            return 'image'
        else:
            raise ValueError(f"couldn't find type of embedding for {type(data)}, please specify the 'type' argument")
            
    def test(self, k):
        for i in range(len(self.index)):
            indexes, distances = self.index.search(self.index.vectors.array[i], k=k)
            print(f"-- search results for {i} {self.metadata[i]}")
            for n in range(k):
                print(f"   * {indexes[n]} {self.metadata[indexes[n]]}  dist={distances[n]}")
                