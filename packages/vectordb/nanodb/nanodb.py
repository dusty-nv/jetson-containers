#!/usr/bin/env python3
import os
import sys
import math
import time
import tqdm
import json
import PIL
import pprint
import torch
import numpy as np

from .clip import CLIPEmbedding
from .vector_index import cudaVectorIndex, DistanceMetrics
from .utils import print_table, tqdm_redirect_stdout

class NanoDB:
    def __init__(self, path=None, model='ViT-L/14@336px', dtype=np.float32, autosave=False, **kwargs):
        """
        kwargs:
            reserve (int) -- reseve memory (in MB) for cudaVectorIndex (default 1024)
            metric (str) -- metric for cudaVectorIndex (default 'cosine')
            max_search_queries (int) -- maximum search batch size (default 1)
            crop (bool) -- enable/disable cropping (default True)
        """
        self.path = path
        self.scans = []
        self.metadata = []
        self.autosave = autosave
        self.img_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
        
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)
     
        if model:
            self.model = CLIPEmbedding(model, dtype=dtype, **kwargs) #AutoEmbedding(dtype=dtype) if model is None else model
            dim = self.model.config.output_shape[-1]
        else:
            self.model = None
            dim = 768
            
        self.index = cudaVectorIndex(dim, dtype, **kwargs)
        
        if model:
            self.model.stream = self.index.torch_stream
        
        if path and self.get_paths(path, check_exists=True):
            self.load(path)
            
    def __len__(self):
        return len(self.index)
    
    def search(self, query, k=4):
        """
        Queries can be text (str or list[str]), tokens (list[int], ndarray[int] or torch.Tensor[int])
        or images (filename or list of filenames, PIL image or a list of PIL images)
        """
        if isinstance(query, (np.ndarray, torch.Tensor)) and len(query.shape) == 2 and query.shape[1] == self.index.shape[1]:
            embedding = query
        else:
            embedding = self.embed(query)
            
        indexes, distances = self.index.search(embedding, k=k)
        print_table(self.index.stats)
        return indexes, distances
     
    def add(self, data, metadata=None, **kwargs):
        if isinstance(data, (np.ndarray, torch.Tensor)) and len(data.shape) == 2 and data.shape[1] == self.index.shape[1]:
            embedding = data
        else:
            embedding = self.embed(data, **kwargs)
            
        index = self.index.add(embedding, sync=True)
        
        if metadata:
            self.metadata.insert(index, metadata)
            
        return index
                
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

        for file in tqdm.tqdm(files, file=sys.stdout):
            with tqdm_redirect_stdout():
                embedding = self.embed(file, **kwargs)
                index = self.index.add(embedding, sync=False)
                self.metadata.insert(index, dict(path=file))
                indexes.append(index)
                if (len(indexes) % 1000 == 0) and self.path and self.autosave:
                    self.save(self.path)
        
        time_elapsed = time.perf_counter() - time_begin
        print(f"-- added {len(indexes)} items to the index in from {path} ({time_elapsed:.1f} sec, {len(indexes)/time_elapsed:.1f} items/sec)")
        self.scans.append(path)
        
        if self.path and self.autosave:
            self.save(self.path)
            
        return indexes

    def load(self, path=None):
        if not path:
            path = self.path
            
        if os.path.splitext(path)[1]:
            raise ValueError(f"database path should be a directory, not a file (was {path})")
        
        print(f"-- loading database from {path}")
        
        self.index.sync()
        paths = self.get_paths(path, check_exists=True, raise_exception=True)

        time_begin = time.perf_counter()
        
        with open(paths['config'], 'r') as file:
            config = json.load(file)
            pprint.pprint(config)
        
        with open(paths['metadata'], 'r') as file:
            self.metadata = json.load(file)
            
        with open(paths['vectors'], 'rb') as file:
            vectors = np.fromfile(file, dtype=config['dtype'])
        
        if math.prod(config['shape']) != vectors.shape[0]:
            raise RuntimeError(f"{paths['vectors']} did not contain the expected number of elements")
        
        if config['shape'][1] != self.index.shape[1]:
            raise RuntimeError(f"{paths['vectors']} has a different vector dimension than the index was allocated with")
            
        if config['shape'][0] > self.index.reserved:
            raise RuntimeError(f"{paths['vectors']} exceeds the reserve memory that the index was allocated with")
        
        self.scans.extend(config['scans'])
        vectors.shape = config['shape']
        
        self.index.vectors.array[:vectors.shape[0]] = vectors
        self.index.shape = (vectors.shape[0], self.index.shape[1])
        
        if self.index.metric == 'l2':
            with open(paths['vector_norms'], 'rb') as file:
                vector_norms = np.fromfile(file, dtype='float32')
            if vector_norms.shape[0] != vectors.shape[0]:
                raise RuntimeError(f"{paths['vector_norms']} didn't contain the expected number of elements")
            self.index.vector_norms.array[:vector_norms.shape[0]] = vector_norms
            
        print(f"-- loaded {self.index.shape} records, {self.index.size()} bytes in {time.perf_counter()-time_begin:.2f} sec")
        
    def save(self, path=None):
        if not path:
            path = self.path
            
        if os.path.splitext(path)[1]:
            raise ValueError(f"database path should be a directory, not a file (was {path})")
            
        print(f"-- saving database to {path}")
        
        self.index.sync()
        os.makedirs(path, exist_ok=True)
        paths = self.get_paths(path)
        
        config = {
            'shape':    self.index.shape,
            'dtype':    str(self.index.dtype),
            'dsize':    self.index.dsize,
            'size':     self.index.size(),
            'reserve':  self.index.reserved_size,
            'autosave': self.autosave,
            'scans':    self.scans,
            'metric':   self.index.metric,
        }
        
        if self.model:
            config['model'] = self.model.config.name
            config['crop'] = self.model.config.crop

        time_begin = time.perf_counter()
        
        with open(paths['config'], 'w') as file:
            file.write(json.dumps(config, indent=2))
            
        with open(paths['metadata'], 'w') as file:
            file.write(json.dumps(self.metadata, indent=2))           

        if self.index.metric == 'l2':
            with open(paths['vector_norms'], 'wb') as file:
                file.write(self.index.vector_norms.array[:len(self.index)].tobytes())
                
        with open(paths['vectors'], 'wb') as file:
            bytes_written = file.write(self.index.vectors.array[:len(self.index)].tobytes())
            
        if bytes_written != self.index.size():
            raise IOError(f"failed to write all data to {path} ({bytes_written} of {self.index.size()} bytes)")
        
        print(f"-- wrote {self.index.shape} records, {self.index.size()} bytes in {time.perf_counter()-time_begin:.2f} sec")
     
    def get_paths(self, path, check_exists=False, raise_exception=False):
        paths = {
            'config': os.path.join(path, 'config.json'),
            'metadata': os.path.join(path, 'metadata.json'),
            'vectors': os.path.join(path, 'vectors.bin')
        }
        
        if self.index.metric == 'l2':
            paths['vector_norms'] = os.path.join(path, 'vector_norms.bin')
            
        if check_exists:
            for key, value in paths.items():
                if not os.path.isfile(value):
                    if raise_exception:
                        raise IOError(f"couldn't find file {value}")
                    else:
                        return None
                    
        return paths

    def embed(self, data, type=None, **kwargs):
        if type is None:
            type = self.embedding_type(data)
            #print(f"-- generating embedding for {data} with type={type}")
         
        if self.model is None:
            raise RuntimeError("nanodb was created without an embedding model")
            
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
            else:
                return "text" 
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            return 'text'
        elif isinstance(data, (PIL.Image.Image, np.ndarray, torch.Tensor)):
            return 'image'
        elif hasattr(data, '__cuda_array_interface__'):
            return 'image'
        else:
            raise ValueError(f"couldn't find type of embedding for {type(data)}, please specify the 'type' argument")
            
    def test(self, k):
        for i in tqdm.tqdm(range(len(self.index)), file=sys.stdout):
            with tqdm_redirect_stdout():
                indexes, distances = self.index.search(self.index.vectors.array[i], k=k)
                print(f"-- search results for {i} {self.metadata[i]['path']}")
                for n in range(k):
                    print(f"   * {indexes[n]} {self.metadata[indexes[n]]['path']}  {'similarity' if self.index.metric == 'cosine' else 'distance'}={distances[n]}")
                    