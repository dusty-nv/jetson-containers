#!/usr/bin/env python3
import os
import math
import time
import pprint
import argparse
import torch
import numpy as np

from .embedding import AutoEmbedding
from .vector_index import cudaVectorIndex, DistanceMetrics


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--scan", action='append', nargs='*', help="a directory or file to extract embeddings from")
parser.add_argument('--max-embeddings', type=int, default=512, help="the maximum number of embeddings in the database")
parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16'], help='datatype of the vectors')
parser.add_argument('--metric', type=str, default='l2', choices=DistanceMetrics, help='the distance metric to use during search')

parser.add_argument('-k', type=int, default=4, help='the number of search results to return per query')

parser.add_argument('--seed', type=int, default=1234, help='change the random seed used')
parser.add_argument('--use-faiss', action='store_true')
parser.add_argument('--do-projection', action='store_true')
parser.add_argument('--crop', action='store_true')

args = parser.parse_args()

if args.scan:
    args.scan = [x[0] for x in args.scan]
    
print(args)

class NanoDB:
    def __init__(self, path=None, model=None, reserve_size=1024, max_search_queries=1, dtype=np.float32, metric='cosine'):
        self.path = path
        self.dtype = dtype
        self.reserve = reserve
        
        self.model = AutoEmbedding(dtype=dtype) if model is None else model
        dim = self.model.embeddings['image'].config.output_shape[-1]
        self.index = cudaVectorIndex(dim, reserve, max_search_queries, dtype=dtype, metric=metric)
   
    def search(self, query, k=4):
        """
        Queries can be text (str or list[str]), tokens (list[int], ndarray[int] or torch.Tensor[int])
        or images (filename or list of filenames, PIL image or a list of PIL images)
        """
        embedding = self.model.embed(query)
        indexes, distances = self.index.search(embedding, k=k)
        return indices, distances
        
    def scan(self, path, max_embeddings=None, **kwargs):
        if os.path.isfile(path):
            return [load(path)]
            
        if max_embeddings is None:
            max_embeddings = self.index.reserved - self.index.shape[0]
            
        entries = sorted([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn])
        files = []
        
        if len(entries) > max_embeddings:
            entries = entries[:max_embeddings]
        
        for entry in entries:
            if not os.path.isfile(entry):
                continue

            if os.path.splitext(entry)[1].lower() in self.model.extensions:
                files.append(entry)
         
        embeddings = []
        metadata = []
        
        for file in files:
            embeddings.append(self.extract(file, **kwargs))
            metadata.append(file)
            
        return embeddings, metadata
        
    def extract(self, path, **kwargs):
        if os.path.isdir(path):
            return scan(path)
          
        if not os.path.isfile(path):
            raise IOError(f"file '{path}' doesn't exist")
            
        ext = os.path.splitext(path)[1].lower()
        
        for key, embedder in self.model.embeddings.items():
            if hasattr(value, 'extensions') and ext in embedder.extensions:
                print(f"-- generating embedding for {path} with {type(embedder)}")
                embedding = embedder.embed(path, **kwargs)
                if hasattr(embedder, 'stats'):
                    print_table(embedder.stats)
                return embedding
                
        return RuntimeError(f"could not find embedding model for {path}")
        
np.random.seed(args.seed)
dtype = np.dtype(args.dtype)

embedding_model = AutoEmbedding(dtype=dtype)
embedding_shape = embedding_model.embeddings['image'].config.hidden_shape if args.do_projection else embedding_model.embeddings['image'].config.pooler_shape
embedding_elements = math.prod(embedding_shape)

print(f"-- embedding elements ({embedding_shape}) -> {embedding_elements} ({dtype})")

embeddings = []
metadata = []


for path in args.scan:
    print(f"-- scanning {path}")
    emb, meta = embedding_model.scan(path, 
        max_embeddings=args.max_embeddings,
        do_projection=args.do_projection,
        crop=args.crop,
        return_tensors='np' if args.use_faiss else 'pt'
    )
    embeddings.extend(emb)
    metadata.extend(meta)

if args.use_faiss: 
    index = cudaVectorIndex(embedding_elements, args.max_embeddings, 1, dtype=dtype, metric=args.metric)

    for i in range(len(embeddings)):
        embeddings[i].shape = (embedding_elements)  # flatten
        #if args.metric == 'inner_product':
        #    print(f"{i} norm {np.linalg.norm(embeddings[i])}")
        #    embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
    
    print(f"-- adding {len(embeddings)} embeddings to index")

    for embedding in embeddings:
        index.add(embedding)

    print(f"-- validating index")
    index.validate(k=args.k)

    for i, embedding in enumerate(embeddings):
        indexes, distances = index.search(embedding, k=args.k)
        print(f"-- search results for {i} {metadata[i]}")
        for k in range(args.k):
            print(f"   * {indexes[k]} {metadata[indexes[k]]}  dist={distances[k]}")

else:
    # https://ai.stackexchange.com/questions/36191/how-to-calculate-a-meaningful-distance-between-multidimensional-tensors
    # https://ai.stackexchange.com/questions/37233/similarities-between-2d-vectors-to-flatten-or-to-not
    def cumsum_3d(a):
        a = torch.cumsum(a, -1)
        a = torch.cumsum(a, -2)
        a = torch.cumsum(a, -3)
        return a

    def norm_3d(a):
        return a / torch.sum(a, dim=(-1,-2,-3), keepdim=True)

    def emd_3d(a, b):
        a = norm_3d(a)
        b = norm_3d(b)
        return torch.mean(torch.square(cumsum_3d(a) - cumsum_3d(b)), dim=(-1,-2,-3))
    
    def cumsum_2d(a):
        a = torch.cumsum(a, -1)
        a = torch.cumsum(a, -2)
        return a

    def norm_2d(a):
        return a / torch.sum(a, dim=(-1,-2), keepdim=True)

    def emd_2d(a, b):
        a = norm_2d(a)
        b = norm_2d(b)
        return torch.mean(torch.square(cumsum_2d(a) - cumsum_2d(b)), dim=(-1,-2))
        
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    et = torch.cat(embeddings)
    print('et', et.shape, et.dtype, et.device)
    
    for n, embedding in enumerate(embeddings):
        print(n, embedding.shape, embedding.dtype, type(embedding))
        
        distances = cos(embedding, et)
        """
        distances = torch.zeros((len(embeddings)), dtype=torch.float32, device='cuda:0')
        
        print(f"distances:  {distances.shape}  {distances.dtype}")
        
        for m, embedding2 in enumerate(embeddings):
            distances[m] = emd_2d(embedding, embedding2)
            
        print(distances)
        """
        
        indexes = torch.argsort(distances, descending=True).squeeze()
        
        print(f"-- search results for {n} {metadata[n]}")
        for k in range(args.k):
            print(f"   * {indexes[k]} {metadata[indexes[k]]}  dist={distances[indexes[k]]}")
        
"""    
print(f"-- searching {queries.shape} vectors (metric={args.metric}, k={args.k})")
time_begin=time.perf_counter()
#for i in range(3):
indexes, distances = index.search(queries, k=args.k)
time_end=time.perf_counter()

print(indexes)
print(distances)
print(f"time:  {(time_end-time_begin)*1000} ms")
"""
