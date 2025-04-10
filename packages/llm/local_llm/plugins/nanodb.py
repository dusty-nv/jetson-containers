#!/usr/bin/env python3
import nanodb

from local_llm import Plugin


class NanoDB(Plugin):
    """
    Plugin that loads a NanoDB database and searches it for incoming text/images   
    """
    def __init__(self, path=None, model='ViT-L/14@336px', reserve=1024, k=1, **kwargs):
        """
        Parameters:
        
          path (str) -- directory to either load or create NanoDB at
          model (str) -- the CLIP embedding model to use
          reserve (int) -- the memory to reserve for the database in MB
          kwargs (dict) -- these get passed to the NanoDB constructor
        """
        super().__init__(**kwargs)
        
        self.db = nanodb.NanoDB(
            path=path, model=model, 
            dtype='float16', metric='cosine', 
            reserve=reserve*(1<<20), crop=True, **kwargs
        )
        
        self.scans = self.db.scans
        self.k = k
        
    def process(self, input, add=False, metadata=None, k=None, **kwargs):
        """
        Search the database for the closest matches to the input.
        
        Parameters:
        
          input (str|PIL.Image) -- either text or an image to search for
                                          
        Returns:
        
          Returns a list of K search results
        """
        if not k:
            k = self.k
            
        if add:
            self.db.add(input, metadata=metadata)
        else:
            if len(self.db) == 0:
                return None
                
            indexes, similarity = self.db.search(input, k=k)
            
            return [dict(index=indexes[n], similarity=float(similarity[n]), metadata=self.db.metadata[indexes[n]])
                    for n in range(k) if indexes[n] >= 0]
        