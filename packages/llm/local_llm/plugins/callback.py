#!/usr/bin/env python3
from local_llm import Plugin

class Callback(Plugin):
    """
    Wrapper for calling a function with the same signature as Plugin.process()
    This is automatically used by Plugin.add() so it's typically not needed.
    """
    def __init__(self, function, **kwargs):
        """
        Parameters:
          function (callable) -- function for processing data like Plugin.process() would
        """
        super().__init__(**kwargs)
        self.function = function
        
    def process(self, input, **kwargs):
        return self.function(input, **kwargs)