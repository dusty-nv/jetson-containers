#!/usr/bin/env python3
from local_llm import Plugin

class Callback(Plugin):
    """
    Wrapper for calling a function with the same signature as Plugin.process()
    This is automatically used by Plugin.add() so it's typically not needed.
    Callbacks are threaded by default and will be run asynchronously.
    If it's a lightweight non-blocking function, you can set threaded=False
    """
    def __init__(self, function, threaded=False, **kwargs):
        """
        Parameters:
          function (callable) -- function for processing data like Plugin.process() would
        """
        super().__init__(threaded=threaded, **kwargs)
        self.function = function
        
    def process(self, input, **kwargs):
        return self.function(input, **kwargs)