#!/usr/bin/env python3
 
class Agent():
    """
    Agents create/manage a pipeline of plugins
    """
    def __init__(self, pipeline=None, **kwargs):
        """
        pipeline should be the first plugin instance in the graph, or None
        """
        self.pipeline = pipeline
        
    def process(self, input, **kwargs):
        """
        Add data to the pipeline's input queue.
        """
        if self.pipeline is None:
            raise NotImplementedError(f"{type(self)} has not implemented a pipeline")
            
        self.pipeline.input(input, **kwargs)
        
    def __call__(self, input, **kwargs):
        """
        Operator overload for process()
        """
        return self.process(input, **kwargs)
        
    def start(self):
        """
        Start threads for all plugins in the graph that have threading enabled.
        """
        self.pipeline.start()
        return self
        
    def run(self, timeout=None):
        """
        Run the agent forever or return after the specified timeout (in seconds)
        """
        self.start()
        self.pipeline.join(timeout)
        return self
        
        
def Pipeline(plugins):
    """
    Connect the plugins from the list together where each is an input to the next.
    This uses plugin.add(), but specifying pipelines in list notation can be cleaner.
    Returns the first plugin in the pipeline from which other plugins can be found.
    """
    if len(plugins) == 0:
        return None
        
    for i in range(len(plugins)-1):
        plugins[i].add(plugins[i+1])
        
    return plugins[0]