#!/usr/bin/env python3
from .plugin import Plugin

class Agent():
    """
    Agents create/manage a pipeline of plugins
    """
    def __init__(self, pipeline=[], **kwargs):
        """
        pipeline should be a list of source plugins from the graph
        """
        if isinstance(pipeline, Plugin):
            self.pipeline = [pipeline]
        elif isinstance(pipeline, list):
            self.pipeline = pipeline
        else:
            raise TypeError(f"expected Plugin or list[Plugin] for 'pipeline' argument (was {type(pipeline)})")
        
    def process(self, input, channel=0, **kwargs):
        """
        Add data to the pipeline's input queue.
        channel is the index of the source plugin from the constructor.
        """
        if len(self.pipeline) == 0:
            raise NotImplementedError(f"{type(self)} has not implemented a pipeline")
            
        self.pipeline[channel].input(input, **kwargs)
        
    def __call__(self, input, channel=0, **kwargs):
        """
        Operator overload for process()
        """
        return self.process[channel](input, **kwargs)
        
    def start(self):
        """
        Start threads for all plugins in the graph that have threading enabled.
        """
        for channel in self.pipeline:
            channel.start()
        return self
        
    def run(self, timeout=None):
        """
        Run the agent forever or return after the specified timeout (in seconds)
        """
        self.start()
        self.pipeline[0].join(timeout)
        return self
        
        
def Pipeline(plugins):
    """
    Connect the `plugins` list feed-forward style where each is an input to the next.
    This uses plugin.add(), but specifying pipelines in list notation can be cleaner.
    Returns the first plugin in the pipeline, from which other plugins can be found.
    """
    if len(plugins) == 0:
        return None
        
    for i in range(len(plugins)-1):
        plugins[i].add(plugins[i+1])
        
    return [plugins[0]]