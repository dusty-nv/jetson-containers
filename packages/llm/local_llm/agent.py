#!/usr/bin/env python3
import logging

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
            
        self.save_mermaid = kwargs.get('save_mermaid')

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
        return self.process(input, channel, **kwargs)
        
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
        
        if self.save_mermaid:
            self.to_mermaid(save=self.save_mermaid)
            
        logging.success(f"{type(self).__name__} - system ready")
        self.pipeline[0].join(timeout)
        return self
        
    def to_mermaid(self, save=None):
        """
        Return or save mermaid diagram of the pipeline
        """
        from .utils import get_class_that_defined_method
        from .plugins import Callback
        
        nodes = []
        
        def get_node_name(plugin):
            if isinstance(plugin, Callback):
                return get_class_that_defined_method(plugin.function).__name__ \
                  + '.' + plugin.function.__name__
            return type(plugin).__name__
            
        def get_nodes(plugin):
            for node in nodes:
                if node['plugin'] == plugin:
                    return
                    
            type_name = get_node_name(plugin)
            inst_name = type_name
            
            while inst_name in [node['inst_name'] for node in nodes]:
                if inst_name[-1].isdigit():
                    inst_name = inst_name[:-1] + f"{int(inst_name[-1]) + 1}"
                else:
                    inst_name = inst_name + '_1'
                    
            if plugin.threaded:
                node_shape = ('[',']')
            else:
                node_shape = ('[[',']]')
                
            nodes.append({
                'plugin': plugin,
                'type_name': type_name,
                'inst_name': inst_name,
                'shape': node_shape,
            })
            
            for output_channel in plugin.outputs:
                for output in output_channel:
                    get_nodes(output)
         
        def find_node(plugin):
            for node in nodes:
                if node['plugin'] == plugin:
                    return node
            return None
            
        for plugin in self.pipeline:
            get_nodes(plugin)
            
        text = "---\n"
        text += f"title: {type(self).__name__}\n"
        text += "---\n"
        text += "graph\n"
        
        for node in nodes:
            text += f'{node["inst_name"]}{node["shape"][0]}"{node["type_name"]}"{node["shape"][1]}\n'
            
        for node in nodes:
            for c, output_channel in enumerate(node['plugin'].outputs):
                for output in output_channel:
                    if c == 0:
                        text += f'{node["inst_name"]} ---> {find_node(output)["inst_name"]}\n'
                    else:
                        text += f'{node["inst_name"]} -- channel {c} ---> {find_node(output)["inst_name"]}\n'
                    
        if save:
            with open(save, 'w') as file:
                file.write(text)
            logging.info(f"saved pipeline mermaid to {save}")
            
        return text
        
  
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