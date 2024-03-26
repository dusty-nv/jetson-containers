#!/usr/bin/env python3
import time
import queue
import threading
import logging
import traceback


class Plugin(threading.Thread):
    """
    Base class for plugins that process incoming/outgoing data from connections
    with other plugins, forming a pipeline or graph.  Plugins can run either
    single-threaded or in an independent thread that processes data out of a queue.

    Frequent categories of plugins:
    
      * sources:  text prompts, images/video
      * llm_queries, RAG, dynamic LLM calls, image postprocessors
      * outputs:  print to stdout, save images/video
      
    Parameters:
    
      output_channels (int) -- the number of sets of output connections the plugin has
      relay (bool) -- if true, will relay any inputs as outputs after processing
      drop_inputs (bool) -- if true, only the most recent input in the queue will be used
      threaded (bool) -- if true, will spawn independent thread for processing the queue.
      
    TODO:  use queue.task_done() and queue.join() for external synchronization
    """
    def __init__(self, output_channels=1, relay=False, drop_inputs=False, threaded=True, **kwargs):
        """
        Initialize plugin
        """
        super().__init__(daemon=True)

        self.relay = relay
        self.drop_inputs = drop_inputs
        self.threaded = threaded
        self.interrupted = False
        self.processing = False
        
        self.outputs = [[] for i in range(output_channels)]
        self.output_channels = output_channels
        
        if threaded:
            self.input_queue = queue.Queue()
            self.input_event = threading.Event()

    def process(self, input, **kwargs):
        """
        Abstract process() function that plugin instances should implement.
        Don't call this function externally unless threaded=False, because
        otherwise the plugin's internal thread dispatches from the queue.
        
        Plugins should return their output data (or None if there isn't any)
        You can also call self.output() directly as opposed to returning it.
        
        kwargs:
        
          sender (Plugin) -- only present if data sent from previous plugin
        """
        raise NotImplementedError(f"plugin {type(self)} has not implemented process()")
    
    def add(self, plugin, channel=0, **kwargs):
        """
        Connect this plugin with another, as either an input or an output.
        By default, this plugin will output to the specified plugin instance.
        
        Parameters:
        
          plugin (Plugin|callable) -- either the plugin to link to, or a callback
          
          mode (str) -- 'input' if this plugin should recieve data from the other
                        plugin, or 'output' if this plugin should send data to it.
                        
        Returns a reference to this plugin instance (self)
        """
        from local_llm.plugins import Callback
        
        if not isinstance(plugin, Plugin):
            if not callable(plugin):
                raise TypeError(f"{type(self)}.add() expects either a Plugin instance or a callable function (was {type(plugin)})")
            plugin = Callback(plugin, **kwargs)
            
        self.outputs[channel].append(plugin)
        
        if isinstance(plugin, Callback):
            logging.debug(f"connected {type(self).__name__} to {plugin.function.__name__} on channel={channel}")  # TODO https://stackoverflow.com/a/25959545
        else:
            logging.debug(f"connected {type(self).__name__} to {type(plugin).__name__} on channel={channel}")
            
        return self
    
    def find(self, type):
        """
        Return the plugin with the specified type by searching for it among
        the pipeline graph of inputs and output connections to other plugins.
        """
        if isinstance(self, type):
            return self
            
        for output_channel in self.outputs:
            for output in output_channel:
                if isinstance(output, type):
                    return output
                plugin = output.find(type)
                if plugin is not None:
                    return plugin
            
        return None
    
    '''
    def __getitem__(self, type):
        """
        Subscript indexing [] operator alias for find()
        """
        return self.find(type)
    '''
    
    def __call__(self, input=None, **kwargs):
        """
        Callable () operator alias for the input() function
        """
        self.input(input, **kwargs)
        
    def input(self, input=None, **kwargs):
        """
        Add data to the plugin's processing queue (or if threaded=False, process it now)
        TODO:  multiple input channels?
        """
        if self.threaded:
            #self.start() # thread may not be started if plugin only called from a callback
            if self.drop_inputs:
                configs = []
                while True:
                    try:
                        config_input, config_kwargs = self.input_queue.get(block=False)
                        if config_input is None and len(config_kwargs) > 0:  # still apply config
                            configs.append((config_input, config_kwargs))
                    except queue.Empty:
                        break
                for config in configs:
                    self.input_queue.put(config)
                    self.input_event.set()

            self.input_queue.put((input,kwargs))
            self.input_event.set()
        else:
            self.dispatch(input, **kwargs)
            
    def output(self, output, channel=0, **kwargs):
        """
        Output data to the next plugin(s) on the specified channel (-1 for all channels)
        """
        if output is None:
            return
            
        if channel >= 0:
            for output_plugin in self.outputs[channel]:
                output_plugin.input(output, **kwargs)
        else:
            for output_channel in self.outputs:
                for output_plugin in output_channel:
                    output_plugin.input(output, **kwargs)
                    
        return output
     
    @property
    def num_outputs(self):
        """
        Return the total number of output connections across all channels
        """
        count = 0
        for output_channel in self.outputs:
            count += len(output_channel) 
        return count
        
    def start(self):
        """
        Start threads for all plugins in the graph that have threading enabled.
        """
        if self.threaded:
            if not self.is_alive():
                super().start()
            
        for output_channel in self.outputs:
            for output in output_channel:
                output.start()
                
        return self
            
    def run(self):
        """
        @internal processes the queue forever when created with threaded=True
        """
        while True:
            try:
                self.input_event.wait()
                self.input_event.clear()
                
                while True:
                    try:
                        input, kwargs = self.input_queue.get(block=False)
                        self.dispatch(input, **kwargs)
                    except queue.Empty:
                        break
            except Exception as error:
                logging.error(f"Exception occurred during processing of {type(self)}\n\n{''.join(traceback.format_exception(error))}")

    def dispatch(self, input, **kwargs):
        """
        Invoke the process() function on incoming data
        """
        if self.interrupted:
            #logging.debug(f"{type(self)} resetting interrupted flag to false")
            self.interrupted = False
          
        self.processing = True
        outputs = self.process(input, **kwargs)
        self.processing = False

        self.output(outputs)
        
        if self.relay:
            self.output(input)
   
    def interrupt(self, clear_inputs=True, recursive=True, block=None):
        """
        Interrupt any ongoing/pending processing, and optionally clear the input queue.
        If recursive is true, then any downstream plugins will also be interrupted.
        If block is true, this function will wait until any ongoing processing has finished.
        This is done so that any lingering outputs don't cascade downstream in the pipeline.
        If block is None, it will automatically be set to true if this plugin has outputs.
        """
        #logging.debug(f"interrupting plugin {type(self)}  clear_inputs={clear_inputs} recursive={recursive} block={block}")
        
        if clear_inputs:
            self.clear_inputs()
          
        self.interrupted = True
        
        num_outputs = self.num_outputs
        block_other = block
        
        if block is None and num_outputs > 0:
            block = True
            
        while block and self.processing:
            #logging.debug(f"interrupt waiting for {type(self)} to complete processing")
            time.sleep(0.01) # TODO use an event for this?
        
        if recursive and num_outputs > 0:
            for output_channel in self.outputs:
                for output in output_channel:
                    output.interrupt(clear_inputs=clear_inputs, recursive=recursive, block=block_other)
                    
    def clear_inputs(self):
        """
        Clear the input queue, dropping any data.
        """
        while True:
            try:
                self.input_queue.get(block=False)
            except queue.Empty:
                return         
            