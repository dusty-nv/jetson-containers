#!/usr/bin/env python3
import os
import time
import pickle
import logging
import threading
import multiprocessing as mp

from local_llm import Plugin


class ProcessProxy(Plugin):
    """
    Proxy wrapper for running a plugin in a subprocess
    """
    def __init__(self, plugin_factory, **kwargs):
        """
        Parameters:
          plugin_factory (callable) -- Factory function that returns a plugin instance.
                                       This will be called from the new process to create it.
        """
        self.data_parent, self.data_child = mp.Pipe(duplex=True)
        self.control_parent, self.control_child = mp.Pipe(duplex=True)
        
        mp.set_start_method('spawn')
        
        self.subprocess = mp.Process(target=self.run_process, args=(plugin_factory, kwargs))
        self.subprocess.start()
        
        init_msg = self.control_parent.recv()
        
        if init_msg['status'] != 'initialized':
            raise RuntimeError(f"subprocess has an invalid initialization status ({init_msg['status']})")

        super().__init__(output_channels=init_msg['output_channels'], **kwargs)
        logging.info(f"ProcessProxy initialized, output_channels={self.output_channels}")

    def input(self, input):
        #time_begin = time.perf_counter()
        #input_type = type(input)
        input = pickle.dumps(input, protocol=-1)
        #logging.debug(f"ProcessProxy time to pickle {input_type} - {(time.perf_counter()-time_begin)*1000} ms")
        self.data_parent.send_bytes(input)
        
    def start(self):
        self.control_parent.send('start')
        self.assert_message(self.control_parent.recv(), 'started')
        return super().start()

    def run(self):
        while True:
            output, channel = pickle.loads(self.data_parent.recv_bytes())
            #logging.debug(f"parent process recieved {type(output)} (channel={channel})")
            self.output(output, channel)

    def run_process(self, factory, kwargs):
        logging.debug(f"subprocess {os.getpid()} started")
        
        from cuda.cudart import (
            cudaSetDevice,
            cudaInitDevice,
            cudaGetLastError,
            cudaGetErrorString,
            cudaError_t
        )

        #error = cudaInitDevice(0,0,0)[0]
        
        #if error != cudaError_t.cudaSuccess:
        #    raise RuntimeError(f"cudaInitDevice() error {error} -- {cudaGetErrorString(error)[1]}")
        
        error = cudaSetDevice(0)[0]
        
        if error != cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaSetDevice() error {error} -- {cudaGetErrorString(error)[1]}")
            
        try:
            #plugin = factory(**kwargs)  # create an instance of the plugin
            
            if factory == 'ChatQuery':
                from local_llm.plugins import ChatQuery
                plugin = ChatQuery(**kwargs)
            else:
                raise TypeError(f"unsupported proxy class type {factory}")
                
        except Exception as error:
            self.control_child.send({'status': str(type(error))})
            raise error
            
        self.control_child.send({
            'status': 'initialized',
            'output_channels': plugin.output_channels,
        })
        
        # forward outputs back to parent process
        for i in range(plugin.output_channels):
            plugin.add(OutputProxy(self.data_child, i), i)
            
        # start the plugin processing
        self.assert_message(self.control_child.recv(), 'start')

        try:
            plugin.start()
        except Exception as error:
            self.control_child.send(str(type(error)))
            raise error

        # start a thread to recieve control messages from the parent
        control_thread = threading.Thread(target=self.run_control)
        control_thread.start()

        self.control_child.send('started')
        
        # recieve inputs from the parent to process
        while True:
            #time_begin = time.perf_counter()
            input = self.data_child.recv_bytes()
            input = pickle.loads(input)
            #logging.debug(f"subprocess recieved {str(type(input))} input  (depickle={(time.perf_counter()-time_begin)*1000} ms)")
            plugin(input)
    
    def run_control(self):
        while True:
            msg = self.control_child.recv()
            logging.debug(f"subprocess recieved control msg from parent process ('{msg}')")
            
    def assert_message(self, msg, expected, verbose=True):
        if msg != expected:
            raise RuntimeError(f"recieved unexpected cross-process message '{msg}' (expected '{expected}'")
        elif verbose:
            logging.debug(f"recieved cross-process message '{msg}'")
            
            
class OutputProxy(Plugin):
    def __init__(self, pipe, channel, **kwargs):
        super().__init__(threaded=False)
        self.pipe = pipe
        self.channel = channel
        self.enabled = True
        
    def process(self, input, **kwargs):
        #logging.debug(f"subprocess sending {type(input)} {input} (channel={self.channel})")
        try:
            if self.enabled:
                self.pipe.send_bytes(pickle.dumps((input, self.channel), protocol=-1)) #self.pipe.send((input, self.channel))
        except ValueError as error:
            if 'pickle' in str(error):
                logging.info(f"subprocess output could not be pickled ({type(input)}), disabling channel {self.channel}")
                self.enabled = False
                