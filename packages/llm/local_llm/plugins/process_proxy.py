#!/usr/bin/env python3
import os
import time
import pickle
import logging
import traceback
import threading
import multiprocessing as mp

from local_llm import Plugin
from local_llm.utils import LogFormatter


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
        
        self.data_lock = threading.Lock()
        self.control_lock = threading.Lock()
        
        init_msg = self.control_parent.recv()
        
        if init_msg['status'] != 'initialized':
            raise RuntimeError(f"subprocess has an invalid initialization status ({init_msg['status']})")

        super().__init__(output_channels=init_msg['output_channels'], **kwargs)
        logging.info(f"ProcessProxy initialized, output_channels={self.output_channels}")

    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
            
        self.control_lock.acquire()
        self.control_parent.send({'get': name})
        reply = self.control_parent.recv()
        self.control_lock.release()
        
        if name != reply.get('name') or 'value' not in reply:
            raise RuntimeError(f"ProcessProxy failed to get attribute {name} from subprocess (reply was malformed)")
        
        return reply['value']
        
    '''
    def __setattr__(self, name, value):
        if self.is_alive():  # this should only be active after ProcessProxy creates all it's internal members
            self.control_parent.send({'set': {
                'name': name, 
                'value': value
            }})
    '''
    
    def input(self, input=None, **kwargs):
        self.data_lock.acquire()
        input = pickle.dumps((input, kwargs), protocol=-1)
        self.data_parent.send_bytes(input)
        self.data_lock.release()
        
    def interrupt(self, **kwargs):
        self.control_lock.acquire()
        self.control_parent.send({'interrupt': kwargs})
        self.assert_message(self.control_parent.recv(), 'interrupt_ack')
        self.control_lock.release()
        super().interrupt(**kwargs)

    def start(self):
        if not self.is_alive():
            self.control_lock.acquire()
            self.control_parent.send('start')
            self.assert_message(self.control_parent.recv(), 'started')
            self.control_lock.release()
        return super().start()

    def run(self):
        while True:
            output, channel = pickle.loads(self.data_parent.recv_bytes())
            #logging.debug(f"parent process recieved {type(output)} (channel={channel})")
            self.output(output, channel)

    def run_process(self, factory, kwargs):
        log_level = kwargs.get('log_level', 'info')
        
        if kwargs.get('debug') or kwargs.get('verbose'):
            log_level = "debug"
            
        LogFormatter.config(level=log_level)
            
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
                self.plugin = ChatQuery(**kwargs)
            else:
                raise TypeError(f"unsupported proxy class type {factory}")
                
        except Exception as error:
            self.control_child.send({'status': str(type(error))})
            raise error
            
        self.control_child.send({
            'status': 'initialized',
            'output_channels': self.plugin.output_channels,
        })
        
        # forward outputs back to parent process
        for i in range(self.plugin.output_channels):
            self.plugin.add(OutputProxy(self.data_child, i), i)
            
        # start the plugin processing
        self.assert_message(self.control_child.recv(), 'start')

        try:
            self.plugin.start()
        except Exception as error:
            self.control_child.send(str(type(error)))
            raise error

        # start a thread to recieve control messages from the parent
        control_thread = threading.Thread(target=self.run_control)
        control_thread.start()

        self.control_child.send('started')
        
        # recieve inputs from the parent to process
        while True:
            try:
                #time_begin = time.perf_counter()
                input = self.data_child.recv_bytes()
                input, kwargs = pickle.loads(input)
                #logging.debug(f"subprocess recieved {str(type(input))} input  (depickle={(time.perf_counter()-time_begin)*1000} ms)")
                self.plugin(input, **kwargs)
            except Exception as error:
                logging.error(f"Exception occurred in {type(self.plugin)} subprocess data thread\n\n{''.join(traceback.format_exception(error))}")
                
    def run_control(self):
        while True:
            try:
                msg = self.control_child.recv()
                #logging.debug(f"{type(self.plugin)} subprocess recieved control msg from parent process:  {msg}")
                if not isinstance(msg, dict):
                    continue
                if 'get' in msg:
                    self.control_child.send({'name': msg['get'], 'value': getattr(self.plugin, msg['get'])})
                if 'set' in msg:
                    setattr(self.plugin, msg['set']['name'], msg['set']['value'])
                if 'interrupt' in msg:
                    self.plugin.interrupt(**msg['interrupt'])
                    self.control_child.send('interrupt_ack')
            except Exception as error:
                logging.error(f"Exception occurred in {type(self.plugin)} subprocess control thread\n\n{''.join(traceback.format_exception(error))}")
                
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
                