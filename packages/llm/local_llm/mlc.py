#!/usr/bin/env python3
import os
import time
import json
import queue
import threading

from mlc_chat import ChatModule, ChatConfig, ConvConfig
from mlc_chat.callback import DeltaCallback

from .local_llm import LocalLM


class MLCModel(LocalLM):
    """
    MLC model (https://github.com/mlc-ai/mlc-llm)
    """
    def __init__(self, model_path, **kwargs):
        super(MLCModel, self).__init__(**kwargs)

        self.model = ChatModule(model=model_path)
        self.model_path = model_path
        
        self.config.name = self.model.chat_config.local_id #.model_name
        self.config.type = self.model.chat_config.model_category
        
        with open(self.model.config_file_path) as file:
            json_config = json.load(file)
                
        self.config.max_length = json_config['max_window_size']
        self.config.vocab_size = json_config['vocab_size']
        
        self.is_chat_model = 'chat' in self.config.name
        self.max_new_tokens = self.model.chat_config.max_gen_len
        
        if not self.is_chat_model:
            self.model.chat_config.conv_template = 'LM'

        # ChatConfig(model_lib='Llama-2-7b-chat-hf-q4f16_1', local_id='Llama-2-7b-chat-hf-q4f16_1', conv_template='llama-2', temperature=0.7, repetition_penalty=1.0, top_p=0.95, mean_gen_len=128, max_gen_len=512, shift_fill_factor=0.3, tokenizer_files=['tokenizer.model', 'tokenizer.json'], conv_config=None, model_category='llama', model_name='Llama-2-7b-chat-hf')
        # ConvConfig(name=None, system=None, roles=None, messages=None, offset=None, separator_style=None, seps=None, role_msg_sep=None, role_empty_sep=None, stop_str=None, stop_tokens=None, add_bos=None)
        
    def generate(self, inputs, streaming=True, **kwargs):
        reset_chat = False
    
        if 'max_new_tokens' in kwargs:
            if self.model.chat_config.max_gen_len != kwargs['max_new_tokens']:
                self.model.chat_config.max_gen_len = kwargs['max_new_tokens']
                reset_chat = True

        if reset_chat or not self.is_chat_model:
            self.model.reset_chat(self.model.chat_config)
            
        if streaming:
            callback = StreamIterator(self)
            threading.Thread(target=lambda: self.model.generate(inputs, progress_callback=callback)).start()
            return callback
        else:
            text = self.model.generate(prompt=inputs)
            #print(cm.benchmark_generate(prompt=prompt, generate_length=args.max_new_tokens).strip())
            return text


class StreamIterator(DeltaCallback):
    def __init__(self, model):
        super().__init__()
        
        self.model = model
        self.queue = queue.Queue()
        self.event = threading.Event()
        
        self.stopped = False
        
        self.callback_interval = 1
        self.model.stats.prefill_latency = 0
        self.model.stats.decode_tokens = 0

    def delta_callback(self, delta_message: str):
        self.queue.put(delta_message)
        self.event.set()
        #print(delta_message, end="", flush=True)

    def stopped_callback(self):
        self.stopped = True
        self.event.set()
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.model.stats.decode_tokens == 0:
            self.time_begin = time.perf_counter()
            
        self.event.wait()
        self.event.clear()

        if self.stopped:
            #print(self.model.model.stats())
            raise StopIteration
            
        token = self.queue.get()
        
        time_current = time.perf_counter()
        time_elapsed = time_current - self.time_begin
        
        if self.model.stats.decode_tokens == 0:
            self.model.stats.prefill_latency = time_elapsed
            self.time_begin = time_current
            #self.model.generate_stats.prefill_rate = 
            
        self.model.stats.decode_tokens += 1
        self.model.stats.decode_time = time_elapsed
        
        if self.model.stats.decode_tokens > 1:
            self.model.stats.decode_rate = (self.model.stats.decode_tokens-1) / time_elapsed
            
        return token
        