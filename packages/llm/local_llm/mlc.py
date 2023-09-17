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
        
        self.conv_template = self.model.chat_config.conv_template
        self.model.reset_chat(ChatConfig(conv_template='LM', max_gen_len=64))  # disable internal templating
        
        print(self.model.chat_config)
        print(self.model.chat_config.conv_config)
        
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True).start()
        
        self.embedding_cache = {}

        # ChatConfig(model_lib='Llama-2-7b-chat-hf-q4f16_1', local_id='Llama-2-7b-chat-hf-q4f16_1', conv_template='llama-2', temperature=0.7, repetition_penalty=1.0, top_p=0.95, mean_gen_len=128, max_gen_len=512, shift_fill_factor=0.3, tokenizer_files=['tokenizer.model', 'tokenizer.json'], conv_config=None, model_category='llama', model_name='Llama-2-7b-chat-hf')
        # ConvConfig(name=None, system=None, roles=None, messages=None, offset=None, separator_style=None, seps=None, role_msg_sep=None, role_empty_sep=None, stop_str=None, stop_tokens=None, add_bos=None)
        
    def generate(self, inputs, streaming=True, **kwargs):
        stream = StreamIterator(self)
        self.queue.put((inputs, stream, kwargs))
        
        if not streaming:
            text = ''
            for token in stream:
                text += token
            return text
        
        return stream

    def _run(self):
        while True:
            inputs, stream, kwargs = self.queue.get()
            
            if 'max_new_tokens' in kwargs:
                self.model.chat_config.max_gen_len = kwargs['max_new_tokens']
                
            self.model.reset_chat(self.model.chat_config)
            
            # https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_chat/chat_module.py
            time_begin_embed = time.perf_counter()
            
            embedding = self.embedding_cache.get(inputs)
            
            if embedding is None:
                embedding = self.model.embed_text(inputs)
                self.embedding_cache[inputs] = embedding
            else:
                print('CACHE HIT')
                
            self.model.reset_chat(self.model.chat_config)
            
            print('embedding', embedding.shape)

            time_begin_prefill = time.perf_counter()
            self.model._prefill_with_embed(embedding)#, decode_next_token=False)
            time_begin_decode = time.perf_counter()
            self.stats.decode_tokens = 0
            
            last_msg_len = 0
            
            while not self.model._stopped():
                self.model._decode()
                self.stats.decode_tokens += 1
                msg = self.model._get_message()
                msg_len = len(msg)
                msg = msg[last_msg_len:]
                last_msg_len = msg_len
                stream.queue.put(msg)
                stream.event.set()
                
            time_end = time.perf_counter()
            
            stream.stop = True
            stream.event.set()
            
            self.stats.embed_time = time_begin_prefill - time_begin_embed
            self.stats.prefill_time = time_begin_decode - time_begin_prefill
            self.stats.prefill_rate = embedding.shape[1] / self.stats.prefill_time
            self.stats.decode_time = time_end - time_begin_decode
            self.stats.decode_rate = self.stats.decode_tokens / self.stats.decode_time
            
            
class StreamIterator():
    def __init__(self, model):
        super().__init__()
        
        self.model = model
        self.queue = queue.Queue()
        self.event = threading.Event()
        self.stop = False

    def __iter__(self):
        return self

    def __next__(self):
        self.event.wait()
        self.event.clear()

        if self.stop:
            raise StopIteration
            
        return self.queue.get()
        
"""           
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
"""       