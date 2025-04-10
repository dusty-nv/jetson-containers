#!/usr/bin/env python3
import time
import torch
import threading 

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from accelerate import init_empty_weights as init_empty_weights_ctx

from local_llm import LocalLM

class HFModel(LocalLM):
    """
    Huggingface Transformers model
    """
    def __init__(self, model_path, load=True, init_empty_weights=False, **kwargs):
        """
        Initializer
        """
        super(HFModel, self).__init__(**kwargs)
    
        self.model_path = model_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        if not load:
            return
            
        if init_empty_weights:
            with init_empty_weights_ctx():
                self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            if 'gtpq' in self.model_path:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, device=self.device, 
                    torch_dtype=torch.float16, low_cpu_mem_usage=True
                ).eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path,
                    torch_dtype=torch.float16, low_cpu_mem_usage=True
                ).to(self.device).eval()
                
        self.load_config()
        
    def load_config(self):
        """
        @internal get the configuration info from the model
        """
        self.config.type = self.model.config.model_type
        self.config.max_length = self.model.config.max_length
        self.config.vocab_size = self.model.config.vocab_size
    
    def generate(self, inputs, streaming=True, **kwargs):
        """
        Generate output from input tokens or text.
        For kwargs, see https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        """
        if isinstance(inputs, str):
            inputs = self.tokenizer(inputs, return_tensors='pt').input_ids.to(self.device)
 
        self.stats.encode_tokens = len(inputs[0])
        
        skip_special_tokens = kwargs.get('skip_special_tokens', True)
        do_sample = kwargs.get('do_sample', False)
        
        """
        generate_cfg = {
            'inputs': inputs,
            'min_new_tokens': kwargs.get('min_new_tokens'),
            'max_new_tokens': kwargs.get('max_new_tokens'),
        }
        """
        if streaming:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=skip_special_tokens)
            
            def _generate():
                with torch.inference_mode():
                    self.model.generate(inputs=inputs, streamer=streamer, do_sample=do_sample, **kwargs)
            
            thread = threading.Thread(target=_generate)
            thread.start()
            
            return TextIteratorWithStats(self, streamer)
        else:
            time_begin = time.perf_counter()
            outputs = self.model.generate(inputs=inputs, do_sample=do_sample, **kwargs)[0]
            self.stats.decode_time = time.perf_counter()-time_begin
            text = self.tokenizer.decode(outputs, skip_special_tokens=skip_special_tokens)
            self.stats.decode_tokens = len(outputs)
            self.stats.decode_rate = self.stats.decode_tokens / self.stats.decode_time
            return text
            
class TextIteratorWithStats:
    def __init__(self, model, streamer):
        self.model = model
        self.streamer = streamer
        self.model.stats.prefill_latency = 0
        self.model.stats.decode_tokens = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.model.stats.decode_tokens == 0:
            self.time_begin = time.perf_counter()
            
        token = self.streamer.__next__()
        
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
                