#!/usr/bin/env python3
import os
import tvm
import time
import math
import json
import queue
import threading
import subprocess
import random
import logging

import torch
import numpy as np

from tvm.runtime.relax_vm import VirtualMachine
from transformers import AutoTokenizer, AutoConfig

from local_llm import LocalLM, StreamingResponse


#
# TODO does not respect do_sample=False
#
class MLCModel(LocalLM):
    """
    MLC model (https://github.com/mlc-ai/mlc-llm)
    """
    def __init__(self, model_path, quant='q4f16_ft', **kwargs):
        """
        Parameters:
        
          model_path (str) -- the original model on HuggingFace - used for getting
                              the tokenizer and original model config.  If this is a 
                              Llama2 model, it will be automatically set if Llama/ect
                              is in the path. Otherwise, this should be set to the name.
                              
          quant (str) -- either a directory path containing the mlc_llm quantized model,
                         or the quantization method to use (q4f16_1, q4f16_ft, ect)          
                         If a path, there should be a .so in this dir, with params/ 
                         directory under it containing the weights and MLC config.
        """
        super(MLCModel, self).__init__(model_path, **kwargs)

        # perform quantization if needed
        if not quant:
            quant = 'q4f16_ft'
            
        if not os.path.isdir(quant):
            quant = MLCModel.quantize(model_path, quant, **kwargs)
            
        self.config.quant = quant.split('-')[-1]  # recover the quant method        
        self.quant_path = quant
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        
        # initialize tvm device
        self.device = tvm.runtime.cuda(0)  # tvm.runtime.Device(tvm.runtime.Device.kDLCUDAManaged, 0)
        assert(self.device.exist) # this is needed to initialize CUDA?
        logging.info(f"device={self.device}, name={self.device.device_name}, compute={self.device.compute_version}, max_clocks={self.device.max_clock_rate}, multiprocessors={self.device.multi_processor_count}, max_thread_dims={self.device.max_thread_dimensions}, api_version={self.device.api_version}, driver_version={self.device.driver_version}")

        # load model config
        with open(os.path.join(quant, 'params/mlc-chat-config.json'), 'r') as file:
            config = json.load(file)
        
        #self.config.name = config['local_id']  # model_name
        self.config.type = config['model_category']  # 'conv_template'
        self.config.max_length = config['max_window_size']
        self.config.vocab_size = config['vocab_size']
        
        # load model's dynamic library
        self.module_path = os.path.join(quant, os.path.basename(quant) + '-cuda.so')
        
        if not os.path.isfile(self.module_path):
            raise IOError(f"MLC couldn't find {self.module_path}")
            
        logging.info(f"loading {self.config.name} from {self.module_path}")
        load_time_begin = time.perf_counter()
        self.module = tvm.runtime.load_module(self.module_path)
        
        self.vm = self.module['vm_load_executable']()
        self.vm['vm_initialization'](
            self.device.device_type, self.device.device_id,
            VirtualMachine.POOLED_ALLOCATOR,
            tvm.runtime.Device.kDLCPU, 0,  # kDLCUDAManaged kDLCUDAHost  kDLCUDA
            VirtualMachine.POOLED_ALLOCATOR
        )    
        
        # embedding/generation functions
        self._embed = self.vm['embed']
        self._decode = self.vm['decode']
        self._prefill_with_embed = self.vm['prefill_with_embed']
        
        # KV cache manipulation functions
        self._create_kv_cache = self.vm['create_kv_cache']

        if self.vm.implements_function('reset_kv_cache'):
            self._clear_kv_cache = self.vm['reset_kv_cache']
            self.backtracking_kv = False
        else:
            self._clear_kv_cache = tvm.get_global_func('vm.builtin.attention_kv_cache_array_clear')
            self.backtracking_kv = True
            
        self._pop_kv_cache = tvm.get_global_func('vm.builtin.attention_kv_cache_array_popn')
        self._append_kv_cache = tvm.get_global_func('vm.builtin.attention_kv_cache_append')
        self._update_kv_cache = tvm.get_global_func('vm.builtin.attention_kv_cache_update')
        self._view_kv_cache = tvm.get_global_func('vm.builtin.attention_kv_cache_view')
        
        self._sample_top_p_from_prob = tvm.get_global_func('vm.builtin.sample_top_p_from_prob')
        self._sample_top_p_from_logits = tvm.get_global_func('vm.builtin.sample_top_p_from_logits')
        
        self._apply_repetition_penalty = tvm.get_global_func('vm.builtin.apply_repetition_penalty')
        self._apply_softmax_with_temperature = tvm.get_global_func('vm.builtin.apply_softmax_with_temperature')

        #self.kv_cache = None #create_kv_cache()
        
        # param loading functions
        self._load_cache = tvm.get_global_func('vm.builtin.ndarray_cache.load')
        self._load_params = tvm.get_global_func('vm.builtin.param_array_from_cache')
        self._clear_cache = tvm.get_global_func('vm.builtin.ndarray_cache.clear')

        # load model weights
        self._load_cache(os.path.join(self.quant_path, 'params'), self.device.device_type, self.device.device_id)
        self.params = self._load_params('param', -1)
        self._clear_cache() # after we get params, it is safe to simply clear the cached version
        self.config.load_time = time.perf_counter() - load_time_begin
        
        # add up the total weight size
        self.params_size = 0

        for param in self.params:
            self.params_size += math.prod(param.shape) * np.dtype(param.dtype).itemsize

        self.config.params_size = self.params_size / (1024 * 1024)
        
        # create multithreading
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True).start()        
        self.embedding_cache = {}

    
    @staticmethod
    def quantize(model, method='q4f16_ft', output='/data/models/mlc/dist', **kwargs):
        """
        Quantize a model with the given method.  It will be saved under the output directory,
        in a subdirectory based on the model name and quant method (Llama-2-7b-chat-hf-q4f16_ft)
        """
        model_name = kwargs.get('name', os.path.basename(model))
        model_path = os.path.join(output, 'models', model_name)
        quant_path = os.path.join(output, model_name + '-' + method)
        
        if os.path.isdir(quant_path):
            return quant_path
            
        if not os.path.isdir(model_path):
            os.symlink(model, model_path, target_is_directory=True)
            
        cmd = f"python3 -m mlc_llm.build --model {model_path} --quantization {method} "
        cmd += f"--target cuda --use-cuda-graph --use-flash-attn-mqa --sep-embed "
        cmd += f"--max-seq-len {AutoConfig.from_pretrained(model).max_position_embeddings} "
        cmd += f"--artifact-path {output}"
        
        logging.info(f"running MLC quantization:\n\n{cmd}\n\n")
        subprocess.run(cmd, executable='/bin/bash', shell=True, check=True)  
        
        return quant_path
        
    @staticmethod
    def get_kv_cache_size(kv_cache, length=None):
        size = 0
        for n in range(len(kv_cache)):
            view = view_kv_cache(kv_cache[n])
            #print(f"kv_cache {n}  {view.shape} {view.dtype}")
            if length is None:
                length = view.shape[0]
            size += length * view.shape[1] * view.shape[2] * np.dtype(view.dtype).itemsize
        return size
    
    def embed_text(self, text, return_tensors='np', add_special_tokens=False, use_cache=False):  # pt, np, tvm
        if use_cache:
            embedding = self.embedding_cache.get(text)
        else:
            embedding = None
            
        if embedding is None:
            tokens = self.tokenizer(text, 
                add_special_tokens=add_special_tokens, 
                return_tensors='np').input_ids
            embedding = self.embed_tokens(tokens)
            self.embedding_cache[text] = embedding
            self.device = embedding.device
        else:
            logging.debug(f'text embedding cache hit ({text})')
         
        if return_tensors == 'np':
            embedding = embedding.numpy()
            
        return embedding
    
    def embed_tokens(self, tokens, return_tensors='np'):  # pt, np, tvm
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
            
        if isinstance(tokens, np.ndarray):
            tokens = tvm.nd.array(tokens.astype(np.int32), self.device)

        time_begin = time.perf_counter()
        embedding = self._embed(tokens, self.params)
        self.stats.embed_time = time.perf_counter() - time_begin
        return embedding
        
    def generate(self, inputs, streaming=True, **kwargs):
        """
        Parameters:
        
          inputs (str|ndarray) -- text or embedding inputs to the model
          streaming (bool) -- if True, an iterator will be returned that returns text chunks.
                              Otherwise, this function will block and return the generated text.
        
        kwargs:

          max_new_tokens (int) -- the number of tokens to output in addition to the prompt (default: 128)
          min_new_tokens (int) -- force the model to generate a set number of output tokens (default: -1)
          do_sample (bool) -- if True, temperature/top_p will be used.  Otherwise, greedy search (default: False)
          repetition_penalty -- the parameter for repetition penalty. 1.0 means no penalty (default: 1.0)  
          temperature (float) -- randomness token sampling parameter (default=0.7, only used if do_sample=True)
          top_p (float) -- if set to float < 1 and do_sample=True, only the smallest set of most probable tokens
                           with probabilities that add up to top_p or higher are kept for generation (default 0.95)
          stop_tokens (list[int]) -- defaults to EOS token ID
          kv_cache (ndarray) -- previous kv_cache that the inputs will be appended to.  By default, a blank kv_cache 
                                will be created for each generation (i.e. a new chat).  This generation's kv_cache
                                will be set in the returned StreamingResponse after the request is complete.
                                
          TODO start_tokens
        """
        stream = StreamingResponse(self, inputs, **kwargs)
        self.queue.put(stream)
        
        if not streaming:
            text = ''
            for token in stream:
                text += token
            return text
        
        return stream

    def _generate(self, stream):
        max_new_tokens = stream.kwargs.get('max_new_tokens', 128)
        min_new_tokens = stream.kwargs.get('min_new_tokens', -1)
        
        do_sample = stream.kwargs.get('do_sample', False)
        temperature = stream.kwargs.get('temperature', 0.7)
        top_p = stream.kwargs.get('top_p', 0.95)
        repetition_penalty = stream.kwargs.get('repetition_penalty', 1.0)
        
        stop_tokens = stream.kwargs.get('stop_tokens', [self.tokenizer.eos_token_id])
        
        if isinstance(stop_tokens, int):
            stop_tokens = [stop_tokens]

        if isinstance(stream.input, str):
            embedding = self.embed_text(stream.input, return_tensors='tvm')
        else:
            embedding = stream.input
            
        if not isinstance(embedding, tvm.runtime.ndarray.NDArray):
            embedding = tvm.nd.array(embedding, device=self.device)
       
        self.stats.input_tokens = embedding.shape[1]
        self.stats.output_tokens = 0
        
        # create a kv_cache if needed
        if stream.kv_cache is None:
            stream.kv_cache = self._create_kv_cache()
            stream.kv_cache.num_tokens = 0

        stream.kv_cache.num_tokens += embedding.shape[1]

        # returns a list[logits, [kv_cache]] - logits are [1,1,32000] float32 (on the GPU)
        time_begin_prefill = time.perf_counter()
        
        output = self._prefill_with_embed(embedding, 
            tvm.runtime.ShapeTuple([stream.kv_cache.num_tokens]), 
            stream.kv_cache, self.params
        )

        # decode until EOS or max_new_tokens
        time_begin_decode = time.perf_counter()
        
        while True:
            #time_begin_sample = time.perf_counter()
            token = self._sample(output[0], do_sample, temperature, top_p, repetition_penalty)
            #sample_time = (time.perf_counter() - time_begin_sample) * 1000
            #print(f"SAMPLE_TIME:  {sample_time:.2f}  SAMPLE_RATE:  {1000/sample_time:.2f}")
            
            stream.output_tokens.append(token)
            stream.event.set()
            
            if token in stop_tokens and len(stream.output_tokens) > min_new_tokens:
                break

            if len(stream.output_tokens) >= max_new_tokens:
                break
                
            if stream.stopping or stream.stopped:
                break

            stream.kv_cache.num_tokens += 1
            self.stats.output_tokens += 1
            
            #time_begin_decode2 = time.perf_counter()
            output = self._decode(
                tvm.nd.array(np.array([[stream.output_tokens[-1]]], dtype=np.int32), self.device),
                tvm.runtime.ShapeTuple([stream.kv_cache.num_tokens]), stream.kv_cache, self.params
            )
            #decode_time = (time.perf_counter() - time_begin_decode2) * 1000
            #print(f"DECODE_TIME:  {decode_time:.2f}  DECODE_RATE:  {1000/decode_time:.2f}")
            #time.sleep(0.225)

        time_end_decode = time.perf_counter()
        
        stream.stopped = True
        stream.event.set()
        
        self.stats.prefill_time = time_begin_decode - time_begin_prefill
        self.stats.prefill_rate = self.stats.input_tokens / self.stats.prefill_time
        self.stats.decode_time = time_end_decode - time_begin_decode
        self.stats.decode_rate = self.stats.output_tokens / self.stats.decode_time
       
    def _sample(self, logits, do_sample, temperature, top_p, repetition_penalty):
        # TODO implement repetition penalty
        # https://github.com/mlc-ai/mlc-llm/blob/6e40c21fb6433aeffe50ee321f1b589ef846b6fb/cpp/llm_chat.cc#L1044
        if do_sample:
            return self._sample_top_p_from_logits(logits, top_p, temperature, random.random())
        else:
            return np.argmax(logits.numpy()) #, axis=-1)

    def _run(self):
        while True:
            stream = self.queue.get()
            self._generate(stream)
            