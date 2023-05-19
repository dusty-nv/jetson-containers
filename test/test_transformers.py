#!/usr/bin/env python3

import os
import time
import argparse
import onnxruntime

from transformers import AutoTokenizer

from optimum.onnxruntime import ORTModelForCausalLM
from optimum.exporters.tasks import TasksManager


# distilgpt2, optimum/gpt2
def test_gpt(provider, model='optimum/gpt2', runs=100, warmup=10, verbose=False):

    print(f"loading {model} with '{provider}'")
    
    provider_options = {}
    
    if provider == 'TensorrtExecutionProvider':
        trt_cache_path = 'test/data/transformers/trt_cache'
        os.makedirs(trt_cache_path, exist_ok=True)
        
        provider_options = {
            #'trt_fp16_enable': True,          # TODO investigate this
            'trt_dla_enable': False,
            'trt_detailed_build_log': True,    # not in onnxruntime 1.14
            'trt_timing_cache_enable': True,   # not in onnxruntime 1.14
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': trt_cache_path
        }

    # setup session options
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 0 if verbose else 3  # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2
    
    # list supported tasks
    #tasks = TasksManager.get_supported_tasks_for_model_type(model, "onnx")
    #print(f"Supported tasks for {model} => {tasks}")

    # load the model
    ort_model = ORTModelForCausalLM.from_pretrained(model,
        export=not model.startswith('optimum'), 
        use_cache=False,
        use_io_binding=False, # use_cache=False, use_io_binding=True is not supported
        provider=provider,
        provider_options=provider_options,
        session_options=session_options
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model)

    # encode example input
    text = "what's your name?"
    tokens = tokenizer(text, return_tensors="pt")
    
    if provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']:
        tokens = tokens.to("cuda")
    
    # run inference
    time_begin = time.perf_counter()
    response = tokenizer.decode(ort_model.generate(**tokens)[0])
    time_elapsed = (time.perf_counter() - time_begin) * 1000
    
    print(response)
    return time_elapsed
    

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='optimum/gpt2')
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # run model inference tests
    perf = {}
    
    for provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider']:
        perf[provider] = test_gpt(provider, **vars(args))
    
    print(f"\nPerformance Summary for {args.model} (over {args.runs} runs)")
    
    for key, value in perf.items():
        print(f"    {key} -- {value:.2f} ms")
        
    print("\ntransformers OK\n")
