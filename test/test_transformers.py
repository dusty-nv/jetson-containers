#!/usr/bin/env python3

import os
import time
import argparse
import onnxruntime

from transformers import AutoTokenizer

from optimum.onnxruntime import ORTModelForCausalLM
from optimum.exporters.tasks import TasksManager


# distilgpt2, optimum/gpt2, MBZUAI/LaMini-GPT-124M
def test_gpt(provider, model='optimum/gpt2', runs=100, warmup=10, verbose=False, do_sample=False):

    print(f"loading {model} with '{provider}'")
    
    provider_options = {}
    
    if provider == 'TensorrtExecutionProvider':
        trt_cache_path = os.path.join(os.path.dirname(__file__), 'data/transformers/trt_cache')
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
    text = "Hello, I’m a language model"  #"what's your name?" "My name is Arthur and I live in"
    tokens = tokenizer(text, return_tensors="pt")
    
    if provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']:
        tokens = tokens.to("cuda")
    
    # run inference
    time_avg = 0.0
    last_response = ""
    
    for run in range(runs+warmup):
        time_begin = time.perf_counter()
        response = tokenizer.decode(ort_model.generate(**tokens, do_sample=do_sample)[0])
        time_elapsed = (time.perf_counter() - time_begin) * 1000
    
        print_time = verbose or (run % 10 == 0)
        
        if run >= warmup:
            time_avg += time_elapsed
        
        if response != last_response:
            print(f"\nResponse: {response}\n")
            last_response = response
            print_time = True
            
        if print_time:
            print(f"{provider} {'warmup' if run < warmup else 'run'} {run} -- {time_elapsed:.2f} ms")
            
    time_avg /= runs
    print(f"\nResponse: {response}\n")
    print(f"done running {model} with '{provider}' (avg={time_avg:.2f} ms, runs={runs}, do_sample={do_sample})")
    return time_avg
    

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='MBZUAI/LaMini-GPT-124M')
    parser.add_argument('--runs', type=int, default=25)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # run model inference tests
    perf = {}
    
    for provider in ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']: 
        perf[provider] = test_gpt(provider, **vars(args))
    
    print(f"\nPerformance Summary for {args.model} (runs={args.runs}, do_sample={args.do_sample})")
    
    for key, value in perf.items():
        print(f"    {key} -- {value:.2f} ms")
        
    print("\ntransformers OK\n")
