#!/usr/bin/env python3

import os
import time
import pprint
import argparse
import numpy as np

print('testing onnxruntime...')
import onnxruntime as ort
print('onnxruntime version: ' + str(ort.__version__))
print(ort.get_build_info())

# verify execution providers
providers = ort.get_available_providers()

print(f'execution providers:  {providers}')

for provider in ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']:
    if provider not in providers:
        raise RuntimeError(f"missing provider '{provider}' from available execution providers {providers}")
 
# test model inference
def test_infer(provider, model='resnet18.onnx', runs=100, warmup=10, verbose=False):

    provider_options = {}
    
    if provider == 'TensorrtExecutionProvider':
        trt_cache_path = os.path.join(os.path.dirname(model), 'trt_cache')
        os.makedirs(trt_cache_path, exist_ok=True)
        
        provider_options = {
            "trt_detailed_build_log": True,    # not in onnxruntime 1.14
            "trt_timing_cache_enable": True,   # not in onnxruntime 1.14
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache_path
        }
        
    # setup session options
    session_options = ort.SessionOptions()
    
    session_options.log_severity_level = 0 if verbose else 3  # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2
    
    # load the model
    print(f"\nloading {model} with '{provider}'")
    session = ort.InferenceSession(model, sess_options=session_options, providers=[provider], provider_options=[provider_options])
    
    # verify the provider
    session_providers = session.get_providers()
    
    print(f"session providers:  {session_providers}")

    if provider not in session_providers:
        raise RuntimeError(f"missing provider '{provider}' from session providers {session_providers}")
        
    pprint.pprint(session.get_provider_options()[provider])
    
    # get inputs/outputs
    inputs = {}
    outputs = []
    
    for i, input in enumerate(session.get_inputs()):
        print(f"input  {i}  name={input.name}  shape={input.shape}  type={input.type}")
        inputs[input.name] = np.random.random_sample(input.shape).astype(np.float32)
        
    for i, output in enumerate(session.get_outputs()):
        print(f"output {i}  name={output.name}  shape={output.shape}  type={output.type}")
        outputs.append(output.name)
        
    # run inference
    time_avg = 0.0
    
    for run in range(runs+warmup):
        time_begin = time.perf_counter()
        output = session.run(outputs, inputs)
        time_elapsed = (time.perf_counter() - time_begin) * 1000
        
        if run >= warmup:
            time_avg += time_elapsed
        
        if verbose:
            print(f"{provider} {'warmup' if run < warmup else 'run'} {run} -- {time_elapsed:.2f} ms")
    
    time_avg /= runs
    print(f"done running {model} with '{provider}'  (avg={time_avg:.2f} ms)")
    return time_avg
    
    
if __name__ == '__main__':  
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='resnet18.onnx')
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # run model inference tests
    perf = {}
    
    for provider in providers:
        perf[provider] = test_infer(provider, **vars(args))
    
    print(f"\nPerformance Summary for {args.model} (over {args.runs} runs)")
    
    for key, value in perf.items():
        print(f"    {key} -- {value:.2f} ms")
        
    print("\nonnxruntime OK\n")
