#!/usr/bin/env python3

import os
import time
import shutil
import pprint
import argparse
import requests
import numpy as np

from packaging.version import Version

print('testing onnxruntime...')
import onnxruntime as ort
print('onnxruntime version: ' + str(ort.__version__))

ort_version = Version(ort.__version__)

if ort_version > Version('1.10'):
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
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': trt_cache_path
        }
        
        if ort_version >= Version('1.15'):
            provider_options['trt_detailed_build_log'] = True
            provider_options['trt_timing_cache_enable'] = True
        
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
    
    parser.add_argument('--model', type=str, default='/data/models/onnx/cat_dog_epoch_100/resnet18.onnx')
    parser.add_argument('--model-url', type=str, default='https://nvidia.box.com/shared/static/zlvb4y43djygotpjn6azjhwu0r3j0yxc.gz')
    parser.add_argument('--model-tar', type=str, default='cat_dog_epoch_100.tar.gz')
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    print(args)
    
    # download/extract model
    if not os.path.isfile(args.model):
        model_root = os.path.dirname(os.path.dirname(args.model))
        model_tar = os.path.join(model_root, args.model_tar)
        os.makedirs(model_root, exist_ok=True)
        print(f"Downloading {args.model_url} to {model_tar}")
        request = requests.get(args.model_url, allow_redirects=True)
        open(model_tar, 'wb').write(request.content)
        shutil.unpack_archive(model_tar, model_root)
    
    # run model inference tests
    perf = {}
    
    for provider in providers:
        perf[provider] = test_infer(provider, args.model, runs=args.runs, warmup=args.warmup, verbose=args.verbose)
    
    print(f"\nPerformance Summary for {args.model} (over {args.runs} runs)")
    
    for key, value in perf.items():
        print(f"    {key} -- {value:.2f} ms")
        
    print("\nonnxruntime OK\n")
