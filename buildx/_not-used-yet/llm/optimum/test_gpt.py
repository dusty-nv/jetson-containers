#!/usr/bin/env python3
# script for benchmarking huggingface CausalLM transformers with Optimum / onnxruntime
import os
import gc
import sys
import time
import psutil
import socket
import datetime
import argparse
import functools

import onnxruntime

from transformers import AutoTokenizer

from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig
from optimum.exporters.tasks import TasksManager


def is_onnx(model):
    """
    Determine if the model has already been exported to ONNX
    """
    if 'onnx' in model or model.startswith('optimum'):
        return True  # check the name for models hosted online
        
    if os.path.isdir(model):  # check file extensions for local models
        for file in os.listdir(model):
            if os.path.splitext(file)[1] == '.onnx':
                return True
                
    return False
    

def benchmark_gpt(model='distilgpt2', provider='TensorrtExecutionProvider',
                  runs=25, warmup=10, do_sample=False, fp16=False, int8=False, 
                  output='', verbose=False, **kwargs):
    """
    Run benchmarking on a text generation language model.
    Models to try:  distilgpt2, optimum/gpt2, MBZUAI/LaMini-GPT-124M
    """
    process = psutil.Process(os.getpid())
    memory_begin = process.memory_info().vms  # https://stackoverflow.com/a/21049737/6037395
    
    print(f"loading {model} with '{provider}'")
    
    provider_options = {}
    
    if provider == 'TensorrtExecutionProvider':
        trt_cache_path = os.path.join(os.path.dirname(__file__), 'data/transformers/trt_cache')
        os.makedirs(trt_cache_path, exist_ok=True)
        
        provider_options = {
            'trt_fp16_enable': fp16,
            'trt_int8_enable': int8,
            'trt_dla_enable': False,
            'trt_detailed_build_log': True,    # not in onnxruntime 1.14
            'trt_timing_cache_enable': True,   # not in onnxruntime 1.14
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': trt_cache_path
        }

    print(f"provider options:  {provider_options}")
    
    # setup session options
    session_options = onnxruntime.SessionOptions()
    
    session_options.log_severity_level = 0 if verbose else 3  # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2
    #session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    # list supported tasks
    #tasks = TasksManager.get_supported_tasks_for_model_type(model, "onnx")
    #print(f"Supported tasks for {model} => {tasks}")

    # load the model
    onnx_model = ORTModelForCausalLM.from_pretrained(model,
        export=not is_onnx(model), 
        use_cache=False,
        use_io_binding=False, # use_cache=False, use_io_binding=True is not supported
        provider=provider,
        provider_options=provider_options,
        session_options=session_options
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model)

    # encode example input
    text = "My name is Arthur and I live in" #"Hello, I’m a language model"  #"what's your name?" "My name is Arthur and I live in"
    tokens = tokenizer(text, return_tensors="pt")
    
    if provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']:
        tokens = tokens.to("cuda")
    
    # benchmark inference
    avg_latency = 0.0
    last_response = ""
    
    for run in range(runs+warmup):
        time_begin = time.perf_counter()
        response = tokenizer.decode(onnx_model.generate(**tokens, do_sample=do_sample)[0])
        time_elapsed = (time.perf_counter() - time_begin) * 1000
    
        print_time = verbose or (run % 10 == 0)
        
        if run >= warmup:
            avg_latency += time_elapsed
        
        if response != last_response:  # when do_sample=True, responses can change
            print(f"\nResponse: {response}\n")
            last_response = response
            print_time = True
            
        if print_time:
            print(f"{provider} {'warmup' if run < warmup else 'run'} {run} -- {time_elapsed:.2f} ms")
            
    avg_latency /= runs
    avg_qps = 1000.0 / avg_latency
    memory_usage = (process.memory_info().vms - memory_begin) / 1024 ** 2
    
    print(f"\nResponse: {response}\n")
    print(f"done running {model} with '{provider}' (latency={avg_latency:.2f} ms, qps={avg_qps:.2f}, memory={memory_usage:.2f} MB, runs={runs}, do_sample={do_sample}, fp16={fp16}, int8={int8})")
    
    # save results to csv
    if output:
        if not os.path.isfile(output):  # csv header
            with open(output, 'w') as file:
                file.write(f"timestamp, hostname, model, provider, do_sample, fp16, int8, latency, qps, memory\n")
        with open(output, 'a') as file:
            file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
            file.write(f"{model}, {provider}, {do_sample}, {fp16}, {int8}, {avg_latency}, {avg_qps}, {memory_usage}\n")

    # reclaim memory now to get a more accurate measurement for the next run
    del onnx_model
    del tokenizer
    
    gc.collect()
    
    return avg_latency, avg_qps, memory_usage
    
    
def quantize_gpt(model='distilgpt2', provider='TensorrtExecutionProvider', output='', **kwargs):
    """
    Apply static int8 quantization to model using the specified dataset for calibration
    """
    if not output:
        output = os.path.join('data/transformers', f'{os.path.basename(model)}-int8')
    
    print(f"loading {model} with '{provider}' for int8 quantization")
    
    onnx_model = ORTModelForCausalLM.from_pretrained(model,
        export=not is_onnx(model), 
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model)
    quantizers = [ORTQuantizer.from_pretrained(onnx_model)]
    #quantizers = [
    #    ORTQuantizer.from_pretrained(onnx_model.model_save_dir, file_name=onnx_file)
    #    for onnx_file in os.listdir(onnx_model.model_save_dir) 
    #    if os.path.splitext(onnx_file)[1] == '.onnx' and onnx_file != 'decoder_with_past_model.onnx'
    #]
    
    if provider == 'TensorrtExecutionProvider':
        qconfig = AutoQuantizationConfig.tensorrt(per_channel=False)
    elif provider == 'CPUExecutionProvider':
        qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
    else:
        raise ValueError(f"unsupported provider '{provider}'")

    def preprocess_fn(ex, tokenizer):
        return tokenizer(ex["sentence"])

    for quantizer in quantizers:
         # Create the calibration dataset
        calibration_dataset = quantizer.get_calibration_dataset(
            "glue",
            dataset_config_name="sst2",
            preprocess_function=functools.partial(preprocess_fn, tokenizer=tokenizer),
            num_samples=50,
            dataset_split="train",
        )

        # Create the calibration configuration containing the parameters related to calibration.
        calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

        # Perform the calibration step: computes the activations quantization ranges
        ranges = quantizer.fit(
            dataset=calibration_dataset,
            calibration_config=calibration_config,
            operators_to_quantize=qconfig.operators_to_quantize,
        )

        # Apply static quantization on the model
        model_quantized_path = quantizer.quantize(
            save_dir=output,
            calibration_tensors_range=ranges,
            quantization_config=qconfig,
        )


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--provider', type=str, default='cuda,tensorrt,cpu')
    parser.add_argument('--model', type=str, default='distilgpt2')
    parser.add_argument('--runs', type=int, default=25)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--output', type=str, default='')
    
    args = parser.parse_args()
    
    # expand provider shorthand
    providers = args.provider.split(',')
    
    for i, provider in enumerate(providers):
        if provider.lower() == 'tensorrt':
            providers[i] = 'TensorrtExecutionProvider'
        elif provider.lower() == 'cuda':
            providers[i] = 'CUDAExecutionProvider'
        elif provider.lower() == 'cpu':
            providers[i] = 'CPUExecutionProvider'

    # quantize for int8
    if args.quantize:
        for provider in providers:
            quantize_gpt(**{**vars(args), **{'provider':provider}})
        sys.exit()
        
    # run model inference tests
    perf = {}
    
    for provider in providers: 
        perf[provider] = benchmark_gpt(**{**vars(args), **{'provider':provider}})
    
    print(f"\nPerformance Summary for {args.model} (runs={args.runs}, do_sample={args.do_sample}, fp16={args.fp16}, int8={args.int8})")
    
    for key, (latency, qps, memory) in perf.items():
        print(f"    {key} -- {latency:.2f} ms, {qps:.2f} qps ({memory:.2f} MB)")
        
    print("\noptimum OK\n")
