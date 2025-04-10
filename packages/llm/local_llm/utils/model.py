#!/usr/bin/env python3
import os
import logging

import numpy as np
import onnxruntime as ort

from glob import glob
from packaging.version import Version

from huggingface_hub import snapshot_download, hf_hub_download, login


def download_model(model, type='model', cache_dir='$TRANSFORMERS_CACHE', use_safetensors=False, **kwargs):
    """
    Get the local path to a cached model or file in the cache_dir, or download it from HuggingFace Hub if needed.
    If the asset is private and authentication is required, set the HUGGINGFACE_TOKEN environment variable.
    cache_dir is where the model gets downloaded to - by default, set to $TRANSFORMERS_CACHE (/data/models/huggingface)
    By default, the PyTorch .bin weights will be downloaded instead of the .safetensors (use_safetensors=False)
    """
    token = os.environ.get('HUGGINGFACE_TOKEN', os.environ.get('HUGGING_FACE_HUB_TOKEN'))
    
    if token:
        login(token=token)
       
    if not cache_dir or cache_dir == '$TRANSFORMERS_CACHE':
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/root/.cache/huggingface')
        
    # handle either "org/repo" or individual "org/repo/file"
    # the former has 0-1 slashes, while the later has 2.
    num_slashes = 0
    
    for c in model:
        if c == '/':
            num_slashes += 1
            
    if num_slashes >= 2:  
        slash_count = 0
        
        for idx, i in enumerate(model):
            if i == '/':
                slash_count += 1
                if slash_count == 2:
                    break
                    
        repo_id = model[:idx]
        filename = model[idx+1:]
        
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=type, cache_dir=cache_dir, resume_download=True)
    else:
        repo_path = snapshot_download(repo_id=model, repo_type=type, cache_dir=cache_dir, resume_download=True, ignore_patterns=['*.safetensors', '*.gguf'])
                                      
        if glob(os.path.join(repo_path, '*model*.pt')) or glob(os.path.join(repo_path, '*model*.bin')):
            return repo_path
            
        return snapshot_download(repo_id=model, repo_type=type, cache_dir=cache_dir, resume_download=True, ignore_patterns=['*.gguf'])
    
    
def default_model_api(model_path, quant_path=None):
    """
    Given the local path to a model, determine the type of API to use to load it.
    TODO check the actual model files / configs instead of just parsing the paths
    """
    if quant_path:
        quant_api = default_model_api(quant_path)
        
        if quant_api != 'hf':
            return quant_api

    model_path = model_path.lower()

    if 'ggml' in model_path or 'ggml' in model_path:
        return 'llama.cpp'
    elif 'gptq' in model_path:
        return 'auto_gptq'  # 'exllama'
    elif 'awq' in model_path:
        return 'awq'
    elif 'mlc' in model_path:
        return 'mlc'
    else:
        return 'hf'
        
        
class ONNXRuntimeModel:
    """
    Base class for OnnxRuntime models.
    """
    def __init__(self, model, providers='CUDAExecutionProvider', debug=False, **kwargs):
        """
        Load an ONNX Runtime model.
        """
        self.model_path = model
        
        if isinstance(providers, str):
            providers = [providers]
        
        provider_options = []
        
        for provider in providers:
            if provider == 'TensorrtExecutionProvider':
                trt_cache_path = os.path.join(os.path.dirname(self.model_path), 'trt_cache')
                os.makedirs(trt_cache_path, exist_ok=True)
                
                options = {
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': trt_cache_path
                }
                
                if ort_version >= Version('1.15'):
                    options['trt_detailed_build_log'] = True
                    options['trt_timing_cache_enable'] = True
        
                provider_options.append(options)
            else:
                provider_options.append({})
                
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 0 if debug else 3  # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2
    
        logging.info(f"loading ONNX model '{self.model_path}' with onnxruntime ({provider})")
        self.model = ort.InferenceSession(model, sess_options=session_options, providers=providers, provider_options=provider_options)
        logging.info(f"loaded ONNX model '{self.model_path}' with onnxruntime ({provider})")
        
        self.inputs = self.model.get_inputs()
        self.outputs = self.model.get_outputs()
        
        for idx, binding in enumerate(self.inputs):
            print('')
            print(f"input {idx} - {binding.name}")
            print(f"   shape: {binding.shape}")
            print(f"   type:  {binding.type}")
            print('')
 
    def execute(self, inputs, return_dict=False, **kwargs):
        """
        Run the DNN model in ONNXRuntime.  The inputs are provided as numpy arrays in a list/tuple/dict.
        Note that run() doesn't perform any pre/post-processing - this is typically done in subclasses.
        
        Parameters:
          inputs (array, list[array], dict[array]) -- the network inputs as numpy array(s).
                         If there is only one input, it can be provided as a single numpy array.
                         If there are multiple inputs, they can be provided as numpy arrays in a
                         list, tuple, or dict.  Inputs in lists and tuples are assumed to be in the
                         same order as the input bindings.  Inputs in dicts should have keys with the
                         same names as the input bindings.
          return_dict (bool) -- If True, the results will be returned in a dict of numpy arrays, where the
                                keys are the names of the output binding names. By default, the results will 
                                be returned in a list of numpy arrays, in the same order as the output bindings.
          
        Returns the model output as a numpy array (if only one output), list[ndarray], or dict[ndarray].
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        
        assert len(inputs) == len(self.inputs)
        
        if isinstance(inputs, (list,tuple)):
            inputs = {self.inputs[i].name : input for i, input in enumerate(inputs)}
        elif not isinstance(inputs, dict):        
            raise ValueError(f"inputs must be a list, tuple, or dict (instead got type '{type(inputs).__name__}')")
            
        outputs = self.model.run(None, inputs)
        
        if return_dict:
            return {self.outputs[i].name : output for i, output in enumerate(outputs)}
            
        if len(outputs) == 1:
            return outputs[0]
        
        return outputs
        