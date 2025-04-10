#!/usr/bin/env python3
from .hf import HFModel

from auto_gptq import AutoGPTQForCausalLM


class AutoGPTQModel(HFModel):
    """
    AutoGPTQ (https://github.com/PanQiWei/AutoGPTQ)
    """
    def __init__(self, model_path, **kwargs):
        super(AutoGPTQModel, self).__init__(model_path, load=False, **kwargs)

        self.model = AutoGPTQForCausalLM.from_quantized(
            model_path, 
            device=self.device, 
            use_safetensors=True, 
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        
        self.load_config()
        self.config.precision = self.model.config.quantization_config['bits']