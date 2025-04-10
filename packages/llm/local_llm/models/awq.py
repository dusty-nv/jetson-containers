#!/usr/bin/env python3
import os

from awq.quantize.quantizer import real_quantize_model_weight
from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp
from accelerate import load_checkpoint_and_dispatch

from .hf import HFModel

class AWQModel(HFModel):
    """
    AWQ model (https://github.com/mit-han-lab/llm-awq)
    """
    def __init__(self, model_path, quant_path, w_bit=4, q_group_size=128, zero_point=True, **kwargs):
        super(AWQModel, self).__init__(model_path, init_empty_weights=True, **kwargs)

        if not quant_path:
            raise ValueError(f"AWQ model needs to have the --quant argument provided, with the path to the quantized model")
            
        if not os.path.isfile(quant_path):
            raise ValueError(f"AWQ quantized model not found: {quant_path}")
        
        self.quant_path = quant_path
        self.config.quant = os.path.basename(quant_path)
        self.config.precision = w_bit
        
        self.q_config = {
            'zero_point': zero_point,
            'q_group_size': q_group_size,
        }

        real_quantize_model_weight(self.model, w_bit=w_bit, q_config=self.q_config, init_only=True)

        self.model = load_checkpoint_and_dispatch(
            self.model, quant_path, device_map='balanced', 
            no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer"]
        )
        
        make_quant_attn(self.model, self.device)
        make_quant_norm(self.model)
        make_fused_mlp(self.model)
        
        self.model.eval()