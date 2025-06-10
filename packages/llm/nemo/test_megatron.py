#!/usr/bin/env python3
# https://github.com/NVIDIA/NeMo/blob/v1.20.0/examples/nlp/language_modeling/megatron_gpt_eval.py
import time
import pprint
import argparse

from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True, help="path to .nemo file")
parser.add_argument('--config', type=str, default="/opt/nemo/examples/nlp/language_modeling/conf/megatron_gpt_inference.yaml")
parser.add_argument('--prompt', type=str, default='Once upon a time,')

args = parser.parse_args()
print(args)

#print(MegatronGPTModel.list_available_models())

print(f"-- loading config {args.config}")
cfg = OmegaConf.load(args.config)
print(OmegaConf.to_yaml(cfg))

trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=args.model,
            trainer=trainer,
            return_config=True,
        )
OmegaConf.set_struct(pretrained_cfg, True)
with open_dict(pretrained_cfg):
    pretrained_cfg.sequence_parallel = False
    pretrained_cfg.activations_checkpoint_granularity = None
    pretrained_cfg.activations_checkpoint_method = None
    pretrained_cfg.precision = trainer.precision
    if trainer.precision == "16":
        pretrained_cfg.megatron_amp_O2 = False

model = MegatronGPTModel.restore_from(args.model, trainer=trainer, override_config_path=pretrained_cfg)

model.freeze()

# Have to turn off activations_checkpoint_method for inference
try:
    model.model.language_model.encoder.activations_checkpoint_method = None
except AttributeError:
    pass

length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

sampling_params: SamplingParam = {
    "use_greedy": cfg.inference.greedy,
    "temperature": cfg.inference.temperature,
    "top_k": cfg.inference.top_k,
    "top_p": cfg.inference.top_p,
    "repetition_penalty": cfg.inference.repetition_penalty,
    "add_BOS": cfg.inference.add_BOS,
    "all_probs": cfg.inference.all_probs,
    "compute_logprob": cfg.inference.compute_logprob,
    "end_strings": cfg.inference.end_strings,
}
    
fp8_enabled = hasattr(model.cfg, "fp8") and (model.cfg.fp8 == True)
if fp8_enabled:
    print("-- fp8 enabled")
    raise NotImplementedError("fp8 padding not implemented")
    nb_paddings = 0
    while len(cfg.prompts) % 8 != 0:
        cfg.prompts.append("")
        nb_paddings += 1

print(args.prompt)

response = model.generate(
    inputs=[args.prompt], length_params=length_params, sampling_params=sampling_params
)

print(args.response)
