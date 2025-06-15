#!/usr/bin/env python3
print("Testing SpargeAttention...")

import torch
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel
import torch, argparse
from modify_model.modify_flux import set_spas_sage_attn_flux
import os, gc
from spas_sage_attn.autotune import (
    extract_sparse_attention_state_dict,
    load_sparse_attention_state_dict,
)

file_path = "evaluate/datasets/video/prompts.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Flux Evaluation")

    parser.add_argument("--use_spas_sage_attn", action="store_true", help="Use Sage Attention")
    parser.add_argument("--tune", action="store_true", help="tuning hyperpamameters")
    parser.add_argument('--parallel_tune', action='store_true', help='enable prallel tuning')
    parser.add_argument('--l1', type=float, default=0.06, help='l1 bound for qk sparse')
    parser.add_argument('--pv_l1', type=float, default=0.065, help='l1 bound for pv sparse')
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument(
        "--out_path",
        type=str,
        default="evaluate/datasets/image/flux_sparge",
        help="out_path",
    )
    parser.add_argument(
        "--model_out_path",
        type=str,
        default="evaluate/models_dict/flux_saved_state_dict.pt",
        help="model_out_path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(file_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()

    model_id = "black-forest-labs/FLUX.1-dev"
    if args.parallel_tune:
        os.environ['PARALLEL_TUNE'] = '1'
    if args.tune == True:
        os.environ["TUNE_MODE"] = "1"  # enable tune mode

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.float16,
        )
        if args.use_spas_sage_attn:
            set_spas_sage_attn_flux(transformer, verbose=args.verbose, l1=args.l1, pv_l1=args.pv_l1)

        pipe = FluxPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=torch.float16,
        )

        pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()

        for i, prompt in enumerate(prompts[:10]):
            image = pipe(
                prompt.strip(),
                height=1024,  # tune in 512 and infer in 1024 result in a good performance
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator(device="cuda").manual_seed(42)
            ).images[0]

            del image
            gc.collect()
            torch.cuda.empty_cache()

        saved_state_dict = extract_sparse_attention_state_dict(transformer)
        torch.save(saved_state_dict, args.model_out_path)

    else:
        os.environ["TUNE_MODE"] = ""  # disable tune mode

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            local_files_only=True,
            subfolder="transformer",
            torch_dtype=torch.float16,
        )
        if args.use_spas_sage_attn:
            set_spas_sage_attn_flux(transformer, verbose=args.verbose, l1=args.l1, pv_l1=args.pv_l1)
            saved_state_dict = torch.load(args.model_out_path)
            load_sparse_attention_state_dict(transformer, saved_state_dict)

        pipe = FluxPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=torch.float16,
        )

        pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()

        for i, prompt in enumerate(prompts):
            image = pipe(
                prompt.strip(),
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator(device="cuda").manual_seed(42)
            ).images[0]

            image.save(f"{args.out_path}/{i}.jpg")
            del image
            gc.collect()
            torch.cuda.empty_cache()
print('SpargeAttention OK\n')