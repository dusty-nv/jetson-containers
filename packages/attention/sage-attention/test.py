#!/usr/bin/env python3
print("Testing SageAttention...")
import torch
import torch.nn.functional as F
from sageattention import sageattn
import argparse
from typing import Optional
from diffusers import MochiPipeline
from diffusers.models import MochiTransformer3DModel
from diffusers.utils import export_to_video

# copy the attention processor from the diffusers library
class MochiAttnProcessor2_0:
    """Attention processor used in Mochi."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MochiAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        if image_rotary_emb is not None:

            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x_even = x[..., 0::2].float()
                x_odd = x[..., 1::2].float()

                cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                return torch.stack([cos, sin], dim=-1).flatten(-2)

            query = apply_rotary_emb(query, *image_rotary_emb)
            key = apply_rotary_emb(key, *image_rotary_emb)

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        encoder_query, encoder_key, encoder_value = (
            encoder_query.transpose(1, 2),
            encoder_key.transpose(1, 2),
            encoder_value.transpose(1, 2),
        )

        sequence_length = query.size(2)
        encoder_sequence_length = encoder_query.size(2)
        total_length = sequence_length + encoder_sequence_length

        batch_size, heads, _, dim = query.shape
        attn_outputs = []
        for idx in range(batch_size):
            mask = attention_mask[idx][None, :]
            valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()

            valid_encoder_query = encoder_query[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_key = encoder_key[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_value = encoder_value[idx : idx + 1, :, valid_prompt_token_indices, :]

            valid_query = torch.cat([query[idx : idx + 1], valid_encoder_query], dim=2)
            valid_key = torch.cat([key[idx : idx + 1], valid_encoder_key], dim=2)
            valid_value = torch.cat([value[idx : idx + 1], valid_encoder_value], dim=2)

            ### replace with custom attention ###
            if attn.attention_type == "sage":
                attn_output = sageattn(valid_query, valid_key, valid_value, is_causal=False)
            elif attn.attention_type == "fa3":
                from sageattention.fa3_wrapper import fa3
                attn_output = fa3(valid_query, valid_key, valid_value, is_causal=False)
            elif attn.attention_type == "fa3_fp8":
                from sageattention.fa3_wrapper import fa3_fp8
                attn_output = fa3_fp8(valid_query, valid_key, valid_value, is_causal=False)
            else:
                attn_output = F.scaled_dot_product_attention(
                    valid_query, valid_key, valid_value, dropout_p=0.0, is_causal=False
                )
            ####################################

            valid_sequence_length = attn_output.size(2)
            attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
            attn_outputs.append(attn_output)

        hidden_states = torch.cat(attn_outputs, dim=0)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

def set_attention_mochi(
        model: MochiTransformer3DModel,
        attention_type: str,
):
    # skip the last layer
    for block in model.transformer_blocks[:-1]:
        block.attn1.attention_type = attention_type

        processor = MochiAttnProcessor2_0()
        block.attn1.processor = processor

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="genmo/mochi-1-preview", help='Model path')
parser.add_argument('--compile', action='store_true', help='Compile the model')
parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'sage', 'fa3', 'fa3_fp8'], help='Attention type')
args = parser.parse_args()

pipe = MochiPipeline.from_pretrained(args.model_path, variant="bf16", torch_dtype=torch.bfloat16).to("cuda")

set_attention_mochi(pipe.transformer, args.attention_type)

if args.compile:
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

# Enable memory savings
# pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = "A serene night scene in a forested area. The first frame shows a tranquil lake reflecting the star-filled sky above. The second frame reveals a beautiful sunset, casting a warm glow over the landscape. The third frame showcases the night sky, filled with stars and a vibrant Milky Way galaxy. The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. The style of the video is naturalistic, emphasizing the beauty of the night sky and the peacefulness of the forest."

with torch.no_grad():
    frames = pipe(
        prompt,
        height=480,
        width=848,
        num_frames=1, # can be changed to 84 for shorter video
        guidance_scale=6.0,
        num_inference_steps=64,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

export_to_video(frames, f"mochi_{args.attention_type}.mp4", fps=30)
print('SageAttention OK\n')