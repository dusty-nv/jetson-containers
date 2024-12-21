#!/usr/bin/env python3
print('testing hymba...')

import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from transformers.utils.import_utils import is_torch_fx_available

from torch.utils.checkpoint import checkpoint
from functools import partial

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    else:
        q_embed = None

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids=None):
        if position_ids is None:
            # position_ids: [bsz, seq_len]
            position_ids = torch.arange(x.shape[2], device=x.device, dtype=torch.int64).unsqueeze(0).expand(x.shape[0],
                                                                                                            -1)

        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HymbaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaBranch(nn.Module):
    def __init__(self,
                 intermediate_size,
                 conv_kernel_size,
                 time_step_rank,
                 ssm_state_size,
                 ):
        super().__init__()

        self.intermediate_size = intermediate_size
        self.conv_kernel_size = conv_kernel_size
        self.time_step_rank = time_step_rank
        self.ssm_state_size = ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=True,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1
        )

        num_ssm_param = 1
        self.x_proj = nn.ModuleList(
            [nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False) for _ in
             range(num_ssm_param)])
        self.dt_proj = nn.ModuleList(
            [nn.Linear(self.time_step_rank, self.intermediate_size, bias=True) for _ in range(num_ssm_param)])

        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.ParameterList([nn.Parameter(torch.log(A)) for _ in range(num_ssm_param)])

        self.D = nn.ParameterList([nn.Parameter(torch.ones(self.intermediate_size)) for _ in range(num_ssm_param)])

        self.dt_layernorm = HymbaRMSNorm(self.time_step_rank, eps=1e-06)
        self.B_layernorm = HymbaRMSNorm(self.ssm_state_size, eps=1e-06)
        self.C_layernorm = HymbaRMSNorm(self.ssm_state_size, eps=1e-06)

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, hidden_states, gate):
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        hidden_states = causal_conv1d_fn(
            hidden_states, conv_weights, self.conv1d.bias, activation="silu"
        )
        # we only have a single mamba head at this point
        index = 0
        ssm_parameters = self.x_proj[index](hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        time_step, B, C = self._apply_layernorms(time_step, B, C)
        time_proj_bias = self.dt_proj[index].bias
        self.dt_proj[index].bias = None
        discrete_time_step = self.dt_proj[index](time_step).transpose(1, 2)  # [batch, intermediate_size, seq_len]
        self.dt_proj[index].bias = time_proj_bias

        A = -torch.exp(self.A_log[index].float())

        time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None

        # mammba kernel from mamba_ssm
        outputs = selective_scan_fn(
            hidden_states,
            discrete_time_step,
            A,
            B.transpose(1, 2),
            C.transpose(1, 2),
            self.D[index].float(),
            z=gate,
            delta_bias=time_proj_bias,
            delta_softplus=True,
            return_last_state=True,
        )

        if len(outputs) == 3:
            scan_outputs, ssm_state, _ = outputs
        else:
            scan_outputs, ssm_state = outputs

        scan_outputs = scan_outputs.transpose(1, 2)

        return scan_outputs


class AttentionBranch(nn.Module):

    def __init__(self, num_attention_heads, num_key_value_heads, attention_head_size, attention_window_size=None,
                 modify_attention_mask=False, num_meta_tokens=None, seq_length=None, use_positional_embedding=False,
                 rope_base=None):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_size = attention_head_size
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.attention_window_size = attention_window_size
        self.modify_attention_mask = modify_attention_mask
        self.num_meta_tokens = num_meta_tokens
        self.seq_length = seq_length

        self.use_positional_embedding = use_positional_embedding
        self.rope_base = rope_base

        if self.modify_attention_mask:
            assert num_meta_tokens is not None
            assert self.attention_window_size is not None

            # when using sliding window attention with meta token, we modify the attention mask to
            # for example, when window_size = 3, num_meta_tokens = 2, the attention mask becomes
            # 1
            # 1 1
            # 1 1 1
            # 1 1 1 1
            # 1 1 1 1 1
            # 1 1 0 1 1 1
            # 1 1 0 0 1 1 1
            # 1 1 0 0 0 1 1 1

            # in order to support the modified attention mask, we have to use flexattention (PyTorch>=2.5.0) instead of flash_attention

            try:
                from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks, or_masks

            except ImportError:
                print(
                    "Please install PyTorch>=2.5.0 to use flex_attention if you want to use modify_attention_mask=True")

            # precompile the attention mask for efficiency purposes
            self.create_block_mask = create_block_mask

            def sliding_window(b, h, q_idx, kv_idx):
                return q_idx - kv_idx <= self.attention_window_size

            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

            attn_mask = and_masks(causal_mask, sliding_window)

            def prefix_mask(b, h, q_idx, kv_idx):
                return kv_idx < self.num_meta_tokens

            register_mask = and_masks(causal_mask, prefix_mask)

            self.attn_mask = or_masks(attn_mask, register_mask)  # real mask we use

            qk_length = self.seq_length + self.num_meta_tokens

            self.block_mask = self.create_block_mask(
                self.attn_mask,
                B=None, H=None, Q_LEN=qk_length, KV_LEN=qk_length)

            self.flex_attention = torch.compile(flex_attention)

        if self.use_positional_embedding:
            self.rotary_emb = RotaryEmbedding(
                dim=self.attention_head_size,
                base=self.rope_base)

    def forward(self, query_states, key_states, value_states):
        bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size).transpose(1,
                                                                                                                   2).contiguous()

        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1,
                                                                                                               2).contiguous()

        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1,
                                                                                                                   2).contiguous()

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.use_positional_embedding:
            cos, sin = self.rotary_emb(query_states)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not self.modify_attention_mask:
            # Reashape to the expected shape for Flash Attention
            query_states = query_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
            key_states = key_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
            value_states = value_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)

            if self.attention_window_size is not None:
                attn_outputs = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    causal=True,
                    window_size=(self.attention_window_size, self.attention_window_size),
                )
            else:
                attn_outputs = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    causal=True,
                )

            attn_outputs = attn_outputs.reshape(bsz, q_len,
                                                int(self.num_attention_heads * self.attention_head_size)).contiguous()

        else:
            # print("Using flex_attention")
            if key_states.shape[-2] <= self.block_mask.shape[-2] - 128 or key_states.shape[-2] > self.block_mask.shape[
                -2]:
                # reuse the mask if possible
                # 128 is the minimum block size for flex_attention
                block_mask = self.create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=key_states.shape[-2],
                                                    KV_LEN=key_states.shape[-2])

            else:
                block_mask = self.block_mask

            attn_outputs = self.flex_attention(query_states, key_states, value_states, block_mask=block_mask)

            attn_outputs = attn_outputs.transpose(1, 2).contiguous()
            attn_outputs = attn_outputs.reshape(bsz, q_len,
                                                int(self.num_attention_heads * self.attention_head_size)).contiguous()

        return attn_outputs


class HymbaBlock(nn.Module):
    def __init__(
            self,
            mamba_expand=2,
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=4,
            conv_kernel_size=3,
            time_step_rank=8,
            ssm_state_size=16,
            attention_window_size=None,
            modify_attention_mask=False,
            num_meta_tokens=None,
            seq_length=None,
            use_positional_embedding=False,
            rope_base=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.mamba_expand = mamba_expand  # mamba always expand the hidden size by 2 before operating
        self.conv_kernel_size = conv_kernel_size
        self.time_step_rank = time_step_rank
        self.ssm_state_size = ssm_state_size

        self.intermediate_size = int(self.mamba_expand * self.hidden_size)

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_size = int(self.intermediate_size / self.num_attention_heads)

        self.latent_dim = self.intermediate_size  # will go to mamba branch
        self.latent_dim += self.intermediate_size  # will go to attn branch - query
        self.latent_dim += self.attention_head_size * self.num_key_value_heads * 2  # will go to attn branch - key, value

        # config merging
        self.pre_avg_layernorm1 = HymbaRMSNorm(self.intermediate_size)
        self.pre_avg_layernorm2 = HymbaRMSNorm(self.intermediate_size)

        # in and out proj for (mamba x attn) heads
        self.in_proj = nn.Linear(self.hidden_size, self.latent_dim + self.intermediate_size, bias=True)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

        self.self_attn = AttentionBranch(self.num_attention_heads, self.num_key_value_heads, self.attention_head_size,
                                         attention_window_size, modify_attention_mask, num_meta_tokens, seq_length,
                                         use_positional_embedding, rope_base)

        self.mamba = MambaBranch(self.intermediate_size, self.conv_kernel_size, self.time_step_rank,
                                 self.ssm_state_size)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape

        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        # mamba branch's gate
        hidden_states, gate = projected_states.tensor_split((self.latent_dim,), dim=1)

        # attn branch's query, key, value and mamba branch's x
        query_states, key_states, value_states, hidden_states = hidden_states.tensor_split((self.intermediate_size,
                                                                                            self.intermediate_size + self.attention_head_size * self.num_key_value_heads,
                                                                                            self.intermediate_size + self.attention_head_size * self.num_key_value_heads * 2,),
                                                                                           dim=1)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_outputs = self.self_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states
        )

        mamba_outputs = self.mamba(hidden_states=hidden_states, gate=gate)

        assert attn_outputs.shape == mamba_outputs.shape
        hidden_states = (self.pre_avg_layernorm1(attn_outputs) + self.pre_avg_layernorm2(mamba_outputs)) / 2
        contextualized_states = self.out_proj(hidden_states)

        return contextualized_states

# test the model
layer = HymbaBlock().to(torch.bfloat16).to("cuda")
input = torch.randn(256, 10, 768).to(torch.bfloat16).to("cuda")
output = layer(input)
print(output.shape)


print('hymba OK\n')