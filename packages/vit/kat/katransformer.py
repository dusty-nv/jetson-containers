""" Kolmogorov-Arnold Transformer (KAT) model
author = "Xingyi Yang"
"""
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations
from timm.models.layers import to_2tuple

import numpy as np


__all__ = ['KAT']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)


from kat_rational import KAT_Group


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalisation
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'gelu':
        return math.sqrt(2.3567850379928976)
    elif nonlinearity == 'silu' or nonlinearity == 'swish':
        return math.sqrt(2.8178205270359653)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    
torch.nn.init.calculate_gain = calculate_gain


class KAN(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=KAT_Group,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            act_init="gelu",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act1 = KAT_Group(mode="identity")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.act2 = KAT_Group(mode=act_init)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x

def get_ortho_like(dim, alpha, beta, sign=1, dist='uniform'):
    """
    Generate an orthogonal-like matrix with specified dimensions and properties.

    Args:
        dim (int): The dimension of the matrix.
        alpha (float): The scaling factor for the random matrix.
        beta (float): The scaling factor for the identity matrix.
        sign (int, optional): The sign of the identity matrix. Defaults to 1.
        dist (str, optional): The distribution of the random matrix. 
            Can be either 'normal' or 'uniform'. Defaults to 'uniform'.

    Returns:
        tuple: A tuple containing the left and right orthogonal-like matrices.
    """
    if dist == 'normal':
        A = alpha * np.random.normal(size=(dim,dim)) / (dim**0.5) + sign * beta * np.eye(dim)
    if dist == 'uniform':
        A = alpha * np.random.uniform(size=(dim,dim), low=-3**0.5 / (dim**0.5), high = 3**0.5 / (dim**0.5)) + sign * beta * np.eye(dim)

    U, S, V = np.linalg.svd(A)
    L = U @ np.diag(np.sqrt(S))
    R = np.diag(np.sqrt(S)) @ V
    return L, R


def get_ortho_like_gaussian2d(dim, alpha, beta, sigma=1.0, sign=1, dist='uniform'):
    if dist == 'normal':
        A = alpha * np.random.normal(size=(dim, dim)) / (dim**0.5)
    elif dist == 'uniform':
        A = alpha * np.random.uniform(low=-3**0.5 / (dim**0.5), high=3**0.5 / (dim**0.5), size=(dim, dim))
    
    size = int(np.sqrt(dim))
    gauss_diag = beta * gaussian_correlation_matrix((size, size), sigma=sigma)
    gauss_diag = sign * gauss_diag

    A += gauss_diag
    
    # Compute the SVD of the updated matrix A
    U, S, V = np.linalg.svd(A)

    # Create the left and right matrices
    L = U @ np.diag(np.sqrt(S))
    R = np.diag(np.sqrt(S)) @ V
    
    return L, R
    

def gaussian_correlation_matrix(image_shape, sigma=1.0):
    """
    Generate a Gaussian correlation matrix for a flattened 2D image.

    Args:
        image_shape (tuple): The shape of the image (height, width).
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        np.array: A 2D array where each entry represents the correlation
                  between two flattened image pixels.
    """
    # Create an array of all pixel coordinates
    y, x = np.indices(image_shape)
    
    # Flatten the coordinate arrays
    y = y.ravel()
    x = x.ravel()
    
    # Compute distances between every pair of pixels
    distance_matrix = np.sqrt((y[:, np.newaxis] - y[np.newaxis, :])**2 + 
                              (x[:, np.newaxis] - x[np.newaxis, :])**2)
    
    # Compute the Gaussian correlation matrix using the distance matrix
    correlation_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))

    return correlation_matrix

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = KAN,
            act_init: str = 'gelu',
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            act_init=act_init,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x



class KATVisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', '', 'kan', 'kan_mimetic', 'mimetic', 'kan_mimetic2d'] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None, # None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = KAN,
            act_init: str = 'gelu',
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.act_init = act_init

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                act_init=act_init,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()
            
    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '', 'kan', 'kan_mimetic', 'mimetic')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        
        mode = mode + f'_{self.act_init}' if 'kan' in mode else mode
        named_apply(get_init_weights_vit(mode, head_bias), self)
        
        if 'mimetic' in mode:
            named_apply(init_weights_attn_mimetic, self)
        # named_apply(init_weights_attn_mimetic, self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        if checkpoint_path.endswith('.npz'):
            # load from .npz checkpoint
            _load_weights(self, checkpoint_path, prefix)
        else:
            print(f'Loading model weights from: {checkpoint_path}')
            # load from .pth checkpoint
            state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:]
            msg = self.load_state_dict(state_dict, strict=False)
            print(msg)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        if hasattr(self.patch_embed, 'set_grad_checkpointing'):
            self.patch_embed.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int], Tuple[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int], Tuple[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, List[int], Tuple[int]] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> List[torch.Tensor]:
        """ Intermediate layer accessor inspired by DINO / DINOv2 interface.
        NOTE: This API is for backwards compat, favour using forward_intermediates() directly.
        """
        return self.forward_intermediates(
            x, n,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            output_fmt='NCHW' if reshape else 'NLC',
            intermediates_only=True,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
        
        
def init_weights_vit_kan_gelu(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        if 'mlp' in name:
            if 'fc1' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            elif 'fc2' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='gelu')
                 
            # nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
           
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
        
def init_weights_vit_kan_swish(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        if 'mlp' in name:
            if 'fc1' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            elif 'fc2' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='swish')
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
        
def init_weights_attn_mimetic(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, Attention):
        alpha1 = 0.7
        beta1 = 0.7
        alpha2 = 0.4
        beta2 = 0.4
        head_dim = module.head_dim
        embed_dim = module.head_dim * module.num_heads

        for h in range(module.num_heads):
            Q, K = get_ortho_like(embed_dim, alpha1, beta1, 1)
            Q = Q[:,:head_dim]
            K = K.T[:,:head_dim]

            module.qkv.weight.data[(h*head_dim):((h+1)*head_dim)] = torch.tensor(Q.T).float()
            module.qkv.weight.data[embed_dim+(h*head_dim):embed_dim+((h+1)*head_dim)] = torch.tensor(K.T).float()

        V, Proj = get_ortho_like(embed_dim, alpha2, beta2, -1)
        module.qkv.weight.data[2*embed_dim:] = torch.tensor(V).float()
        module.proj.weight.data = torch.tensor(Proj).float()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.0) -> None:
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0) -> Callable:
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    elif 'kan' in mode:
        if 'gelu' in mode:
            return init_weights_vit_kan_gelu
        elif 'swish' in mode:
            return init_weights_vit_kan_swish
        else:
            AssertionError(f'Unknown mode {mode}')
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
) -> torch.Tensor:
    """ Rescale the grid of position embeddings when loading from state_dict.
    *DEPRECATED* This function is being deprecated in favour of using resample_abs_pos_embed
    """
    ntok_new = posemb_new.shape[1] - num_prefix_tokens
    ntok_old = posemb.shape[1] - num_prefix_tokens
    gs_old = [int(math.sqrt(ntok_old))] * 2
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    return resample_abs_pos_embed(
        posemb, gs_new, gs_old,
        num_prefix_tokens=num_prefix_tokens,
        interpolation=interpolation,
        antialias=antialias,
        verbose=True,
    )


@torch.no_grad()
def _load_weights(model: KATVisionTransformer, checkpoint_path: str, prefix: str = '') -> None:
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True, idx=None):
        if idx is not None:
            w = w[idx]
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
        elif 'params/img/embedding/kernel' in w:
            prefix = 'params/img/'
            big_vision = True

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if (isinstance(model.head, nn.Linear) and
            f'{prefix}head/bias' in w and
            model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        model.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        model.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        if f'{prefix}Transformer/encoderblock/LayerNorm_0/scale' in w:
            block_prefix = f'{prefix}Transformer/encoderblock/'
            idx = i
        else:
            block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
            idx = None
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale'], idx=idx))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias'], idx=idx))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False, idx=idx).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False, idx=idx).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel'], idx=idx).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias'], idx=idx))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale'], idx=idx))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias'], idx=idx))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel'], idx=idx))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias'], idx=idx))


def _convert_openai_clip(
        state_dict: Dict[str, torch.Tensor],
        model: KATVisionTransformer,
        prefix: str = 'visual.',
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
        out_dict[k] = v
    return out_dict


def _convert_dinov2(
        state_dict: Dict[str, torch.Tensor],
        model: KATVisionTransformer,
) -> Dict[str, torch.Tensor]:
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: KATVisionTransformer,
        adapt_layer_scale: bool = False,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if 'visual.class_embedding' in state_dict:
        state_dict = _convert_openai_clip(state_dict, model)
    elif 'module.visual.class_embedding' in state_dict:
        state_dict = _convert_openai_clip(state_dict, model, prefix='module.visual.')
    elif "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)
    elif "encoder" in state_dict:
        # IJEPA, vit in an 'encoder' submodule
        state_dict = state_dict['encoder']
        prefix = 'module.'
    elif 'visual.trunk.pos_embed' in state_dict or 'visual.trunk.blocks.0.norm1.weight' in state_dict:
        # OpenCLIP model with timm vision encoder
        prefix = 'visual.trunk.'
        if 'visual.head.proj.weight' in state_dict and isinstance(model.head, nn.Linear):
            # remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
            out_dict['head.weight'] = state_dict['visual.head.proj.weight']
            out_dict['head.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.875,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }

default_cfgs = {
    'kat_tiny_patch16_224': _cfg(
        url='https://huggingface.co/adamdad/kat_pretained/resolve/main/kat_tiny_patch16_224_1f3ad3b2e69821f3d412f2924cf159a0e266f142d739cb68f68f796f5a0fe289.pth'),
    'kat_tiny_patch16_224.vitft': _cfg(
        url='https://huggingface.co/adamdad/kat_pretained/resolve/main/kat_tiny_patch16_224-finetune_64f124d003803e4a7e1aba1ba23500ace359b544e8a5f0110993f25052e402fb.pth'),
    'kat_small_patch16_224': _cfg(
        url='https://huggingface.co/adamdad/kat_pretained/resolve/main/kat_small_patch16_224_32487885cf13d2c14e461c9016fac8ad43f7c769171f132530941e930aeb5fe2.pth'),
    'kat_small_patch16_224.vitft': _cfg(
        url='https://huggingface.co/adamdad/kat_pretained/resolve/main/kat_small_patch_224-finetune_3ae087a4c28e2993468eb377d5151350c52c80b2a70cc48ceec63d1328ba58e0.pth'),
    'kat_base_patch16_224': _cfg(
        url='https://huggingface.co/adamdad/kat_pretained/resolve/main/kat_base_patch16_224_abff874d925d756d15cde97303f772a3460ddbd44b9c53fb9ce5cf15be230fb6.pth'),
    'kat_base_patch16_224.vitft': _cfg(
        url='https://huggingface.co/adamdad/kat_pretained/resolve/main/kat_base_patch16_224-finetune_440bf1ead9dd8ecab642078cfb60ae542f1fa33ca65517260501e02c011e38f2.pth')
}


def _create_kat_transformer(variant: str, pretrained: bool = False, **kwargs) -> KATVisionTransformer:
    out_indices = kwargs.pop('out_indices', 3)
    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        KATVisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )

@register_model
def kat_tiny_patch16_224(pretrained: bool = False, **kwargs) -> KATVisionTransformer:
    """ KAT-Tiny with rational activations (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, 
                      act_layer=KAT_Group, 
                      act_init='swish',
                      mlp_layer=KAN, 
                      weight_init="kan_mimetic")
    model = _create_kat_transformer('kat_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def kat_tiny_gelu_patch16_224(pretrained: bool = False, **kwargs) -> KATVisionTransformer:
    """ KAT-Tiny with rational activations (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, 
                      act_layer=KAT_Group, 
                      act_init='gelu',
                      mlp_layer=KAN, 
                      weight_init="kan_mimetic")
    model = _create_kat_transformer('kat_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def kat_tiny_swish_patch16_224(pretrained: bool = False, **kwargs) -> KATVisionTransformer:
    """ KAT-Tiny with rational activations (ViT-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, 
                      act_layer=KAT_Group, 
                      act_init='swish',
                      mlp_layer=KAN, 
                      weight_init="kan_mimetic")
    model = _create_kat_transformer('kat_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def kat_small_patch16_224(pretrained: bool = False, **kwargs) -> KATVisionTransformer:
    """ KAT-Small with rational activations (ViT-S/16)"""
    model_args = dict(patch_size=16, 
                      embed_dim=384, 
                      depth=12, 
                      num_heads=6, 
                      act_init='swish',
                      act_layer=KAT_Group, 
                      mlp_layer=KAN, 
                      weight_init="kan_mimetic")
    model = _create_kat_transformer('kat_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def kat_small_gelu_patch16_224(pretrained: bool = False, **kwargs) -> KATVisionTransformer:
    """ KAT-Small with rational activations (ViT-S/16)"""
    model_args = dict(patch_size=16, 
                      embed_dim=384, 
                      depth=12, 
                      num_heads=6, 
                      act_init='gelu',
                      act_layer=KAT_Group, 
                      mlp_layer=KAN, weight_init="kan_mimetic") # , init_values=1e-5
    model = _create_kat_transformer('kat_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def kat_small_swish_patch16_224(pretrained: bool = False, **kwargs) -> KATVisionTransformer:
    """ KAT-Small with rational activations (ViT-S/16)"""
    model_args = dict(patch_size=16, 
                      embed_dim=384, 
                      depth=12, num_heads=6, 
                      act_init='swish',
                      act_layer=KAT_Group, 
                      mlp_layer=KAN, weight_init="kan_mimetic") # , init_values=1e-5
    model = _create_kat_transformer('kat', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def kat_base_patch16_224(pretrained: bool = False, **kwargs) -> KATVisionTransformer:
    """ KAT-Base with rational activations (ViT-B/16)"""
    model_args = dict(patch_size=16, 
                      embed_dim=768, 
                      depth=12, 
                      num_heads=12, 
                      act_layer=KAT_Group, 
                      mlp_layer=KAN, 
                      act_init='swish',
                      weight_init="kan_mimetic")
    model = _create_kat_transformer('kat_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def kat_base_gelu_patch16_224(pretrained: bool = False, **kwargs) -> KATVisionTransformer:
    """ KAT-Base with rational activations (ViT-B/16)"""
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                      act_layer=KAT_Group, 
                      mlp_layer=KAN, 
                      act_init='gelu',
                      weight_init="kan_mimetic")
    model = _create_kat_transformer('kat_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def kat_base_swish_patch16_224(pretrained: bool = False, **kwargs) -> KATVisionTransformer:
    """ KAT-Base with rational activations (ViT-B/16)"""
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                      act_layer=KAT_Group, 
                      mlp_layer=KAN, 
                      act_init='swish',
                      weight_init="kan_mimetic")
    model = _create_kat_transformer('kat_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
