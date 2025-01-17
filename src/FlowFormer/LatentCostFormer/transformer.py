import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum

from ..common import (
    MLP,
    FeedForward,
    MultiHeadAttention,
    pyramid_retrieve_tokens,
    retrieve_tokens,
    sampler,
    sampler_gaussian_fix,
)
from ..encoders import twins_svt_large, twins_svt_large_context
from ..position_encoding import LinearPositionEncoding, PositionEncodingSine
from ..utils import bilinear_sampler, coords_grid, upflow8
from .cnn import BasicEncoder
from .decoder import MemoryDecoder
from .encoder import MemoryEncoder
from .twins import PosConv


class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == "twins":
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == "basicencoder":
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn="instance")
            raise NotImplementedError
        
        self.cls_head = nn.Linear(1024, 2, bias=True)

    def forward(self, image1, image2, output=None, flow_init=None):
        # # Following https://github.com/princeton-vl/RAFT/
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)

        h, w = context.shape[-2:]
        cost_memory = self.memory_encoder(image1, image2, data, context)
        cost_memory = rearrange(cost_memory, "(b s) n c -> b s n c", s=h*w)
        cost_memory = cost_memory.mean(dim=1)
        cost_memory = rearrange(cost_memory, "b n c -> b (n c)")
        pred = self.cls_head(cost_memory)
        return pred

        # flow_predictions = self.memory_decoder(
        #     cost_memory, context, data, flow_init=flow_init
        # )

        # return flow_predictions
