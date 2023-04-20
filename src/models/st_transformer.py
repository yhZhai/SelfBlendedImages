import copy
from typing import Callable

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from einops import rearrange
from einops.layers.torch import Rearrange


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class SpatioTemporalTransformerLayer(nn.Module):
    def __init__(
        self, d_model: int = 512, nhead: int = 8, activation: str = "gelu"
    ) -> None:
        super().__init__()

        self.pe = PositionalEncoding(d_model)
        self.spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, activation=_get_activation_fn(activation)
        )
        self.temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, activation=_get_activation_fn(activation)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x (Tensor): b, t, num_landmarks, d_model
        """
        b, t, n, d = x.shape
        x = rearrange(x, "b t n d -> n (b t) d")
        x = self.pe(x)
        x = self.spatial_layer(x)
        x = rearrange(x, "n (b t) d -> t (b n) d", b=b, t=t, n=n)
        x = self.temporal_layer(x)
        x = rearrange(x, "t (b n) d -> b t n d", b=b, t=t, n=n)
        return x


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        activation: str = "gelu",
        patch_size: int = 32,
        in_channels: int = 3,
        mid_channels: int = 64,
        nframes: int = 8,
        num_landmarks: int = 81,
        num_layers: int = 6,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        assert d_model % 4 == 0

        self.d_model = d_model
        self.patch_dim = patch_size * patch_size * in_channels

        self.to_patch_embedding = nn.Sequential(
            # Rearrange("b t n c h w-> (b t n) c h w", h=patch_size, w=patch_size),
            # nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=2, padding=2),
            # nn.GELU(),
            # Rearrange("(b t n) c h w -> b t n (c h w)", t=nframes, n=num_landmarks),
            Rearrange("b t n c h w-> b t n (c h w)", h=patch_size, w=patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.mlp = nn.Linear(d_model * 2, d_model)
        self.transformer_layer = SpatioTemporalTransformerLayer(
            d_model=d_model, nhead=nhead, activation=activation
        )
        self.transformer = _get_clones(self.transformer_layer, num_layers)

        self.pred = nn.Linear(d_model, num_classes)

        div_term = torch.exp(
            torch.arange(0, d_model, 4).float() * (-np.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, landmark_loc, patches) -> Tensor:
        """
        landmark_loc: b, t, num_landmarks, 2 (x and y coordinates)
        patches: b, t, num_landmarks, 3, h, w
        """
        b, t, n, _, h, w = patches.shape
        patches = self.to_patch_embedding(patches)  # b t n, d_model

        landmark_loc_pe = self._pos_enc(landmark_loc)
        patches = torch.cat([patches, landmark_loc_pe], dim=3)
        patches = self.mlp(patches)

        for layer in self.transformer:
            patches = layer(patches)
        
        patches = rearrange(patches, "b t n d -> b (t n) d")
        patches = patches.mean(dim=1)
        pred = self.pred(patches)
        return pred

    def _pos_enc(self, x):
        x_sin = torch.sin(x[:, :, :, 0, None] * self.div_term)
        x_cos = torch.cos(x[:, :, :, 0, None] * self.div_term)
        y_sin = torch.sin(x[:, :, :, 1, None] * self.div_term)
        y_cos = torch.cos(x[:, :, :, 1, None] * self.div_term)
        pe = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=-1)
        return pe


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)
