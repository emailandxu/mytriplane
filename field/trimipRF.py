import math
from typing import Callable

import torch
from torch import Tensor, nn
import tinycudann as tcnn

from .trimipEnc import TriMipEncoding
from .truncExp import trunc_exp


class TriMipRF(nn.Module):
    def __init__(
        self,
        n_levels: int = 0,
        planes = None,
        plane_size: int = 512,
        feature_dim: int = 16,
        geo_feat_dim: int = 15,
        net_depth_base: int = 2,
        net_depth_color: int = 4,
        net_width: int = 128,
    ) -> None:
        super().__init__()
        self.plane_size = plane_size
        self.log2_plane_size = math.log2(plane_size)
        self.geo_feat_dim = geo_feat_dim

        self.encoding = TriMipEncoding(n_levels, plane_size, feature_dim)
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        self.mlp_base = tcnn.Network(
            n_input_dims=self.encoding.dim_out,
            n_output_dims=geo_feat_dim + 1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_base,
            },
        )
        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + geo_feat_dim, # with direction
            # n_input_dims=geo_feat_dim, # without direction
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_color,
            },
        )

    def density_activation(self, x):
        return trunc_exp(x - 1)

    def query_density(
        self, x: Tensor, planes:Tensor, return_feat: bool = False
    ):
        # print("query_density", x.shape)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        enc = self.encoding(
            x.view(-1, 3),
            planes
            # level=level.view(-1, 1),
        )
        x = (
            self.mlp_base(enc)
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        return {
            "density": density,
            "feature": base_mlp_out if return_feat else None,
        }

    def query_rgb(self, dir, embedding):
        # dir in [-1,1]
        dir = (dir + 1.0) / 2.0  # SH encoding must be in the range [0, 1]
        d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
        h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1) # with direction
        # h = embedding.view(-1, self.geo_feat_dim) # without direction
        rgb = (
            self.mlp_head(h)
            .view(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        return {"rgb": rgb}
