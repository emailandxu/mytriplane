import torch
from torch import nn

import nvdiffrast.torch


class TriMipEncoding(nn.Module):
    def __init__(
        self,
        n_levels: int = 0,
        plane_size: int = 512,
        feature_dim: int = 16,
        include_xyz: bool = False,
    ):
        super(TriMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size
        self.feature_dim = feature_dim
        self.include_xyz = include_xyz

        # self.register_parameter(
        #     "fm",
        #     nn.Parameter(torch.zeros(3, plane_size, plane_size, feature_dim)),
        # )
        # self.init_parameters()
        self.dim_out = self.feature_dim * 3 + 3 if include_xyz else self.feature_dim * 3

    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def forward(self, x, fm, level=None):

        # x in [0,1], level in [0,max_level]
        # x is Nx3, level is Nx1
        if 0 == x.shape[0]:
            return torch.zeros([x.shape[0], self.feature_dim * 3]).to(x)
        decomposed_x = torch.stack(
            [
                x[:, None, [1, 2]],
                x[:, None, [0, 2]],
                x[:, None, [0, 1]],
            ],
            dim=0,
        )  # 3xNx1x2
        if 0 == self.n_levels:
            level = None
        else:
            # assert level.shape[0] > 0, [level.shape, x.shape]
            torch.stack([level, level, level], dim=0)
            level = torch.broadcast_to(level, decomposed_x.shape[:3]).contiguous()
        # print(decomposed_x.min(), decomposed_x.max())

        SAMPLE_POINT_CHANNEL = 1
        if decomposed_x.shape[SAMPLE_POINT_CHANNEL] > 1e6:
            # nvdiffrast texture doesn't support high resolution texture mapping
            # so I seperate the uv into 10 parts and concate the sampling feature
            parts = 10
            partion = decomposed_x.shape[SAMPLE_POINT_CHANNEL] // parts
            indices = [partion * i for i in range(parts+1)]
            if decomposed_x.shape[SAMPLE_POINT_CHANNEL] % parts != 0:
                indices.append(None)

            enc = torch.concat(
                [
                    nvdiffrast.torch.texture(
                        fm,
                        decomposed_x[:, indices[index-1] : indices[index]].contiguous(),
                        mip_level_bias=level,
                        boundary_mode="clamp",
                        max_mip_level=self.n_levels - 1 if self.n_levels > 0 else None,
                    )  # 3xNx1xC
                    for index in range(1, len(indices))
                ],
                dim=SAMPLE_POINT_CHANNEL,
            )

            assert enc.shape[SAMPLE_POINT_CHANNEL] == decomposed_x.shape[SAMPLE_POINT_CHANNEL]

        else:
            enc = nvdiffrast.torch.texture(
                fm,
                decomposed_x,
                mip_level_bias=level,
                boundary_mode="clamp",
                max_mip_level=self.n_levels - 1 if self.n_levels > 0 else None,
            )  # 3xNx1xC


        enc = (
            enc.permute(1, 2, 0, 3)
            .contiguous()
            .view(
                x.shape[0],
                self.feature_dim * 3,
            )
        )  # Nx(3C)
        if self.include_xyz:
            enc = torch.cat([x, enc], dim=-1)
        return enc
