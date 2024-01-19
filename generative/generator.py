# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn

from .encoders.dino_wrapper import DinoWrapper
from .transformer import TriplaneTransformer
from .camera_embedder import CameraEmbedder


class LRMGenerator(nn.Module):
    """
    Full model of the large reconstruction model.
    """
    def __init__(self, kwargs_camera_embedder, kwargs_dino, kwargs_transformer):
        super().__init__()
        # modules
        self.encoder = DinoWrapper(**kwargs_dino)
        self.camera_embedder = CameraEmbedder(**kwargs_camera_embedder)
        self.transformer = TriplaneTransformer(**kwargs_transformer)


    def forward_planes(self, image, camera):
        # image: [N, C_img, H_img, W_img]
        # camera: [N, D_cam_raw]
        assert image.shape[0] == camera.shape[0], "Batch size mismatch for image and camera"
        N = image.shape[0]
        # encode image
        image_feats = self.encoder(image)

        # embed camera
        camera_embeddings = self.camera_embedder(camera)

        # transformer generating planes
        planes = self.transformer(image_feats, camera_embeddings)
        assert planes.shape[0] == N, "Batch size mismatch for planes"
        assert planes.shape[1] == 3, "Planes should have 3 channels"

        return planes

    # def forward(self, image, source_camera, render_cameras, render_size: int):
    #     # image: [N, C_img, H_img, W_img]
    #     # source_camera: [N, D_cam_raw]
    #     # render_cameras: [N, M, D_cam_render]
    #     # render_size: int
    #     assert image.shape[0] == source_camera.shape[0], "Batch size mismatch for image and source_camera"
    #     assert image.shape[0] == render_cameras.shape[0], "Batch size mismatch for image and render_cameras"
    #     N, M = render_cameras.shape[:2]

    #     planes = self.forward_planes(image, source_camera)

    #     # render target views
    #     render_results = self.synthesizer(planes, render_cameras, render_size)
    #     assert render_results['images_rgb'].shape[0] == N, "Batch size mismatch for render_results"
    #     assert render_results['images_rgb'].shape[1] == M, "Number of rendered views should be consistent with render_cameras"

    #     return {
    #         'planes': planes,
    #         **render_results,
    #     }
