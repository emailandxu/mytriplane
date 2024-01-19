from .transformer import TriplaneTransformer
from .encoders.dino_wrapper import DinoWrapper
from .camera_embedder import CameraEmbedder
from .generator import LRMGenerator

kwargs_camera_embedder = {'raw_dim': 16, 'embed_dim': 1024}

kwargs_dino_model = {"model_name": "facebook/dino-vitb16", "freeze": False}

kwargs_transformer_model = {
    "inner_dim": 1024,
    "num_layers": 12,
    "num_heads": 16,
    "image_feat_dim": 768,
    "camera_embed_dim": 1024,
    "triplane_low_res": 32,
    "triplane_high_res": 64,
    "triplane_dim": 40,
}
