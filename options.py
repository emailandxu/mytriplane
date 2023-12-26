
import json
rendering_kwargs = json.loads("""{
    "depth_resolution": 48,
    "depth_resolution_importance": 48,
    "ray_start": 2.25,
    "ray_end": 3.3,
    "box_warp": 1,
    "avg_camera_radius": 2.7,
    "avg_camera_pivot": [
        0,
        0,
        0.2
    ],
    "image_resolution": 512,
    "disparity_space_sampling": false,
    "clamp_mode": "softplus",
    "superresolution_module": "training.superresolution.SuperresolutionHybrid8XDC",
    "c_gen_conditioning_zero": false,
    "c_scale": 1.0,
    "superresolution_noise_mode": "none",
    "density_reg": 0.25,
    "density_reg_p_dist": 0.004,
    "reg_type": "l1",
    "decoder_lr_mul": 1.0,
    "sr_antialias": true
}""")

dataset_kwargs = {
    "rootdir":"data/rendering/data/2d/sculpture_bust_of_roza_loewenfeld"
}