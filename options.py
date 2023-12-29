import json
rendering_kwargs = {
    "depth_resolution": 48,
    "depth_resolution_importance": 48,
    "ray_start": 1.0,
    "ray_end": 3.0,
    "box_warp": 1,
    "avg_camera_radius": 2.7,
    "avg_camera_pivot": [
        0,
        0,
        0.2
    ],
    "triplane_output_res": 64,
    "image_resolution": 64,
    "disparity_space_sampling": False,
    "clamp_mode": "softplus",
    "superresolution_module": "networks.SuperresolutionHybrid8XDC",
    "c_gen_conditioning_zero": False,
    "c_scale": 1.0,
    "superresolution_noise_mode": "none",
    "density_reg": 0.25,
    "density_reg_p_dist": 0.004,
    "reg_type": "l1",
    "decoder_lr_mul": 1.0,
    "sr_antialias": True
}

dataset_kwargs = {
    # "rootdir":"data/2d/sculpture_bust_of_roza_loewenfeld",
    # "rootdir":"data/2d/rmin15rmax22",
    # "rootdir": "data/2d/rmin22rmax27",
    "rootdir": "data/2d/jupiter",
    "resolution": rendering_kwargs["image_resolution"]
}
