import numpy as np

def apply_filter(filter, img):
    img = img.permute(0, 3, 1, 2)
    img = filter(img)
    return img.permute(0, 2, 3, 1).contiguous()

def create_mipmap(texs, max_mip_level=4, filter=None):
    import torch
    from torch.nn import functional as F
    import torchvision.transforms as transforms

    if filter is None:
        filter = transforms.GaussianBlur(5, sigma=1)

    # Create a Mipmap pyramid with 4 levels
    mipmap_levels = [texs]
    for i in range(max_mip_level):
        b, h, w, c = mipmap_levels[-1].shape
        res = h
        # Apply Gaussian blur for downsampling
        filtered = apply_filter(
            transforms.Compose([filter, transforms.Resize(res//2)]),
            mipmap_levels[-1]
        )
        mipmap_levels.append(filtered)
    
    mipmap_levels.pop(0)
    return mipmap_levels

def pad_image_to_size(original_image, padded_height, padded_width):
    # Get the dimensions of the original image
    original_height, original_width, channels = original_image.shape

    # Calculate the padding values for the top, bottom, left, and right sides
    pad_top = (padded_height - original_height) // 2
    pad_bottom = padded_height - original_height - pad_top
    pad_left = (padded_width - original_width) // 2
    pad_right = padded_width - original_width - pad_left

    # Use numpy.pad to pad the image
    padded_image = np.pad(original_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')

    return padded_image
