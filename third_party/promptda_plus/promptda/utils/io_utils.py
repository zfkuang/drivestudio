from typing import Tuple
import torch
import imageio
import cv2
import numpy as np  
from promptda.utils.vis_utils import colorize_depth_maps
import os

def load_image(image_path: str,
               tar_size: Tuple[int, int] = (630, 840),
               ) -> torch.Tensor: 
    '''
    Load image and resize to target size.
    Args:
        image_path: Path to input image.
        tar_size: Target size (h, w).
    Returns:
        image: Image tensor with shape (1, 3, h, w).
    '''
    image = imageio.imread(image_path)
    # assert image.shape 是 720 x 1280
    assert image.shape == (720, 1280, 3), f'Image shape is {image.shape}, expected (720, 1280, 3)'
    image = image[:, 160:-160, :]
    image = cv2.resize(image, tar_size[::-1], interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return image

def load_depth(depth_path: str,
               tar_size: Tuple[int, int] = (630, 840),
               num_samples: int = 1500) -> torch.Tensor:
    '''
    depth is in mm and stored in 16-bit PNG
    '''
    depth = imageio.imread(depth_path)
    assert depth.shape == (720, 1280), f'Depth shape is {depth.shape}, expected (720, 1280)'
    depth = (depth / 1000.).astype(np.float32)
    depth = depth[:, 160:-160]
    depth = cv2.resize(depth, tar_size[::-1], interpolation=cv2.INTER_NEAREST)
    if (depth > 0.1).sum() > num_samples:
        height, width = depth.shape
        depth = depth.reshape(-1)
        nonzero_index = np.array(list(np.nonzero(depth>0.1))).squeeze()
        index = np.random.permutation(nonzero_index)[:num_samples]
        sample_mask = np.ones_like(depth)
        sample_mask[index] = 0.
        depth[sample_mask.astype(bool)] = 0.
        depth = depth.reshape(height, width)
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    return depth

def plot_depth(image: torch.Tensor, depth: torch.Tensor, prompt_depth: torch.Tensor, output_path: str) -> None:
    depth_min, depth_max = depth.min().item(), depth.max().item()
    vis_img = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    prompt_depth = prompt_depth[0, 0].detach().cpu().numpy()
    # prompt_depth = cv2.resize(prompt_depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    prompt_depth = cv2.resize(prompt_depth, (vis_img.shape[1], vis_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis_prompt_depth = colorize_depth_maps(prompt_depth, min_depth=depth_min, max_depth=depth_max, cmap='Spectral')
    vis_depth = colorize_depth_maps(depth[0, 0].detach().cpu().numpy(), min_depth=depth_min, max_depth=depth_max, cmap='Spectral')
    vis_img = np.concatenate([vis_img, vis_depth], axis=1)
    vis_img = np.concatenate([vis_img, vis_prompt_depth], axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.imwrite(output_path, vis_img)