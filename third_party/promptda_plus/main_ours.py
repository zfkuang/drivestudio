import torch
import tyro
import numpy as np
import cv2
from safetensors.torch import load_file
from sklearn.neighbors import KNeighborsRegressor
import os
from PIL import Image

from promptda.model import PromptDepthAnythingPlus
from promptda.utils.io_utils import load_image, load_depth, plot_depth

def ensure_multiple_of(x, multiple_of=14):
    return int(x // multiple_of * multiple_of)

def create_dense_depth_map(sparse_coords, depth_values, height, width, n_neighbors=5):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    knn.fit(sparse_coords, depth_values)
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    all_coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    dense_depth = knn.predict(all_coords).reshape(height, width)
    return dense_depth

def create_sparse_depth_map(sparse_coords, depth_values, height, width, max_points=None):
    """Create a sparse depth map with values only at LiDAR points.
    Args:
        sparse_coords: Nx2 array of (x,y) coordinates
        depth_values: N array of depth values
        height, width: Size of output depth map
        max_points: If set, randomly sample this many points (None = use all points)
    """
    if max_points is not None and max_points < len(depth_values):
        # Randomly sample points
        indices = np.random.choice(len(depth_values), max_points, replace=False)
        sparse_coords = sparse_coords[indices]
        depth_values = depth_values[indices]
        
    # Filter out depth values that are larger than 40 or smaller than 0
    valid_depth_mask = (depth_values > 0) & (depth_values < 40)
    sparse_coords = sparse_coords[valid_depth_mask]
    depth_values = depth_values[valid_depth_mask]
    
    # Check if we still have valid points after filtering
    if len(depth_values) == 0:
        print("Warning: No valid depth points after filtering (all points were outside the 0-40 range)")
        # Return empty depth map if no valid points
        return np.zeros((height, width), dtype=np.float32)

    sparse_depth = np.zeros((height, width), dtype=np.float32)
    coords_y = sparse_coords[:, 1].astype(int).clip(0, height-1)
    coords_x = sparse_coords[:, 0].astype(int).clip(0, width-1)
    sparse_depth[coords_y, coords_x] = depth_values
    return sparse_depth

def main(
        scene_name: str = "11489533038039664633_4820_000_4840_000",
        root_dir: str = "../../data/waymo/devtool/data/training_GS/",
        model_path: str = 'model.safetensors',
        geometry_type: str = 'disparity',
        depth_method: str = 'sparse',  # Options: 'knn' or 'sparse'
        max_sparse_points: int = 1000,  # Only used if depth_method='sparse'
    ) -> None:
    model = PromptDepthAnythingPlus(geometry_type=geometry_type, model_path=model_path)

    # Setup directories
    root_dir = os.path.join(root_dir, scene_name)
    output_dir = os.path.join(root_dir, "depth_promptda_plus")
    os.makedirs(output_dir, exist_ok=True)
    input_image_dir = os.path.join(root_dir, "images_2")
    input_depth_dir = os.path.join(root_dir, "frame_proj3D")
    image_list = sorted(os.listdir(input_image_dir))

    for image_file in image_list:
        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(input_image_dir, image_file)
        
        # Load and process image
        image = Image.open(image_path)
        width, height = image.size
        image = (np.array(image) / 255.).astype(np.float32)

        # Process LiDAR data
        lidar_name = os.path.join(input_depth_dir, image_name + "_z_depthmap.npy")
        lidar_mask_name = os.path.join(input_depth_dir, image_name + "_mask.npy")
        lidar_selected_name = os.path.join(input_depth_dir, image_name + "_selected.npy")
        
        if not os.path.exists(lidar_name):
            continue
            
        input_depth_lidar = np.load(lidar_name).reshape(-1)
        mask_lidar = np.load(lidar_mask_name).reshape(-1)
        selected_lidar = np.load(lidar_selected_name).reshape(-1, 2)
        
        input_depth_lidar = input_depth_lidar[mask_lidar]
        selected_lidar = selected_lidar[mask_lidar]
        
        xyz_proj_filtered = selected_lidar.copy()
        xyz_proj_filtered[:, 0] = xyz_proj_filtered[:, 0].clip(0, 1) * width 
        xyz_proj_filtered[:, 1] = xyz_proj_filtered[:, 1].clip(0, 1) * height
        
        # Create depth map using selected method
        if depth_method == 'knn':
            prompt_depth = create_dense_depth_map(xyz_proj_filtered, input_depth_lidar, height, width)
        else:  # sparse
            prompt_depth = create_sparse_depth_map(xyz_proj_filtered, input_depth_lidar, height, width, max_sparse_points)
            
        # Resize if needed
        max_size = min(800, max(image.shape))
        max_size = max_size // 14 * 14
        if max(image.shape) > max_size:
            h, w = image.shape[:2]
            scale = max_size / max(h, w)
            tar_h = ensure_multiple_of(h * scale)
            tar_w = ensure_multiple_of(w * scale)
            image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_AREA)
            prompt_depth = cv2.resize(prompt_depth, (tar_w, tar_h), interpolation=cv2.INTER_NEAREST)

        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
        prompt_depth = torch.from_numpy(prompt_depth).unsqueeze(0).unsqueeze(0).float().cuda()

        if geometry_type == "disparity":
            prompt_depth[prompt_depth>0] = 1. / (1e-3 + prompt_depth[prompt_depth>0])
            
        

        # Model inference
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            with torch.autocast(dtype=torch.bfloat16, device_type='cuda'):
                outputs = model.inference(image=image, prompt_depth=prompt_depth)

        depth = outputs[f"{geometry_type}"]
        
        # Save results
        base_name = image_name
        
        # Save depth map
        if geometry_type == "disparity": # save real depth into the npy file
            real_depth = 1 / (1e-3 + depth.cpu().numpy())
            np.save(os.path.join(output_dir, f"{base_name}_depth.npy"), real_depth)
            # np.save(os.path.join(output_dir, f"{base_name}_prompt_depth.npy"), prompt_depth.cpu().numpy().squeeze())
            
        # import pdb
        # pdb.set_trace()
        
        # Save visualizations
        depth_np = depth.cpu().numpy()
        if depth_np.ndim > 2:
            depth_np = depth_np.squeeze()
        # if geometry_type == "disparity":
        #     depth_np[depth_np>0] = 1 / (1e-3 + depth_np[depth_np>0]) # we need to convert disparity to depth for visualization
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        depth_vis = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_depth_vis.jpg"), depth_vis)

        prompt_np = prompt_depth.cpu().numpy().squeeze()
        if prompt_np.ndim > 2:
            prompt_np = prompt_np.squeeze()
        # if geometry_type == "disparity":
        #     prompt_np[prompt_np>0] = 1 / prompt_np[prompt_np>0] # we need to convert disparity to depth for visualization
        prompt_norm = (prompt_np - prompt_np.min()) / (prompt_np.max() - prompt_np.min())
        prompt_vis = cv2.applyColorMap((prompt_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_prompt_vis.jpg"), prompt_vis)
        
        # import pdb; pdb.set_trace()
        print(f"Saved depth maps and visualizations for {image_name} to {output_dir}")

if __name__ == "__main__":
    tyro.cli(main) 