import numpy as np
import os
import sys
import argparse
import time
import cv2
import torch
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor
sys.path.append('third_party/promptda_plus')
from promptda.model import PromptDepthAnythingPlus

def ensure_multiple_of(x, multiple_of=14):
    return int(x // multiple_of * multiple_of)

def create_sparse_depth_map(sparse_coords, depth_values, height, width, max_points=None):
    if max_points is not None and max_points < len(depth_values):
        indices = np.random.choice(len(depth_values), max_points, replace=False)
        sparse_coords = sparse_coords[indices]
        depth_values = depth_values[indices]
    valid_depth_mask = (depth_values > 0) & (depth_values < 40)
    sparse_coords = sparse_coords[valid_depth_mask]
    depth_values = depth_values[valid_depth_mask]
    if len(depth_values) == 0:
        return np.zeros((height, width), dtype=np.float32)
    sparse_depth = np.zeros((height, width), dtype=np.float32)
    coords_y = sparse_coords[:, 1].astype(int).clip(0, height-1)
    coords_x = sparse_coords[:, 0].astype(int).clip(0, width-1)
    sparse_depth[coords_y, coords_x] = depth_values
    return sparse_depth

def create_dense_depth_map(sparse_coords, depth_values, height, width, n_neighbors=5):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    knn.fit(sparse_coords, depth_values)
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    all_coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    dense_depth = knn.predict(all_coords).reshape(height, width)
    return dense_depth

def main(args):
    # Parse the scene list from the split file
    with open(args.split_file, 'r') as f:
        lines = f.readlines()
    
    scenes = []
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split(',')
            if len(parts) >= 2:
                scene_id = parts[0]
                seg_name = parts[1]
                start_timestep = int(parts[2])
                end_timestep = int(parts[3])
                scenes.append((scene_id, seg_name, start_timestep, end_timestep))
    
    # Load the PromptDA model
    model = PromptDepthAnythingPlus(geometry_type='disparity', model_path=args.promptda_path)
    model.eval()
    model.cuda()
    
    for scene_id, seg_name, start_timestep, end_timestep in scenes:
        scene_id = "%03d" % int(scene_id)
        waymo_datadir = os.path.join(args.root_dir, f"{scene_id}")
        
        image_dir = os.path.join(waymo_datadir, 'images')
        projected_depth_dir = os.path.join(waymo_datadir, 'projected_depth')
        projected_depth_vis_dir = os.path.join(waymo_datadir, 'projected_depth_vis')
        completed_depth_dir = os.path.join(waymo_datadir, 'completed_depth')
        completed_depth_vis_dir = os.path.join(waymo_datadir, 'completed_depth_vis')
        os.makedirs(completed_depth_dir, exist_ok=True)
        os.makedirs(completed_depth_vis_dir, exist_ok=True)
        
        print(f"Processing scene {scene_id} ({seg_name})...")
        
        for timestep in range(start_timestep, end_timestep, 10):
            for cam_id in range(args.num_cameras):
                image = Image.open(os.path.join(image_dir, f"{timestep:03d}_{cam_id}.jpg"))
                image = image = (np.array(image) / 255.).astype(np.float32)
                projected_depth = np.load(os.path.join(projected_depth_dir, f"{timestep:03d}_{cam_id}.npy"), allow_pickle=True).item()
                sparse_points_2d = projected_depth['depth_pixel_coords']
                sparse_depth = projected_depth['depth_values']
                
                H, W = image.shape[:2]
                sparse_points_2d[:, 0] = sparse_points_2d[:, 0] * W
                sparse_points_2d[:, 1] = sparse_points_2d[:, 1] * H
                prompt_depth = create_sparse_depth_map(sparse_points_2d, sparse_depth, H, W, max_points=10000)
                    
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

                prompt_depth[prompt_depth>0] = 1. / (1e-3 + prompt_depth[prompt_depth>0])
                    
                with torch.no_grad():
                    with torch.autocast(dtype=torch.bfloat16, device_type='cuda'):
                        outputs = model.inference(image=image, prompt_depth=prompt_depth)

                depth = outputs['disparity']
                
                # Save results
                # Save depth map
                real_depth = 1 / (1e-3 + depth.cpu().numpy())
                np.save(os.path.join(completed_depth_dir, f"{timestep:03d}_{cam_id}.npy"), real_depth)
                                    
                # Save visualizations
                depth_np = depth.cpu().numpy()
                if depth_np.ndim > 2:
                    depth_np = depth_np.squeeze()
                # if geometry_type == "disparity":
                #     depth_np[depth_np>0] = 1 / (1e-3 + depth_np[depth_np>0]) # we need to convert disparity to depth for visualization
                depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
                depth_vis = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                cv2.imwrite(os.path.join(completed_depth_vis_dir, f"{timestep:03d}_{cam_id}.jpg"), depth_vis)

                prompt_np = prompt_depth.cpu().numpy().squeeze()
                if prompt_np.ndim > 2:
                    prompt_np = prompt_np.squeeze()
                # if geometry_type == "disparity":
                #     prompt_np[prompt_np>0] = 1 / prompt_np[prompt_np>0] # we need to convert disparity to depth for visualization
                prompt_norm = (prompt_np - prompt_np.min()) / (prompt_np.max() - prompt_np.min())
                prompt_vis = cv2.applyColorMap((prompt_norm * 65535).astype(np.uint8), cv2.COLORMAP_INFERNO)
                cv2.imwrite(os.path.join(completed_depth_vis_dir, f"{timestep:03d}_{cam_id}_prompt_vis.jpg"), prompt_vis)
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--promptda_path', type=str, required=True, help='Path to the PromptDA model')
    parser.add_argument('--split_file', type=str, required=True, help='File containing scene indices')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of Waymo dataset')
    parser.add_argument('--num_cameras', type=int, default=5, help='Number of cameras')
    args = parser.parse_args()
    
    main(args)
