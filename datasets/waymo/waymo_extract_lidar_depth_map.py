import numpy as np
import os
import sys
import argparse
import time
import cv2
import torch
import omegaconf
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
sys.path.append(os.getcwd())
from datasets.waymo.waymo_sourceloader import WaymoPixelSource, WaymoLiDARSource

def visualize_depth_points_on_image(image, depth_2D_pos, depth_values):
    image = (image.cpu().numpy() * 255.0).astype(np.uint8)
    depth_2D_pos = depth_2D_pos.cpu().numpy()
    depth_values = depth_values.cpu().numpy()
    vis_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for pt, depth in zip(depth_2D_pos, depth_values):
        pt = pt.astype(int)
        depth_color = plt.cm.viridis(depth / 40.0)
        # Convert depth color to BGR
        depth_color = (depth_color[2] * 255, depth_color[1] * 255, depth_color[0] * 255)
        cv2.circle(vis_img, (pt[0], pt[1]), 2, depth_color, -1)
    return vis_img

device = torch.device("cuda")
def main(args):
    # Parse the scene list from the split file
    
    data_config = omegaconf.OmegaConf.load(args.data_config_path).data
    pixel_data_config = data_config.pixel_source
    lidar_data_config = data_config.lidar_source
    
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
        
    for scene_id, seg_name, start_timestep, end_timestep in scenes:
        scene_id = "%03d" % int(scene_id)
        waymo_datadir = os.path.join(args.root_dir, f"{scene_id}")
        
        lidar_dir = os.path.join(waymo_datadir, 'lidar')
        image_dir = os.path.join(waymo_datadir, 'images')
        intrinsics_dir = os.path.join(waymo_datadir, 'intrinsics')
        extrinsics_dir = os.path.join(waymo_datadir, 'extrinsics')
        ego_pose_dir = os.path.join(waymo_datadir, 'ego_pose')
        projected_depth_dir = os.path.join(waymo_datadir, 'projected_depth')
        projected_depth_vis_dir = os.path.join(waymo_datadir, 'projected_depth_vis')
        os.makedirs(projected_depth_dir, exist_ok=True)
        os.makedirs(projected_depth_vis_dir, exist_ok=True)
        
        print(f"Processing scene {scene_id} ({seg_name})...")

        # end_timestep = start_timestep + 10

        lidar_source = WaymoLiDARSource(
            lidar_data_config=lidar_data_config,
            data_path=waymo_datadir,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            device=torch.device("cuda")
        )
        pixel_source = WaymoPixelSource(
            dataset_name="waymo",
            pixel_data_config=pixel_data_config,
            data_path=waymo_datadir,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            device=torch.device("cuda")
        )
        pixel_source.to(device)
        lidar_source.to(device)

        for cam in pixel_source.camera_data.values():
            lidar_depth_maps = []
            for frame_idx in tqdm(
                range(len(cam)), 
                desc="Projecting lidar pts on images for camera {}".format(cam.cam_name),
                dynamic_ncols=True
            ):
                normed_time = pixel_source.normalized_time[frame_idx]
                
                # get lidar depth on image plane
                closest_lidar_idx = lidar_source.find_closest_timestep(normed_time)
                lidar_infos = lidar_source.get_lidar_rays(closest_lidar_idx)
                lidar_points = (
                    lidar_infos["lidar_origins"]
                    + lidar_infos["lidar_viewdirs"] * lidar_infos["lidar_ranges"]
                )
                
                
                # project lidar points to the image plane
                if cam.undistort:
                    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                                cam.intrinsics[frame_idx].cpu().numpy(),
                                cam.distortions[frame_idx].cpu().numpy(),
                                (cam.WIDTH, cam.HEIGHT),
                                alpha=1,
                            )
                    intrinsic_4x4 = torch.nn.functional.pad(
                            torch.from_numpy(new_camera_matrix), (0, 1, 0, 1)
                        ).to(device)
                else:
                    intrinsic_4x4 = torch.nn.functional.pad(
                        cam.intrinsics[frame_idx], (0, 1, 0, 1)
                    )
                intrinsic_4x4[3, 3] = 1.0
                lidar2img = intrinsic_4x4 @ cam.cam_to_worlds[frame_idx].inverse()
                lidar_points = (
                    lidar2img[:3, :3] @ lidar_points.T + lidar2img[:3, 3:4]
                ).T # (num_pts, 3)
                
                depth = lidar_points[:, 2]
                if "lidar_cp_points" in lidar_infos:
                    cp_points = lidar_infos["lidar_cp_points"]
                    valid_mask_1 = cp_points[..., 0] == (cam.cam_id+1)
                    valid_mask_2 = (cp_points[..., 3] == (cam.cam_id+1)) & ~valid_mask_1
                    valid_mask = valid_mask_1 | valid_mask_2
                    cam_points_1 = cp_points[valid_mask_1, 1:3] 
                    cam_points_2 = cp_points[valid_mask_2, 4:6]
                    depth_1 = depth[valid_mask_1]
                    depth_2 = depth[valid_mask_2]
                    depth = torch.cat([depth_1, depth_2], dim=0)
                    cam_points = torch.cat([cam_points_1, cam_points_2], dim=0)
                    cam_points[:, 0] *= cam.WIDTH
                    cam_points[:, 1] *= cam.HEIGHT
                else:
                    cam_points = lidar_points[:, :2] / (depth.unsqueeze(-1) + 1e-6) # (num_pts, 2)
                    valid_mask = (
                        (cam_points[:, 0] >= 0)
                        & (cam_points[:, 0] < cam.WIDTH)
                        & (cam_points[:, 1] >= 0)
                        & (cam_points[:, 1] < cam.HEIGHT)
                        & (depth > 0.05) 
                        & (depth < 80)
                    ) # (num_pts, )
                    # if cam.cam_id == 2:
                    #     import pdb; pdb.set_trace()
                    cam_points = cam_points[valid_mask]
                    depth = depth[valid_mask]

                image = cam.images[frame_idx]
                vis_img = visualize_depth_points_on_image(image, cam_points, depth)
                print(vis_img.shape)
                cv2.imwrite(os.path.join(projected_depth_vis_dir, f"{frame_idx+start_timestep:03d}_{cam.cam_id}.png"), vis_img)
                depth_dict = {
                    "depth_pixel_coords": cam_points.cpu().numpy() * 1.0 / np.array([cam.WIDTH, cam.HEIGHT]),
                    "depth_values": depth.cpu().numpy(),
                }                   
                np.save(os.path.join(projected_depth_dir, f"{frame_idx+start_timestep:03d}_{cam.cam_id}.npy"), depth_dict)
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_file', type=str, required=True, help='File containing scene indices')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of Waymo dataset')
    parser.add_argument('--num_cameras', type=int, default=5, help='Number of cameras')
    parser.add_argument('--data_config_path', type=str, required=True, help='Path to the data config file')
    args = parser.parse_args()
    
    main(args)
