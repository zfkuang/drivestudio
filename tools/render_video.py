from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views
)

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True
):
    trainer.set_eval()

    logger.info("Evaluating Pixels...")
        
    if cfg.render.render_full:
        logger.info("Evaluating Full Set...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
            compute_metrics=False,
            compute_error_map=cfg.render.vis_error,
            force_downscale=2
        )
        
        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_img_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=cfg.render.fps,
            verbose=True,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/full/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
            
def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False
    for folder in ["videos_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device
    )
    
    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False
    
    # do_evaluation(
    #     step=trainer.step,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    #     post_fix="_eval"
    # )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str, required=True)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    parser.add_argument("--save_catted_videos", type=bool, default=False, help="visualize lidar on image")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
        
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    main(args)