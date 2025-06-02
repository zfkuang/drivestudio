import torch.nn.functional as F
import torch
import torch.nn as nn
from promptda.dpt import DPTHead
from promptda.prompt_model import GeneralPromptModel
from promptda.utils.warp_utils import WarpMedian
from safetensors.torch import load_file
import os

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'layer_idxs': [2, 5, 8, 11]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'layer_idxs': [2, 5, 8, 11]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'layer_idxs': [4, 11, 17, 23]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536], 'layer_idxs': [9, 19, 29, 39]}
}


class PromptDepthAnythingPlus(nn.Module):
    encoder = "vitl"
    patch_size = 14
    use_bn = False
    use_clstoken = False

    def __init__(self, 
                 geometry_type: str = "disparity",
                 model_path: str = "model.safetensors"):
        super().__init__()
        model_config = model_configs[self.encoder]
        self.model_config = model_config

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        self.pretrained = torch.hub.load(
            os.path.join(current_file_path, "../torchhub/facebookresearch_dinov2_main"),
            f"dinov2_{self.encoder}14",
            source="local",
            pretrained=False
        )
        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.prompt_model = GeneralPromptModel()
        self.depth_head = DPTHead(
            model_config=model_config,  # using model_config to get the prompt_high_res
            nclass=1,
            in_channels=dim,
            features=model_config["features"],
            out_channels=model_config["out_channels"],
            use_bn=self.use_bn,
            use_clstoken=self.use_clstoken,
            output_act="elu",
            prompt_stage=None,
            high_res_prompt=False,
        )
        self.warp_func = WarpMedian()
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.geometry_type = geometry_type

        if model_path is not None:
            if os.path.exists(model_path):
                self.load_state_dict(load_file(model_path), strict=True)
            else:
                raise FileNotFoundError(f"Model file {model_path} not found")

        # only for inference   
        self.cuda()
        self.eval()
        
    def inference(self, image, prompt_depth, prompt_mask=None):
        if prompt_mask is None:
            prompt_mask = prompt_depth > 0
            
        prompt_depth, prompt_mask, reference_meta = self.warp_func.warp(
            prompt_depth,
            prompt_depth=prompt_depth,
            prompt_mask=prompt_mask,
        )
        depth = self.forward(
            image,
            prompt_depth=prompt_depth,
            prompt_mask=prompt_mask,
        )
        depth = self.warp_func.unwarp(
            depth,
            reference_meta=reference_meta,
            prompt_depth=prompt_depth,
            prompt_mask=prompt_mask,
        )
        if self.geometry_type == "disparity":
            disparity = depth
            depth = 1.0 / torch.clamp(depth, min=5e-3)
            return {"depth": depth, "disparity": disparity}
        elif self.geometry_type == "depth":
            disparity = 1.0 / torch.clamp(depth, min=5e-3)
            return {"depth": depth, "disparity": disparity}

    def forward(self, x, prompt_depth, prompt_mask):
        x = (x - self._mean) / self._std
        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(
            x,
            self.model_config["layer_idxs"],
            return_class_token=True,
        )
        features = [list(feature) for feature in features]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        features = self.prompt_model(features, prompt_depth, prompt_mask, patch_h, patch_w)
        depth = self.depth_head(features, patch_h, patch_w, prompt_depth, prompt_mask)
        return depth
