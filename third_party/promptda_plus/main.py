import torch
import tyro
from safetensors.torch import load_file

from promptda.model import PromptDepthAnythingPlus
from promptda.utils.io_utils import load_image, load_depth, plot_depth

def main(
        input_image_path: str = 'data/examples/rs_color.jpg',
        input_depth_path: str = 'data/examples/rs_depth.png',
        model_path: str = 'model.safetensors',
        output_path: str = 'results/depth_results.png',
        geometry_type: str = 'disparity',
    ) -> None:
    model = PromptDepthAnythingPlus(geometry_type=geometry_type, model_path=model_path)

    image, prompt_depth = load_image(input_image_path), load_depth(input_depth_path)
    image, prompt_depth = image.cuda(), prompt_depth.cuda()
    if geometry_type == "disparity":
        prompt_depth[prompt_depth>0] = 1. / prompt_depth[prompt_depth>0]
    with torch.no_grad():
        with torch.autocast(dtype=torch.bfloat16, device_type='cuda'):
            outputs = model.inference(image=image, prompt_depth=prompt_depth)

    depth = outputs[f"{geometry_type}"]
    plot_depth(image, depth, prompt_depth, output_path)
    import pdb
    pdb.set_trace()
    
    
if __name__ == "__main__":
    tyro.cli(main)

