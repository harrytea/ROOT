import torch
import numpy as np
from PIL import Image
import os.path as osp
import os
from foundation.Depth_Anything.metric_depth.zoedepth.models.builder import build_model
from foundation.Depth_Anything.metric_depth.zoedepth.utils.config import get_config
import torchvision.transforms as transforms
import cv2


class IndoorDepthEstimator:
    def __init__(self, config):
        self.config = config
        self.model = self._init_depth_model()

    def _init_depth_model(self):
        print("Loading depth estimation model...")
        config = get_config('zoedepth', "eval", 'nyu')
        config.pretrained_resource = self.config.depth_checkpoint
        model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        return model

    def _prepare_output_dir(self, image_path):
        output_dir = osp.join(
            self.config.output_dir, 
            osp.basename(osp.dirname(image_path)), 
            osp.basename(image_path).split(".")[0]
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def process_image(self, image_path):
        output_dir = self._prepare_output_dir(image_path)
        depth_output_path = osp.join(output_dir, "depth_map.png")
        depth_range_path = osp.join(output_dir, "depth_range.npy")
        depth_original_path = osp.join(output_dir, "depth_original.npy")

        # Check cache if enabled
        if self.config.use_cache and osp.exists(depth_output_path) and osp.exists(depth_original_path):
            depth_map = cv2.imread(depth_output_path, cv2.IMREAD_UNCHANGED)
            depth_original = np.load(depth_original_path)
            return depth_map, depth_original

        # If no cache or cache disabled, process the depth map
        image = Image.open(image_path)
        depth_map = self.predict_depth(image)

        # Process depth map
        depth_original = depth_map.copy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 0:
            depth_map = ((depth_map - depth_min) * (255 / (depth_max - depth_min))).astype(np.uint8)
        else:
            depth_map = np.full_like(depth_map, 128, dtype=np.uint8)

        # Save depth map and original depth information
        depth_image = Image.fromarray(depth_map)
        depth_image.save(depth_output_path)
        np.save(depth_original_path, depth_original)
        
        # Save depth range information
        depth_range = {'min': float(depth_min), 'max': float(depth_max)}
        np.save(depth_range_path, depth_range)
        
        return depth_map, depth_original

    def predict_depth(self, color_image):
        original_width, original_height = color_image.size
        image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        pred = self.model(image_tensor, dataset='nyu')
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]

        pred = pred.squeeze().detach().cpu().numpy()

        resized_pred = Image.fromarray(pred).resize((original_width, original_height), Image.NEAREST)
        return np.array(resized_pred)

if __name__ == "__main__":
    print()