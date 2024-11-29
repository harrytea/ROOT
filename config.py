from dataclasses import dataclass
import argparse

@dataclass
class Config:
    device: str = 'cuda'
    vlm_model: str = 'gpt-4v'
    grounded_checkpoint: str = "ckpts/grounding-dino-base"
    depth_checkpoint: str = "local::/llm-cfs-nj/person/harryyhwang/ft_local/ft_local/ROOT/foundation/Depth_Anything/depth_anything_metric_depth_indoor.pt"
    box_threshold: float = 0.3
    iou_threshold: float = 0.5
    min_threshold: float = 0.15
    save_file: bool = False
    mask_filter: bool = False
    output_dir: str = 'outputs2'
    input_image: str = 'asset/0001682.jpg'
    prompt_dir: str = './prompt'
    sam_checkpoint: str = 'ckpts/sam-vit-huge'
    qwen_checkpoint: str = 'ckpts/Qwen2-VL-7B-FULL-full'
    phrase_simplify_checkpoint: str = 'ckpts/Qwen2.5-3B-Instruct'
    use_cache: bool = True

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description='Indoor Scene Pipeline')
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--vlm_model', type=str, default='gpt-4v')
        parser.add_argument('--grounded_checkpoint', type=str, default="ckpts/grounding-dino-base")
        parser.add_argument('--depth_checkpoint', type=str, default="local::/llm-cfs-nj/person/harryyhwang/ft_local/ft_local/ROOT/foundation/Depth_Anything/depth_anything_metric_depth_indoor.pt")
        parser.add_argument('--box_threshold', type=float, default=0.3)
        parser.add_argument('--iou_threshold', type=float, default=0.5)
        parser.add_argument('--min_threshold', type=float, default=0.15)
        parser.add_argument('--save_file', action='store_true')
        parser.add_argument('--mask_filter', action='store_true')
        parser.add_argument('--output_dir', type=str, default='outputs')
        parser.add_argument('--input_image', type=str, default='asset/0001682.jpg')
        parser.add_argument('--prompt_dir', type=str, default='./prompt')
        parser.add_argument('--sam_checkpoint', type=str, default='ckpts/sam-vit-huge')
        parser.add_argument('--qwen_checkpoint', type=str, default='ckpts/Qwen2-VL-7B-FULL-full')
        parser.add_argument('--phrase_simplify_checkpoint', type=str, default='ckpts/Qwen2.5-3B-Instruct')
        parser.add_argument('--use_cache', action='store_true', default=True)
        
        args = parser.parse_args()
        return cls(**vars(args)) 