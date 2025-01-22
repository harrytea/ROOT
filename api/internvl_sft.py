import os
import sys

sys.path.append(os.getcwd())
sys.path.append("..")
sys.path.append("/llm-cfs-nj/person/harryyhwang/InternVL")
sys.path.append("/llm-cfs-nj/person/harryyhwang/InternVL/internvl_chat")


import torch
from PIL import Image
from transformers import AutoTokenizer
import math 

import torch
from internvl_chat.internvl.model.internvl_chat import InternVLChatModel
from internvl_chat.internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from transformers import AutoTokenizer


class InternVLHandler:
    def __init__(self):
        pass
    
    def initialize_llm(self, dynamic=True, load_in_8bit=False, load_in_4bit=False, auto=True,
                       num_beams=5, top_k=50, top_p=0.9, sample=False, max_num=6, model_name="InternVL2-8B",
                       checkpoint="/llm-cfs-nj/person/harryyhwang/InternVL/internvl_chat/work_dirs/internvl_chat_v2_0/internvl2_8b_distance"):

        # model_name: 'InternVL2-1B', 'InternVL2-2B', 'InternVL2-4B', 'InternVL2-8B','InternVL2-26B', 'InternVL2-40B', 'InternVL2-Llama3-76B'
        kwargs = {"device_map": split_model(model_name)}

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
        self.model = InternVLChatModel.from_pretrained(checkpoint, 
            low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **kwargs).eval()
        if not load_in_8bit and not load_in_4bit and not auto:
            self.model = self.model.cuda()
        image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        use_thumbnail = self.model.config.use_thumbnail

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        if total_params > 20 or dynamic:
            num_beams = 1
            print(f'[test] total_params: {total_params}B, use num_beams: {num_beams}')
        else:
            print(f'[test] total_params: {total_params}B')
        print(f'[test] image_size: {image_size}')
        print(f'[test] template: {self.model.config.template}')
        print(f'[test] dynamic_image_size: {dynamic}')
        print(f'[test] use_thumbnail: {use_thumbnail}')
        print(f'[test] max_num: {max_num}')

        self.generation_config = dict(
            do_sample=sample,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=1024,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.dynamic = dynamic
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.image_size = image_size


    
    def run_llm(self, query, image_path):
        query = "<image>\n" + query
        pixel_values = load_image(image_path, self.dynamic, self.use_thumbnail, self.max_num, self.image_size).cuda().to(torch.bfloat16)
        while True:
            try:
                answer = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=query,
                    generation_config=self.generation_config,
                    verbose=True
                )         
                break
            except:
                print()
        return answer


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80
        }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def load_image(image_file, dynamic, use_thumbnail, max_num, input_size=224):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size)
    if dynamic:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    else:
        images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

if __name__ == '__main__':

    # /llm-cfs-nj/person/harryyhwang/InternVL/pretrained/InternVL2-8B
    my_vlm = InternVLHandler()
    my_vlm.initialize_llm(model_name="InternVL2-8B",
                          checkpoint="/llm-cfs-nj/person/harryyhwang/InternVL/pretrained/InternVL2-8B")

    answer = my_vlm.run_llm("what is this", "/llm-cfs-nj/person/harryyhwang/2D-Scene-data/0001909.jpg")
    print(answer)

    # CUDA_VISIBLE_DEVICES=2,3 python infer_internvl.py  --prompt v2
