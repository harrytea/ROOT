import os
import sys

sys.path.append(os.getcwd())
sys.path.append("..")

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

class Qwen2VLHandler:
    def __init__(self):
        pass
    
    def initialize_llm(self, checkpoint):

        # Load the model in half-precision on the available device(s)
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint,torch_dtype="auto",attn_implementation="flash_attention_2", device_map="auto")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint,torch_dtype="auto",attn_implementation="flash_attention_2", device_map="cuda")
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint,torch_dtype="auto",device_map="cpu")
        self.processor = AutoProcessor.from_pretrained(checkpoint)
    
    def run_llm(self, query, image_path):
        query = "<image>\n" + query
        image = Image.open(image_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image",},
                    {"type": "text", "text": f"{query}"},
                ],
            }
        ]
        # Preprocess the inputs
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")


        while True:
            try:
                # Inference: Generation of the output
                output_ids = self.model.generate(**inputs, max_new_tokens=2048)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                answer = answer[0]
                break
            except Exception as e:
                print(f"Error in Qwen2VLHandler.run_llm: {e}")
                
        return answer


if __name__ == '__main__':
    my_vlm = Qwen2VLHandler()
    my_vlm.initialize_llm(checkpoint="/llm-cfs-nj/person/harryyhwang/Qwen2-VL/ckpt/Qwen/Qwen2-VL-7B-Instruct")
    answer = my_vlm.run_llm("what is this", "/llm-cfs-nj/person/harryyhwang/2D-Scene-data/0001909.jpg")
    print(answer)

    # CUDA_VISIBLE_DEVICES=2,3 python infer_internvl.py  --prompt v2

