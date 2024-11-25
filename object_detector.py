import os
import os.path as osp
import time
import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import json

from config import Config
from utils.utils import (
    get_file_list, 
    visualize_detections, 
    save_json,
    expand_box,
    calculate_iou,
    vlm_inference
)
import copy
from foundation.detector import BoxDetector


class IndoorObjectDetector:
    def __init__(self, config: Config):
        self.config = config
        self.vlm = self._init_vlm()
        self.box_detector = self._init_grounded_dino()

    def _init_vlm(self):
        print(f"Loading VLM model: {self.config.vlm_model}")
        if self.config.vlm_model == "gpt-4v":
            from api.gpt4v import GPT4VHandler
            vlm = GPT4VHandler()
            vlm.initialize_llm()
            return vlm
        return None

    def _init_grounded_dino(self):
        print(f"Loading model from {self.config.grounded_checkpoint}")
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.config.grounded_checkpoint).to(self.config.device)
        processor = AutoProcessor.from_pretrained(self.config.grounded_checkpoint)
        return BoxDetector(model, processor, self.vlm, self.config)

    def _prepare_output_dir(self, image_path):
        output_dir = osp.join(self.config.output_dir, osp.basename(osp.dirname(image_path)), osp.basename(image_path).split(".")[0])
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _load_container_caption(self, image_path, query="", sys_message=""):
        prompt_path = "prompt/caption_prompt_container.txt"
        prompt = open(prompt_path).read()
        query = prompt
        sys_message = sys_message if sys_message != "" else "You are an assistant who perfectly describes images."
        response = vlm_inference(self.vlm.run_llm, query, image_path, sys_message)
        return json.loads(response)

    def _sub_object_captioning(self, data_path, container, sys_message=""):
        prompt_path = "prompt/caption_prompt_sub_object.txt"
        prompt = open(prompt_path).read()
        query = prompt.replace("{container}", container)
        sys_message = sys_message if sys_message != "" else "You are an assistant who perfectly describes images."
        return vlm_inference(self.vlm.run_llm, query, data_path, sys_message)


    def process_image(self, image_path):
        output_dir = self._prepare_output_dir(image_path)
        if osp.exists(osp.join(output_dir, "final_result.json")): return
        
        # Load container caption
        if osp.exists(osp.join(output_dir, "container_caption.json")):
            container_caption = json.load(open(osp.join(output_dir, "container_caption.json")))
        else:
            container_caption = self._load_container_caption(image_path)
            save_json(osp.join(output_dir, "container_caption.json"), container_caption)

        # Process detections
        boxes, scores, pred_phrases = self._process_detections(container_caption, image_path, output_dir)
        results = {"boxes": [box.tolist() for box in boxes], "scores": scores, "predictions": pred_phrases}
        save_json(f"{output_dir}/final_result.json", results)
        visualize_detections(image_path, boxes, pred_phrases, output_dir, 'final_result')


    def _process_detections(self, caption, image_path, output_dir):
        text_descriptions = [details['description'].replace(".", "").lower() for details in caption.values() if 'description' in details]
        container_attribute = [details['container'].replace(".", "") == "True" for details in caption.values() if 'container' in details]
        tags = ".".join(text_descriptions)

        # first iteration detection
        boxes, scores, pred_phrases, unconverted_boxes = self.box_detector.detect(tags=tags, image_path=image_path)

        visualize_detections(image_path, boxes, pred_phrases, output_dir, 'middle_result')
        remain_index = [i for i, elem in enumerate(text_descriptions) if elem in pred_phrases]
        container_attribute = [container_attribute[i] for i in remain_index]
        copy_boxes, copy_phrase = copy.deepcopy(boxes), copy.deepcopy(pred_phrases)


        print("="*50 + "\nProcessing sub-boxes\n" + "="*50)
        image_raw = Image.open(image_path).convert("RGB")  # load image
        for b, phrase, contain_item in zip(copy_boxes, copy_phrase, container_attribute):
            if not contain_item: 
                continue

            print(f"Processing container: {phrase}")
            b = expand_box(image_raw.size[1], image_raw.size[0], [i.item() for i in b])
            container_image = image_raw.crop(b)
            container_path = f"{output_dir}/temp_container_{phrase}.jpg"
            container_image.save(container_path)
            
            # Get sub-object captions
            sub_res = self._sub_object_captioning(container_path, container=phrase)
            save_json(f"{output_dir}/sub_object_{phrase}.json", sub_res)
            
            if "-1" in sub_res: 
                continue
            
            try:
                sub_res = json.loads(sub_res)
            except json.JSONDecodeError:
                continue
            
            # Process sub-objects
            sub_objects = [details['description'].replace(".", "") for details in sub_res.values() if 'description' in details]
            sub_tags = ".".join(sub_objects)
            sub_boxes, sub_scores, sub_phrases, _ = self.box_detector.detect(
                tags=sub_tags, image_path=container_path,
                existing_boxes=copy.deepcopy(unconverted_boxes)
            )

            # Add valid sub-objects to results
            added_count = 0
            for sub_b, sub_s, sub_p in zip(sub_boxes, sub_scores, sub_phrases):
                sub_b[[0,2]] += b[0]
                sub_b[[1,3]] += b[1]
                if calculate_iou(sub_b, boxes, threshold=self.config.iou_threshold):
                    boxes.append(sub_b)
                    scores.append(sub_s)
                    pred_phrases.append(sub_p)
                    added_count += 1
            print(f"Added {added_count} sub-objects")
                    
        # Save final results
        save_json(
            f"{output_dir}/final_caption.json", 
            {pred_phrase: {"box": box.tolist(), "score": float(score), "phrase": pred_phrase} 
             for box, score, pred_phrase in zip(boxes, scores, pred_phrases)}
        )
        visualize_detections(image_path, boxes, pred_phrases, output_dir, flag='final_result')
        return boxes, scores, pred_phrases


if __name__ == "__main__":
    start_time = time.time()
    config = Config.from_args()
    detector = IndoorObjectDetector(config)
    
    image_paths = get_file_list(config.input_image)
    for image_path in tqdm.tqdm(image_paths):
        detector.process_image(image_path)
    print(f"Total time cost: {time.time() - start_time:.2f}s")
    print(f"Total files processed: {len(image_paths)}")