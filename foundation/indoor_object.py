import os
import os.path as osp
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import json
from utils.util import (
    visualize_detections, 
    save_json,
    expand_box,
    calculate_iou,
    vlm_inference
)
import copy
from foundation.detector import BoxDetector


class IndoorObjectDetector:
    def __init__(self, config):
        self.config = config
        self.vlm = self._init_vlm()
        self.box_detector = self._init_grounded_dino()

    def _init_vlm(self):
        print(f"Loading VLM model: {self.config.vlm_model}")
        if self.config.vlm_model == "gpt-4v":
            from api.gpt4v_azure import GPT4VHandler
            vlm = GPT4VHandler()
            vlm.initialize_llm()
            return vlm
        elif self.config.vlm_model == "gpt-4v-openai":
            from api.gpt4v_openai import GPT4VOpenAIHandler
            vlm = GPT4VOpenAIHandler()
            vlm.initialize_llm()
            return vlm
        return None

    def _init_grounded_dino(self):
        print(f"Loading model from {self.config.grounded_checkpoint}")
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.config.grounded_checkpoint).to(self.config.device)
        processor = AutoProcessor.from_pretrained(self.config.grounded_checkpoint)
        return BoxDetector(model, processor, self.vlm, self.config)


    def process_image(self, image_path, output_dir=None):
        self.output_dir = self._prepare_output_dir(image_path) if output_dir is None else output_dir
        self.cache_path = osp.join(self.output_dir, "final_result.json")
        if self.config.use_cache and osp.exists(self.cache_path):
            return json.load(open(self.cache_path))
        
        container_caption = self._get_container_caption(image_path, self.output_dir)
        all_results = self._process_detections(container_caption, image_path, self.output_dir)
        return all_results

    def _process_detections(self, container_caption, image_path, output_dir):
        initial_detection = self._perform_initial_detection(container_caption, image_path, output_dir)
        container_results = self._process_containers(initial_detection, image_path, output_dir)
        all_results = self._save_final_results(container_results, image_path, output_dir)
        save_json(self.cache_path, all_results)
        return all_results

    def _perform_initial_detection(self, container_caption, image_path, output_dir):
        text_descriptions = [details['description'].replace(".", "").lower() for details in container_caption.values() if 'description' in details]
        container_attribute = [details['container'].replace(".", "") == "True" for details in container_caption.values() if 'container' in details]
        tags = ".".join(text_descriptions)

        converted_boxes, scores, pred_phrases, unconverted_boxes = self.box_detector.detect(tags=tags, image_path=image_path, output_dir=output_dir)
        middle_result_path = visualize_detections(image_path, converted_boxes, pred_phrases, output_dir, 'middle_result')
        
        return {
            'boxes': converted_boxes, 
            'scores': scores,
            'pred_phrases': pred_phrases,   # prob threshold > 0.3
            'unconverted_boxes': unconverted_boxes,  # gpt4v
            'text_descriptions': text_descriptions,
            'container_attribute': container_attribute,
            'middle_result_path': middle_result_path
        }

    def _process_containers(self, initial_detection, image_path, output_dir):
        image_raw = Image.open(image_path).convert("RGB")
        container_meta = {}
        boxes = copy.deepcopy(initial_detection['boxes'])
        scores = copy.deepcopy(initial_detection['scores'])
        pred_phrases = copy.deepcopy(initial_detection['pred_phrases'])

        matched_phrases = [
            i for i, text in enumerate(initial_detection['text_descriptions']) 
            if text in initial_detection['pred_phrases']
        ]
        container_attribute = [initial_detection['container_attribute'][i] for i in matched_phrases]
        for container in self._iterate_containers(
            initial_detection['boxes'], initial_detection['pred_phrases'], container_attribute
        ):
            sub_objects = self._process_single_container(container, image_raw, output_dir, initial_detection['unconverted_boxes'])
            if sub_objects:
                boxes.extend(sub_objects['boxes'])
                scores.extend(sub_objects['scores'])
                pred_phrases.extend(sub_objects['phrases'])
                container_meta[container['phrase']] = {
                    "boxes": sub_objects['boxes'],
                    "scores": sub_objects['scores'],
                    "predictions": sub_objects['phrases'],
                    "container_path": sub_objects['container_path']
                }

        return {
            'boxes': boxes,
            'scores': scores,
            'pred_phrases': pred_phrases,
            'container_meta': container_meta,
            'initial_detection': initial_detection
        }

    def _iterate_containers(self, boxes, phrases, container_flags):
        for b, phrase, is_container in zip(boxes, phrases, container_flags):
            if is_container:
                yield {'box': b, 'phrase': phrase}

    def _process_single_container(self, container, image_raw, output_dir, unconverted_boxes):
        print(f"Processing container: {container['phrase']}")
        box = expand_box(image_raw.size[1], image_raw.size[0], [i.item() for i in container['box']])
        container_image = image_raw.crop(box)
        container_path = f"{output_dir}/temp_container_{container['phrase']}.jpg"
        container_image.save(container_path)
        
        # Get sub-object descriptions
        sub_res = self._get_sub_object_captions(container_path, container['phrase'], output_dir)
        if not sub_res:
            return None

        return self._detect_sub_objects(sub_res, container_path, box, unconverted_boxes, output_dir)

    def _get_sub_object_captions(self, container_path, container_phrase, output_dir):
        sub_res = self._sub_object_captioning(container_path, container=container_phrase)
        save_json(osp.join(output_dir, f"sub_object_{container_phrase}.json"), sub_res)
        
        if "-1" in sub_res:
            return None
            
        try:
            return json.loads(sub_res)
        except json.JSONDecodeError:
            return None

    def _detect_sub_objects(self, sub_res, container_path, container_box, unconverted_boxes, output_dir):
        # Extract sub-object descriptions
        sub_objects = [
            details['description'].replace(".", "") 
            for details in sub_res.values() 
            if 'description' in details
        ]
        sub_tags = ".".join(sub_objects)
        
        # Detect sub-objects
        sub_boxes, sub_scores, sub_phrases, _ = self.box_detector.detect(
            tags=sub_tags,
            image_path=container_path,
            output_dir=output_dir,
            existing_boxes=copy.deepcopy(unconverted_boxes)
        )

        # Adjust sub-object coordinates to the original image coordinate system
        valid_detections = {'boxes': [], 'scores': [], 'phrases': []}
        
        for sub_b, sub_s, sub_p in zip(sub_boxes, sub_scores, sub_phrases):
            sub_b[[0,2]] += container_box[0]
            sub_b[[1,3]] += container_box[1]
            
            if calculate_iou(sub_b, valid_detections['boxes'], threshold=self.config.iou_threshold):
                valid_detections['boxes'].append(sub_b)
                valid_detections['scores'].append(sub_s)
                valid_detections['phrases'].append(sub_p)
        
        valid_detections['container_path'] = container_path
        print(f"Added {len(valid_detections['boxes'])} sub-objects")
        return valid_detections

    def _save_final_results(self, results, image_path, output_dir):
        final_result_path = visualize_detections(
            image_path,
            results['boxes'],
            results['pred_phrases'],
            output_dir,
            flag='final_result'
        )
        
        # Convert container_meta tensors to lists
        processed_container_meta = {}
        for container_name, container_data in results['container_meta'].items():
            processed_container_meta[container_name] = {
                "boxes": [box.tolist() for box in container_data['boxes']],
                "scores": container_data['scores'],
                "predictions": container_data['predictions'],
                "container_path": container_data['container_path']
            }
        
        return {
            "boxes": [box.tolist() for box in results['boxes']],
            "scores": results['scores'],
            "pred_phrases": results['pred_phrases'],
            "container_meta": processed_container_meta,
            "box_middle": [box.tolist() for box in results['initial_detection']['boxes']],
            "score_middle": results['initial_detection']['scores'],
            "pred_middle": results['initial_detection']['pred_phrases'],
            "middle_result_path": results['initial_detection']['middle_result_path'],
            "final_result_path": final_result_path
        }

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
    
    def _get_container_caption(self, image_path, output_dir):
        container_caption_path = osp.join(output_dir, "container_caption.json")
        if self.config.use_cache and osp.exists(container_caption_path):
            return json.load(open(container_caption_path))
        
        container_caption = self._load_container_caption(image_path)
        save_json(container_caption_path, container_caption)
        return container_caption

    def _prepare_output_dir(self, image_path):
        output_dir = osp.join(
            self.config.output_dir,
            osp.basename(osp.dirname(image_path)),
            osp.basename(image_path).split(".")[0]
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir