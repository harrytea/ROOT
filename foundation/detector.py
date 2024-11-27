import torch
import torchvision
from utils.utils import calculate_iou, convert_boxes, draw_bounding_boxes, save_json, vlm_inference
import copy
import json
import time
import os
import json
import os.path as osp
from PIL import Image

class TextProcessor:
    """Class for processing text and labels"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def preprocess_caption(self, caption):
        caption = caption.lower().strip()
        phrases = caption.split(".")
        caption = caption if caption.endswith(".") else caption + "."
        return caption, phrases

    def process_caption_tokens(self, caption):
        tokenized = self.tokenizer(caption)
        special_ids = [
            self.tokenizer.encode(",", add_special_tokens=False)[0],
            self.tokenizer.encode(".", add_special_tokens=False)[0]
        ]
        
        caption_record = []
        short_record = []
        for i, ids in enumerate(tokenized["input_ids"]):
            if ids == self.tokenizer.bos_token:
                continue
            if ids in special_ids:
                caption_record.append(copy.deepcopy(short_record))
                short_record = []
            else:
                short_record.append(i)
        return caption_record


class BoxSelector:
    """Class for handling bounding box selection and filtering"""
    def __init__(self, config, vlm=None):
        self.config = config
        self.vlm = vlm

    def select_boxes(self, logits, boxes, phrase, iou, position, image_path, 
                    final_boxes_list, use_max=True, use_gpt4=True):
        converted_boxes = convert_boxes(boxes.clone(), image_path)
        scores = logits.max(dim=1).values if use_max else logits.sum(dim=1)

        nms_idx = sorted(torchvision.ops.nms(converted_boxes, scores, iou).numpy().tolist())
        boxes = boxes.clone()[nms_idx, :]
        logits = logits.clone()[nms_idx, :]
        converted_boxes = converted_boxes.clone()[nms_idx, :]

        if len(final_boxes_list) >= 1:
            cleaned_boxes_index = []
            for i in range(converted_boxes.shape[0]):
                if calculate_iou(box=converted_boxes[i], boxes=convert_boxes(copy.deepcopy(final_boxes_list), image_path), threshold=iou):
                    cleaned_boxes_index.append(i)
            boxes = boxes[cleaned_boxes_index, :]
            logits = logits[cleaned_boxes_index, :]

        if boxes.nelement() == 0:
            return 0, 0

        if boxes.shape[0] == 1:
            score = round(logits.max(dim=1).values.squeeze().item(), 3)
            return boxes.squeeze(), score

        scores = scores[nms_idx].tolist()
        sorted_scores_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        max_index, second_max_index = sorted_scores_indexes[:2]
        
        try:    
            if scores[max_index] - scores[second_max_index] > self.config.min_threshold:
                return boxes[max_index].squeeze(), round(scores[max_index], 3)
        except:
            return 0, 0

        if use_gpt4 or position is None:
            return self.select_box_based_on_gpt4v(logits=logits, boxes=boxes, phrase=phrase, image_path=image_path)
        return self.select_box_based_on_value(logits=logits, boxes=boxes, position=position)

    def select_box_based_on_gpt4v(self, logits, boxes, phrase, image_path):
        """Select bounding box based on GPT-4V"""
        full_color_list = ["black", "white", "red", "green", "blue", "yellow", "magenta", "cyan"]
        boxes_list = boxes.clone().tolist()
        selected_color_list = str(full_color_list[:len(boxes_list)])
        
        output_subdir = osp.join(self.config.output_dir, osp.basename(osp.dirname(image_path)), osp.basename(image_path).split(".")[0])
        os.makedirs(output_subdir, exist_ok=True)
        new_image_path = draw_bounding_boxes(image_path, boxes_list, output_subdir, f"{osp.basename(output_subdir)}_{phrase}")

        prompt_path = "./prompt/selecting_box_prompt.txt"
        prompt = open(prompt_path).read()
        prompt = prompt.format(description=phrase, count=len(boxes_list), colors=selected_color_list)

        sys_message = "You are an assistant who perfectly judges images."
        try:
            response = vlm_inference(self.vlm.run_llm, prompt, new_image_path, sys_message)
            save_json(f"{output_subdir}/{phrase}_selecting_box_response.json", response)
            response = json.loads(response)
            color_idx = full_color_list.index(response["color"])
            
            if color_idx < min(len(boxes_list), 8):
                score = round(torch.max(logits[color_idx]).item(), 3)
                return boxes[color_idx].squeeze(), score
        except Exception as e:
            print(f"Error: {str(e)}")
            return 0, 0

    def select_box_based_on_value(self, logits, boxes, position):
        """Select bounding box based on position value"""
        key_function = lambda x: abs(x[0] - position[0]) + abs(x[1] - position[1])
        key_values = torch.tensor([key_function(b) for b in boxes])
        best_box_ids = torch.argmin(key_values).item()
        score = round(torch.max(logits[best_box_ids]).item(), 3)
        return boxes[best_box_ids].squeeze(), score


class BoxDetector:
    """Main class for handling bounding box detection"""
    def __init__(self, model, processor, vlm,   config):
        self.model = model
        self.processor = processor
        self.vlm = vlm
        self.config = config
        self.text_processor = TextProcessor(processor.tokenizer)
        self.box_selector = BoxSelector(config, vlm)

    def detect(self, tags, image_path, positions=None, existing_boxes=None):
        """
        Perform detection and return results
        Args:
            tags: Text labels to detect
            image_path: Path to the image
            positions: Optional list of position information
            existing_boxes: List of existing bounding boxes
        Returns:
            tuple: (list of bounding boxes, list of confidence scores, list of predicted phrases)
        """
        existing_boxes = existing_boxes or []
        
        # run grounding dino model
        image_raw = Image.open(image_path).convert("RGB")
        tags, phrases = self.text_processor.preprocess_caption(tags)
        logits_filt, boxes_filt = self._run_grounding_dino_inference(image_raw, tags)
        
        # Process tokens
        tags_record = self.text_processor.process_caption_tokens(tags)
        if positions and len(tags_record) != len(positions):
            raise ValueError("Mismatch between tags records and positions")
        

        final_boxes, scores, pred_phrases = self._process_phrases_and_get_results(
            phrases, tags_record, positions, logits_filt, boxes_filt, 
            image_path, existing_boxes
        )

        return convert_boxes(final_boxes, image_path), scores, pred_phrases, final_boxes

    def _run_grounding_dino_inference(self, image_raw, caption):
        """Run model inference"""
        inputs = self.processor(images=image_raw, text=caption, return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits, boxes = outputs.logits[0].cpu().sigmoid(), outputs.pred_boxes[0].cpu() # [900, 256], [900, 4]
        mask = logits.max(dim=1)[0] > self.config.box_threshold
        return logits[mask], boxes[mask]

    def _process_phrases_and_get_results(self, phrases, tags_record, positions, logits_filt, boxes_filt, image_path, existing_boxes):
        """Process detection results for each phrase"""
        final_boxes, scores, pred_phrases = [], [], []
        
        for i, (record, phrase) in enumerate(zip(tags_record, phrases)):
            if not phrase.strip():
                continue

            logits_selected = logits_filt.clone().index_select(1, torch.tensor(record))
            boxes_selected = boxes_filt.clone()
            selected_mask = logits_selected.max(dim=1)[0] > self.config.box_threshold

            logits_selected = logits_selected[selected_mask]
            boxes_selected = boxes_selected[selected_mask]

            if logits_selected.nelement() == 0 or boxes_selected.nelement() == 0:
                continue

            checking_list = copy.deepcopy(final_boxes)
            if existing_boxes:
                checking_list.extend(existing_boxes)

            boxes_selected, score = self.box_selector.select_boxes(
                logits_selected, 
                boxes_selected, 
                phrase, 
                self.config.iou_threshold,
                positions[i] if positions else None,
                image_path, 
                checking_list
            )

            if isinstance(boxes_selected, int) and score == 0:
                print(f"- Failed to select valid box for: '{phrase}'\n{'='*50}")
                continue
                
            print(f"- Detected box: {[round(x, 3) for x in boxes_selected.tolist()]}, score: {score:.3f}\n{'='*50}")
            final_boxes.append(boxes_selected)
            scores.append(score)
            pred_phrases.append(phrase)

        return final_boxes, scores, pred_phrases
