import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import json
import random
import os.path as osp
import pdb

import tqdm
from config import Config
from utils.util import (
    get_file_list, extract_json_from_string,
    BLUE, GREEN, ENDC, YELLOW, CYAN, MAGENTA, BOLD, RED
)
from utils.show_point import show_point
from utils.show_relations import show_relations, parse_json_nodes

# Foundation model imports
from foundation.indoor_object import IndoorObjectDetector
from foundation.indoor_depth import IndoorDepthEstimator
from foundation.indoor_sam import IndoorSAMEstimator
from foundation.indoor_distance import IndoorDistanceEstimator

from api.qwen2vl_sft import Qwen2VLHandler
from api.qwen25_sft import Qwen25Handler
from prompt.ssg_prompt import str1, str2



if __name__ == "__main__":
    print(f"{BOLD}{CYAN}Starting Scene Processing Pipeline{ENDC}")
    config = Config.from_args()
    image_paths = get_file_list(config.input_image)
    print(f"{BOLD}{CYAN}Found {len(image_paths)} images to process{ENDC}")
    
    print(f"{BOLD}{MAGENTA}Initializing Foundation Models:{ENDC}")
    object_detector = IndoorObjectDetector(config)
    depth_estimator = IndoorDepthEstimator(config)
    sam_estimator = IndoorSAMEstimator(config)
    distance_estimator = IndoorDistanceEstimator(config)

    print(f"{BOLD}{MAGENTA}Initializing SceneVLM Model:{ENDC}")
    my_vlm = Qwen2VLHandler()
    my_vlm.initialize_llm(checkpoint=config.qwen_checkpoint)


    start_time = time.time()
    for image_path in tqdm.tqdm(image_paths, desc="Processing Images"):
        print(f"\n{BOLD}{BLUE}[Image Processing]{' '*4}{image_path}{ENDC}")
        
        print(f"{CYAN}Step 1: Detecting Objects...{ENDC}")
        results = object_detector.process_image(image_path)
        boxes, scores, pred_phrases = results["boxes"], results["scores"], results["pred_phrases"]

        print(f"{CYAN}Step 2: Estimating Depth...{ENDC}")
        depth_map, depth_original = depth_estimator.process_image(image_path)

        print(f"{CYAN}Step 3: Generating Masks...{ENDC}")
        all_masks, selected_idx = sam_estimator.process_image(image_path, boxes)
        masks = [all_masks[i] for i in selected_idx]
        pred_phrases = [pred_phrases[i] for i in selected_idx]

        print(f"{CYAN}Step 4: Estimating Distances...{ENDC}")
        relative_positions, point_clouds, colors, sizes = distance_estimator.process_image(image_path, masks, pred_phrases, depth_original)       

        # Get the output directory and file path
        output_dir = osp.join(config.output_dir, osp.basename(osp.dirname(image_path)), osp.basename(image_path).split(".")[0])
        relative_positions_path = osp.join(output_dir, "relative_positions.json")
        with open(relative_positions_path, "w") as f:
            json.dump(relative_positions, f, indent=4)

        # Randomly select and print 5 distances
        sample_size = min(5, len(relative_positions))
        if sample_size > 0:
            for pos in random.sample(relative_positions, sample_size):
                obj1, obj2 = pos['object_pair']
                print(f"{YELLOW}Distance between {obj1} and {obj2}: {pos['distance']:.2f} meters{ENDC}")
        
        new_pred_phrases = None
        new_pred_phrases_path = osp.join(output_dir, "new_pred_phrases.json")
        if osp.exists(new_pred_phrases_path):
            with open(new_pred_phrases_path, "r") as f:
                data = json.load(f)
                if "predictions" in data and len(data["predictions"]) == len(pred_phrases):
                    new_pred_phrases = data["predictions"]
                    print("Using cached simplified phrases.")

        if new_pred_phrases is None:
            phrase_simplify = Qwen25Handler()
            phrase_simplify.initialize_llm(checkpoint=config.phrase_simplify_checkpoint)
            with open("prompt/phrase_simplify.txt", "r", encoding="utf-8") as f:
                phrase_simplify_prompt = f.read()
            
            # Step 1: Simplify all phrases in parallel
            def simplify_single_phrase(phrase):
                query = phrase_simplify_prompt.replace("[Insert the phrase here]", phrase)
                simplified_phrase = phrase_simplify.run_llm(query)
                return json.loads(simplified_phrase)["simplified_phrase"]
            
            simplified_results = [simplify_single_phrase(phrase) for phrase in pred_phrases]
            
            # Step 2: Add numbering for duplicate phrases
            phrase_count = {}
            new_pred_phrases = []
            for phrase in simplified_results:
                phrase_count[phrase] = phrase_count.get(phrase, 0) + 1
                new_phrase = f"{phrase}_{phrase_count[phrase]}" if phrase_count[phrase] > 1 else phrase
                new_pred_phrases.append(new_phrase)
            with open(new_pred_phrases_path, "w") as f:
                json.dump({"predictions": new_pred_phrases}, f, indent=4)
            print(f"{CYAN}Simplified {len(new_pred_phrases)} phrases{ENDC}")

        # Print comparison results
        print("\nPhrase Simplification Results:")
        for orig, simp in zip(pred_phrases, new_pred_phrases):
            print(f"{BLUE}Original:{ENDC} {orig} -> {GREEN}Simplified:{ENDC} {simp}")


        print(f"{GREEN}Hierarchical SceneGraph Generation...{ENDC}")
        try:
            visualization_path = show_point(image_path, masks, new_pred_phrases, output_dir=output_dir)
            print(f"{CYAN}Visualization path: {visualization_path}{ENDC}")
            object_list = new_pred_phrases + ['wall', 'ceiling', 'floor']
            query = str1 + ", ".join(object_list) + str2
            answer = my_vlm.run_llm(query, visualization_path)
            print(f"{CYAN}Answer: {answer}{ENDC}")
            with open(osp.join(output_dir, "answer.json"), "w") as f:
                json.dump(extract_json_from_string(answer), f, indent=4)
            print(f"{GREEN}Visualizing relations...{ENDC}")
            relations = extract_json_from_string(answer)
            relations = parse_json_nodes(relations)
            output_path =show_relations(image_path, masks, new_pred_phrases, relations, output_dir=output_dir)
        except Exception as e:
            print(f"{RED}Error extracting JSON from answer: {e}{ENDC}") 
            print(answer)


    print(f"\n{BOLD}{BLUE}{'='*50}\nProcessing Complete! Time cost: {time.time() - start_time:.2f}s Files processed: {len(image_paths)}\n{'='*50}{ENDC}")
