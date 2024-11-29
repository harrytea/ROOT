import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import tqdm
import os.path as osp
import json

from config import Config
from utils.util import get_file_list
from foundation.indoor_object import IndoorObjectDetector
from foundation.indoor_depth import IndoorDepthEstimator
from foundation.indoor_sam import IndoorSAMEstimator
from foundation.indoor_distance import IndoorDistanceEstimator
from api.qwen2vl_sft import Qwen2VLHandler
from api.qwen25_sft import Qwen25Handler
from utils.show_point import show_point
from prompt.ssg_prompt import str1, str2

import pdb

# Simplified color codes - using only two colors for better consistency
BLUE = '\033[94m'    # For main status messages
GREEN = '\033[92m'   # For step indicators
ENDC = '\033[0m'     # Reset color

if __name__ == "__main__":
    start_time = time.time()
    config = Config.from_args()
    print(f"{BLUE}Starting process - Loading configuration...{ENDC}")
    image_paths = get_file_list(config.input_image)
    
    # Initialize detectors
    print(f"{BLUE}Initializing Object Detector...{ENDC}")
    object_detector = IndoorObjectDetector(config)
    
    print(f"{BLUE}Initializing Depth Estimator...{ENDC}")
    depth_estimator = IndoorDepthEstimator(config)
    
    print(f"{BLUE}Initializing SAM Estimator...{ENDC}")
    sam_estimator = IndoorSAMEstimator(config)
    
    print(f"{BLUE}Initializing Distance Estimator...{ENDC}")
    distance_estimator = IndoorDistanceEstimator(config)
    
    print(f"{BLUE}Initializing Vision Language Model...{ENDC}")
    my_vlm = Qwen2VLHandler()
    my_vlm.initialize_llm(checkpoint=config.qwen_checkpoint)
    
    print(f"{BLUE}Initializing Phrase Simplifier...{ENDC}")
    phrase_simplify = Qwen25Handler()
    phrase_simplify.initialize_llm(checkpoint=config.phrase_simplify_checkpoint)

    for image_path in tqdm.tqdm(image_paths):
        print(f"\n{BLUE}Processing image: {osp.basename(image_path)}{ENDC}")
        print(f"{GREEN}Step 1: Detecting objects...{ENDC}")
        boxes, scores, pred_phrases = object_detector.process_image(image_path)
        
        print(f"{GREEN}Step 2: Estimating depth...{ENDC}")
        depth_map_norm, metric_depth = depth_estimator.process_image(image_path)
        
        print(f"{GREEN}Step 3: Running SAM segmentation...{ENDC}")
        masks, selected_idx = sam_estimator.process_image(image_path, boxes)

        boxes = [boxes[i] for i in selected_idx]
        pred_phrases = [pred_phrases[i] for i in selected_idx]
        masks = [masks[i] for i in selected_idx]
        relative_positions, point_clouds, colors, sizes = distance_estimator.process_image(image_path, masks, pred_phrases, metric_depth)       

        # Get the output directory and file path
        output_dir = osp.join(config.output_dir, osp.basename(osp.dirname(image_path)), osp.basename(image_path).split(".")[0])
        new_pred_phrases_path = osp.join(output_dir, "new_pred_phrases.json")
        
        # Check if processed phrases already exist
        if osp.exists(new_pred_phrases_path):
            with open(new_pred_phrases_path, "r") as f:
                new_pred_phrases = json.load(f)["predictions"]
        else:
            # Process phrases only if needed
            new_pred_phrases = []
            with open("prompt/phrase_simplify.txt", "r", encoding="utf-8") as f:
                phrase_simplify_prompt = f.read()
            for phrase in pred_phrases:
                query = phrase_simplify_prompt.replace("[Insert the phrase here]", phrase)
                simplified_phrase = phrase_simplify.run_llm(query)
                print(f"Original: {phrase} -> Simplified: {json.loads(simplified_phrase)['simplified_phrase']}")
                new_pred_phrases.append(json.loads(simplified_phrase)["simplified_phrase"])
                
            # Save the processed results
            with open(new_pred_phrases_path, "w") as f:
                json.dump({"predictions": new_pred_phrases}, f)

        visualization_path = show_point(image_path, masks, new_pred_phrases, output_dir=output_dir)
        if visualization_path:
            print(f"Visualization saved to: {visualization_path}")
            
        # Construct query using pred_phrases
        object_list = new_pred_phrases + ['wall', 'ceiling', 'floor']
        query = str1 + ", ".join(object_list) + str2
        answer = my_vlm.run_llm(query, visualization_path)
        print(answer)



    print(f"Total time cost: {time.time() - start_time:.2f}s")
    print(f"Total files processed: {len(image_paths)}")
