import time
import tqdm

from config import Config
from utils.utils import get_file_list
from foundation.indoor_object import IndoorObjectDetector
from foundation.indoor_depth import IndoorDepthEstimator
from foundation.indoor_sam import IndoorSAMEstimator
from foundation.indoor_distance import IndoorDistanceEstimator

if __name__ == "__main__":
    start_time = time.time()
    config = Config.from_args()
    image_paths = get_file_list(config.input_image)
    
    # Initialize detectors
    object_detector = IndoorObjectDetector(config)
    depth_estimator = IndoorDepthEstimator(config)
    sam_estimator = IndoorSAMEstimator(config)
    distance_estimator = IndoorDistanceEstimator(config)

    for image_path in tqdm.tqdm(image_paths):
        boxes, scores, pred_phrases = object_detector.process_image(image_path)
        depth_map_norm, metric_depth = depth_estimator.process_image(image_path)
        masks, selected_idx = sam_estimator.process_image(image_path, boxes)

        boxes = [boxes[i] for i in selected_idx]
        pred_phrases = [pred_phrases[i] for i in selected_idx]
        masks = [masks[i] for i in selected_idx]
        relative_positions, point_clouds, colors, sizes = distance_estimator.process_image(image_path, masks, pred_phrases, metric_depth)       

        



    print(f"Total time cost: {time.time() - start_time:.2f}s")
    print(f"Total files processed: {len(image_paths)}")
