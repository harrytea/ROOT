import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import cv2
import os.path as osp
import torch
import gradio as gr
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime

from config import Config
from foundation.indoor_object import IndoorObjectDetector
from foundation.indoor_depth import IndoorDepthEstimator
from foundation.indoor_sam import IndoorSAMEstimator
from foundation.indoor_distance import IndoorDistanceEstimator

from api.qwen25_sft import Qwen25Handler
from api.qwen2vl_sft import Qwen2VLHandler
from prompt.ssg_prompt import str1, str2
from utils.util import extract_json_from_string
from utils.show_point import show_point
from utils.show_relations import show_relations, parse_json_nodes

config = Config.from_args()
object_detector = IndoorObjectDetector(config)
depth_estimator = IndoorDepthEstimator(config)
sam_estimator = IndoorSAMEstimator(config)
distance_estimator = IndoorDistanceEstimator(config)

phrase_simplify = Qwen25Handler()
phrase_simplify.initialize_llm(checkpoint=config.phrase_simplify_checkpoint)
my_vlm = Qwen2VLHandler()
my_vlm.initialize_llm(checkpoint=config.qwen_checkpoint)


# ------------------------------------------------------------
# Global Variables & Initialization
# ------------------------------------------------------------
def get_output_dir(image_path=None):
    base_dir = "outputs"
    if image_path:
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(base_dir, img_name)
    return base_dir
output_dir = ""

# ------------------------------------------------------------
# indoor object perception
# ------------------------------------------------------------
def generate_caption(image):
    global output_dir
    # Generate timestamp-based filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"temp_input_{timestamp}.jpg"
    output_dir = get_output_dir(temp_path)
    os.makedirs(output_dir, exist_ok=True)
    
    temp_path = os.path.join(output_dir, temp_path)
    image.save(temp_path)
    
    container_caption = object_detector._get_container_caption(temp_path, output_dir)
    container_caption_state = container_caption
    container_caption = json.dumps(container_caption, indent=4, ensure_ascii=False)

    return container_caption_state, container_caption, temp_path

def tensor_to_python(obj):
    """Convert tensor objects to python native types"""
    if torch.is_tensor(obj):
        return obj.tolist()  # Convert tensor to list
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(item) for item in obj]
    return obj

def initial_detection(container_caption, image_path):
    """Perform initial object detection"""
    initial_detection_results = object_detector._perform_initial_detection(container_caption, image_path, output_dir)
    initial_results_state = initial_detection_results
    # Convert tensors before JSON serialization
    initial_detection_results = tensor_to_python(initial_detection_results)
    initial_detection_results = json.dumps(initial_detection_results, indent=4, ensure_ascii=False)

    return initial_results_state, initial_detection_results, initial_results_state["middle_result_path"]

def process_container(initial_detection_results, image_path):
    """Process container"""
    container_results = object_detector._process_containers(initial_detection_results, image_path, output_dir)
    all_results = object_detector._save_final_results(container_results, image_path, output_dir)
    
    # Simplify container metadata to just phrase->path mapping
    container_meta = {}
    if 'container_meta' in all_results:
        for phrase, container_info in all_results['container_meta'].items():
            path = container_info.get('container_path', '')
            if path:  # Only add if path exists
                container_meta[phrase] = path
    gallery_items = []
    for container_name, image_path in container_meta.items():
        gallery_items.append((image_path, container_name))

    all_results_state = all_results
    all_results = json.dumps(all_results, indent=4, ensure_ascii=False)
    container_meta = json.dumps(container_meta, indent=4, ensure_ascii=False)
    return all_results_state, all_results, all_results_state["final_result_path"], container_meta, gallery_items


def update_gallery(container_meta):
    if not container_meta:
        print("No container meta found")
        return []
    
    gallery_items = []
    for container_name, image_path in container_meta.items():
        label = f"{container_name}"
        print(f"Adding container: {container_name} with path: {image_path}")
        gallery_items.append((image_path, label))
    return gallery_items



# ------------------------------------------------------------
# Add new depth estimation function
# ------------------------------------------------------------
def estimate_depth(image_path):
    """Estimate depth for the input image"""
    depth_map_norm, metric_depth = depth_estimator.process_image(image_path)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_map_norm, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.axis('off')
    
    depth_vis_path = osp.join(output_dir, "depth_visualization.jpg")
    plt.savefig(depth_vis_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    return depth_vis_path, metric_depth




# ------------------------------------------------------------
# Add new function for SAM
# ------------------------------------------------------------
def generate_masks(image_path, all_results):
    """Generate SAM masks for detected objects"""
    overlay_output_path = osp.join(output_dir, "mask_overlay.png")
    mask_output_path = osp.join(output_dir, "masks.npy")
    mask_info_path = osp.join(output_dir, "masks_info.json")
    if config.use_cache and osp.exists(mask_info_path) and osp.exists(mask_output_path) and osp.exists(overlay_output_path):
        all_masks, selected_idx = sam_estimator._load_cached_results(mask_info_path, mask_output_path)
        all_masks_state = all_masks
        return all_masks_state, selected_idx, overlay_output_path
    
    image = Image.open(image_path).convert("RGB")
    masks = []
    all_masks = []
    selected_boxes = []
    selected_idx = []
    for idx, box in enumerate(tqdm(all_results["boxes"], desc="Processing masks")):
        current_mask = sam_estimator._process_single_box(image, box)
        overlap_with_existing = False
        for existing_mask in masks:
            if sam_estimator._check_overlap(current_mask, existing_mask) > 0.95:
                overlap_with_existing = True
                break

        if not overlap_with_existing:
            masks.append(current_mask)
            selected_boxes.append(box)
            selected_idx.append(idx)
        all_masks.append(current_mask)

    sam_estimator._save_results(all_masks, selected_idx, mask_output_path, mask_info_path)
    sam_estimator.visualize_masks(image_path, masks, overlay_output_path)
    all_masks_state = all_masks
    return all_masks_state, selected_idx, overlay_output_path



# ------------------------------------------------------------
# Estimate distances between objects using masks
# ------------------------------------------------------------
def estimate_distances(image_path, all_masks, metric_depth, all_results, selected_idx):
    """Estimate distances between objects using masks"""
    boxes, scores, pred_phrases = all_results["boxes"], all_results["scores"], all_results["pred_phrases"]
    boxes = [boxes[i] for i in selected_idx]
    pred_phrases = [pred_phrases[i] for i in selected_idx]
    masks = [all_masks[i] for i in selected_idx]

    img = cv2.imread(image_path)
    distance_estimator.height, distance_estimator.width = img.shape[:2]
    distance_estimator.intrinsic_parameters = {
        'width': distance_estimator.width, 'height': distance_estimator.height,
        'fx': 1.5 * distance_estimator.width, 'fy': 1.5 * distance_estimator.height, 
        'cx': distance_estimator.width / 2, 'cy': distance_estimator.height / 2,
    }

    # Get point clouds for each mask
    pcd_canonicalized, canonicalized, transform, point_clouds, pcd_paths = \
        distance_estimator._get_segment_pcds(image_path, masks, metric_depth, pred_phrases, output_dir)
    point_clouds = distance_estimator._post_canonicalize_pcd(point_clouds, canonicalized, transform)
    for pcd_path, each_pcd in zip(pcd_paths, point_clouds):
        o3d.io.write_point_cloud(pcd_path, each_pcd)
            
    # Calculate object properties and distances
    centroids = [distance_estimator._calculate_centroid(pcd) for pcd in point_clouds]    
    assert len(centroids) == len(pred_phrases), "Mismatch between pcd and text description"
    relative_positions = distance_estimator._calculate_relative_positions(centroids, pred_phrases)
    
    pred_phrases_state = pred_phrases
    # Ensure relative_positions is a dictionary. Take only the first 5 entries
    if isinstance(relative_positions, list):
        relative_positions = {str(i): v for i, v in enumerate(relative_positions)}
    relative_positions_state = relative_positions
    relative_positions = {str(k): v for k, v in list(relative_positions.items())[:5]}
    relative_positions = json.dumps(relative_positions, indent=4, ensure_ascii=False)
    return pred_phrases_state, relative_positions_state, relative_positions

def simplify_phrases(pred_phrases):
    """Simplify detected object phrases using LLM"""
    with open("prompt/phrase_simplify.txt", "r", encoding="utf-8") as f:
        phrase_simplify_prompt = f.read()
    
    # Process all phrases
    simplified_results = []
    for phrase in pred_phrases:
        query = phrase_simplify_prompt.replace("[Insert the phrase here]", phrase)
        simplified_phrase = phrase_simplify.run_llm(query)
        simplified_results.append(json.loads(simplified_phrase)["simplified_phrase"])
    
    # Handle duplicates with numbering
    phrase_count = {}
    new_pred_phrases = []
    for phrase in simplified_results:
        phrase_count[phrase] = phrase_count.get(phrase, 0) + 1
        new_phrase = f"{phrase}_{phrase_count[phrase]}" if phrase_count[phrase] > 1 else phrase
        new_pred_phrases.append(new_phrase)
    
    # Create mapping for display
    simplification_mapping = {orig: simp for orig, simp in zip(pred_phrases, new_pred_phrases)}
    simplification_mapping = json.dumps(simplification_mapping, indent=4, ensure_ascii=False)
    new_pred_phrases_state = new_pred_phrases
    return new_pred_phrases_state, simplification_mapping

# ------------------------------------------------------------
# Scene Graph Generation
# ------------------------------------------------------------
def generate_scene_graph(image_path, all_masks, new_pred_phrases, selected_idx):
    """Generate scene graph from detected objects and their relationships"""
    # Generate point cloud visualization
    masks = [all_masks[i] for i in selected_idx]
    visualization_path = show_point(image_path, masks, new_pred_phrases, output_dir=output_dir)

    # Generate scene graph using VLM
    object_list = new_pred_phrases + ['wall', 'ceiling', 'floor']
    query = str1 + ", ".join(object_list) + str2
    answer = my_vlm.run_llm(query, visualization_path)
    
    # Save scene graph to file
    answer_path = osp.join(output_dir, "scene_graph.json")
    with open(answer_path, "w") as f:
        json.dump(extract_json_from_string(answer), f, indent=4)
    
    answer_state = answer
    return visualization_path, answer_state, answer

# ------------------------------------------------------------
# Relation Visualization
# ------------------------------------------------------------
def visualize_relations(image_path, masks, pred_phrases, answer, selected_idx):
    """Visualize spatial relations between objects"""
    # Parse relations from answer
    relations = extract_json_from_string(answer)
    relations = parse_json_nodes(relations)
    
    # Generate visualization
    masks = [masks[i] for i in selected_idx]
    output_path = show_relations(image_path, masks, pred_phrases, relations, output_dir=output_dir)
    
    return output_path

# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------
with gr.Blocks(title="Indoor Object Detection", theme=gr.themes.Soft()) as demo:
    # Header
    with gr.Row(elem_id="header"):
        gr.Markdown("""
        # 🏠 ROOT: VLM-based System for Indoor Scene Understanding and Beyond
        
        ### A Comprehensive System for Indoor Scene Analysis and Understanding
        
        *By Research Team @ University of Science and Technology of China & Game AI Center, Tencent IEG*
        
        ### Follow the numbered steps below to analyze your indoor scene 👇
        """)
    
    # State Variables (grouped together)
    states = {
        "temp_image_path": gr.State(value=None),
        "metric_depth_state": gr.State(value=None),
        "detection_results": gr.State(value=None),
        "selected_idx_state": gr.State(value=None),
        "text_descriptions": gr.State(value=None),
        "container_caption_state": gr.State(value=None),
        "initial_results_state": gr.State(value=None),
        "all_results_state": gr.State(value=None),
        "all_masks_state": gr.State(value=None),
        "distance_results_state": gr.State(value=None),
        "pred_phrases_state": gr.State(value=None),
        "new_pred_phrases_state": gr.State(value=None),
        "answer_state": gr.State(value=None),
        "container_meta_state": gr.State(value=None)
    }
    
    # Input Section with Progress Bar
    with gr.Column():
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload Indoor Scene Image", height=300)
            progress = gr.Progress(track_tqdm=True)
        gr.Examples(examples=["asset/0012057.jpg", "asset/3.jpg"], inputs=input_image)
    
    # Main Processing Sections
    with gr.Column():
        # Scene Analysis Section
        gr.Markdown("### 📸 Scene Analysis")
        with gr.Row():
            caption_btn = gr.Button("1️⃣ Generate Caption", variant="primary")
            detect_btn = gr.Button("2️⃣ Initial Detection")
            container_btn = gr.Button("3️⃣ Process Containers")
        
        with gr.Row():
            with gr.Column(scale=1):
                container_caption = gr.Textbox(label="Scene Caption", lines=8, show_copy_button=True)
            with gr.Column(scale=1):
                initial_results = gr.Textbox(label="Detection Results", lines=8, show_copy_button=True)
            with gr.Column(scale=1):
                final_results = gr.Textbox(label="Final Results", lines=8, show_copy_button=True)
        
        with gr.Row():
            middle_image = gr.Image(label="Initial Detection", height=300)
            final_image = gr.Image(label="Final Results", height=300)
            container_gallery = gr.Gallery(label="Detected Objects", columns=2, height=300)

        gr.Markdown("---")  # Divider
        
        # Depth Analysis Section
        gr.Markdown("### 📊 Depth & Spatial Analysis")
        with gr.Row():
            depth_btn = gr.Button("4️⃣ Estimate Depth", variant="primary")
            sam_btn = gr.Button("5️⃣ Generate Masks")
            distance_btn = gr.Button("6️⃣ Calculate Distances")
        
        with gr.Row():
            depth_image = gr.Image(label="Depth Map", height=300)
            mask_image = gr.Image(label="Mask Visualization", height=300)
        
        with gr.Row():
            distance_results = gr.Textbox(label="Distance Results", lines=8, show_copy_button=True)

        gr.Markdown("---")  # Divider
        
        # Scene Understanding Section
        gr.Markdown("### 🔍 Scene Understanding")
        with gr.Row():
            simplify_btn = gr.Button("7️⃣ Simplify Phrases", variant="primary")
            scene_graph_btn = gr.Button("8️⃣ Generate Scene Graph")
            relation_vis_btn = gr.Button("9️⃣ Visualize Relations")
        
        with gr.Row():
            simplified_phrases = gr.Textbox(label="Simplified Phrases", lines=6, show_copy_button=True)
            answer = gr.Textbox(label="Hierarchical Scene Graph", lines=6, show_copy_button=True)
        
        with gr.Row():
            point_cloud_vis = gr.Image(label="Point Cloud Visualization", height=350)
            relation_image = gr.Image(label="Spatial Relations", height=350)

    # Event Handlers (with error handling)
    def handle_error(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                gr.Warning(f"Error: {str(e)}")
                return [None] * len(func.__annotations__['return'])
        return wrapper

    # Connect event handlers with error handling
    caption_btn.click(
        fn=handle_error(generate_caption),
        inputs=[input_image],
        outputs=[states["container_caption_state"], container_caption, states["temp_image_path"]]
    )
    
    detect_btn.click(
        fn=handle_error(initial_detection),
        inputs=[states["container_caption_state"], states["temp_image_path"]],
        outputs=[states["initial_results_state"], initial_results, middle_image]
    )
    container_btn.click(
        fn=handle_error(process_container),
        inputs=[states["initial_results_state"], states["temp_image_path"]],
        outputs=[states["all_results_state"], final_results, final_image, states["container_meta_state"], container_gallery]
    )

    # depth estimation
    depth_btn.click(
        fn=handle_error(estimate_depth),
        inputs=[states["temp_image_path"]],
        outputs=[depth_image, states["metric_depth_state"]]
    )
    # segment anything
    sam_btn.click(
        fn=handle_error(generate_masks),
        inputs=[states["temp_image_path"], states["all_results_state"]],
        outputs=[states["all_masks_state"], states["selected_idx_state"], mask_image]
    )
    # Distance estimation
    distance_btn.click(
        fn=handle_error(estimate_distances),
        inputs=[states["temp_image_path"], states["all_masks_state"], states["metric_depth_state"], states["all_results_state"], states["selected_idx_state"]],
        outputs=[states["pred_phrases_state"], states["distance_results_state"], distance_results]
    )

    simplify_btn.click(
        fn=handle_error(simplify_phrases),
        inputs=[states["pred_phrases_state"]],
        outputs=[states["new_pred_phrases_state"], simplified_phrases]
    )

    scene_graph_btn.click(
        fn=handle_error(generate_scene_graph),
        inputs=[states["temp_image_path"], states["all_masks_state"], states["new_pred_phrases_state"], states["selected_idx_state"]],
        outputs=[point_cloud_vis, states["answer_state"], answer]
    )

    relation_vis_btn.click(
        fn=handle_error(visualize_relations),
        inputs=[states["temp_image_path"], states["all_masks_state"], states["new_pred_phrases_state"], states["answer_state"], states["selected_idx_state"]],
        outputs=[relation_image]
    )



def start_gradio():
    try:
        demo.launch(server_name="11.239.26.17", server_port=8081, share=True)
    except Exception as e:
        demo.launch(share=True)

if __name__ == "__main__":
    start_gradio()