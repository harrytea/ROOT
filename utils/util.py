import os
import os.path as osp
import json
import cv2
import matplotlib.pyplot as plt
from adjustText import adjust_text
from PIL import Image
import torch
import time
import re

BLUE = '\033[94m'    # For main status messages
GREEN = '\033[92m'   # For step indicators
ENDC = '\033[0m'     # Reset color
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BOLD = '\033[1m'
RED = '\033[91m'     # Added red color

def get_file_list(path, img_extensions=('.jpg', '.jpeg', '.png', '.bmp'), recursive=False):
    if osp.isfile(path):
        return [path] if path.lower().endswith(img_extensions) else []
    elif osp.isdir(path):
        files = []
        if recursive:
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if not filename.startswith('.') and filename.lower().endswith(img_extensions):
                        files.append(osp.join(root, filename))
        else:
            for file in os.listdir(path):
                if not file.startswith('.') and file.lower().endswith(img_extensions):
                    files.append(osp.join(path, file))
        return sorted(files)
    else:
        raise IOError("Path doesn't exist")



def visualize_detections(image_path, boxes, pred_phrases, output_dir, flag, save_file=True):
    """Visualize detection results"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    ax = plt.gca()
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    texts = []
    
    for i, (box, label) in enumerate(zip(boxes, pred_phrases)):
        color = colors[i % len(colors)]
        x0, y0 = int(box[0]), int(box[1])
        w, h = int(box[2]) - int(box[0]), int(box[3]) - int(box[1])
        
        rect = plt.Rectangle((x0, y0), w, h, edgecolor=color, fill=False, linewidth=2.0)
        ax.add_patch(rect)
        text = ax.text(
            box[0], box[1] - 10, label, fontsize=5, color='white',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=1.0, edgecolor='none')
        )
        texts.append(text)

    adjust_text(texts)
    plt.axis('off')

    if save_file:
        os.makedirs(output_dir, exist_ok=True)
        file_name = osp.basename(image_path).split('.')[0]
        output_filename = f"{file_name}_{flag}.jpg"
        plt.savefig(osp.join(output_dir, output_filename), bbox_inches="tight", dpi=600, pad_inches=0.0)
        plt.close()
        return osp.join(output_dir, output_filename)
    else:
        plt.show()
        return None

def save_json(filepath, data):
    """Save data to JSON file"""
    os.makedirs(osp.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def calculate_iou(box, boxes, threshold=0.5):
    """Calculate IoU between a box and a list of boxes"""
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    for b in boxes:
        inter_x1 = max(box[0], b[0])
        inter_y1 = max(box[1], b[1])
        inter_x2 = min(box[2], b[2])
        inter_y2 = min(box[3], b[3])

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            b_area = (b[2] - b[0]) * (b[3] - b[1])
            iou = inter_area / (box_area + b_area - inter_area)
            if iou > threshold:
                return False
    return True

def expand_box(height, width, bbox, expansion_factor=1.5):
    """Expand bounding box by a factor"""
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    
    new_width = (x2 - x1) * expansion_factor
    new_height = (y2 - y1) * expansion_factor
    
    new_x1 = max(0, int(center_x - new_width / 2))
    new_y1 = max(0, int(center_y - new_height / 2))
    new_x2 = min(width, int(center_x + new_width / 2))
    new_y2 = min(height, int(center_y + new_height / 2))
    
    return [new_x1, new_y1, new_x2, new_y2] 

def convert_boxes(boxes, image_path):
    """
    Convert the boxes into (x1, y1, x2, y2) format
    Args:
        boxes: Original boxes in (x_center, y_center, width, height) format
        image_path: Path to the image file
    Returns:
        new_boxes: Converted boxes in (x1, y1, x2, y2) format
    """
    image = Image.open(image_path)
    width, height = image.size
    for i in range(len(boxes)):
        boxes[i] = boxes[i] * torch.Tensor([width, height, width, height])
        boxes[i][:2] -= boxes[i][2:] / 2
        boxes[i][2:] += boxes[i][:2]
    return boxes

def convert_boxes_no_clone(boxes, image_path):
    """
    Convert the boxes into (x1, y1, x2, y2) format
    Args:
        boxes: Original boxes in (x_center, y_center, width, height) format
        image_path: Path to the image file
    Returns:
        new_boxes: Converted boxes in (x1, y1, x2, y2) format
    """
    image = Image.open(image_path)
    width, height = image.size
    new_boxes = []
    for box in boxes:
        scaled_box = box * torch.Tensor([width, height, width, height])
        converted_box = scaled_box.clone()  # Create a copy
        converted_box[:2] -= scaled_box[2:] / 2
        converted_box[2:] += converted_box[:2]
        new_boxes.append(converted_box)
    return new_boxes


def draw_bounding_boxes(image_path, bounding_boxes, save_dir, phrase=None):
    """Draw colored bounding boxes on image for visualization."""
    # Limit to 8 boxes due to color limitation
    bounding_boxes = bounding_boxes[:8]
    image = cv2.imread(image_path)
    save_path = osp.join(save_dir, f"tmp_{phrase}.jpg")

    # Pre-defined colors (BGR format)
    colors = [
        (0, 0, 0),      # Black
        (255, 255, 255), # White
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]

    image_height, image_width, _ = image.shape
    for index, box in enumerate(bounding_boxes):
        # Convert normalized coordinates to pixel coordinates
        x_center = int(box[0] * image_width)
        y_center = int(box[1] * image_height)
        box_width = int(box[2] * image_width)
        box_height = int(box[3] * image_height)

        # Calculate box corners
        x1 = int(max(0, x_center - box_width/2))
        y1 = int(max(0, y_center - box_height/2))
        x2 = int(min(image_width, x_center + box_width/2))
        y2 = int(min(image_height, y_center + box_height/2))

        cv2.rectangle(image, (x1, y1), (x2, y2), colors[index], 9)

    cv2.imwrite(save_path, image)
    return save_path


def vlm_inference(vlm_run_func, query, image_path, sys_message="", max_retries=3, delay=1, strip_json=True):
    for attempt in range(max_retries):
        try:
            response = vlm_run_func(query, image_path, sys_message)
            if strip_json:
                response = response.strip("```").strip("json")
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise RuntimeError(f"Failed to process after {max_retries} attempts") from e
            

def extract_json(text):
    pattern = r'```json(.*?)```'
    result = re.search(pattern, text, re.DOTALL)
    if result:
        return result.group(1).strip()
    else:
        return None
    
def extract_json_from_string(text):
    json_str = extract_json(text)  ###########################################################################################
    data_str = json_str.replace("'", '"')  
    data_json = json.loads(data_str)
    return data_json