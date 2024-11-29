import os
import os.path as osp
import shutil
import json
from tqdm import tqdm
import glob
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
from adjustText import adjust_text

import json
import numpy as np
import matplotlib.image as pltimg



def delete_node(json_data, key_to_delete):
    if isinstance(json_data, dict):
        if key_to_delete in json_data:
            del json_data[key_to_delete]
        else:
            keys_to_delete = []
            for key in json_data:
                delete_node(json_data[key], key_to_delete)
                # Convert empty lists to empty dictionaries for specific relationship keys
                if isinstance(json_data[key], list) and not json_data[key] and key in ['support', 'contain', 'hang', 'attach']:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del json_data[key]
    elif isinstance(json_data, list):
        for item in json_data:
            if key_to_delete in item:
                json_data.remove(item)
            else:
                delete_node(item, key_to_delete)


def extract_items(data, exclude_keys=('support', 'attach', 'hang', 'contain')):
    items = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key not in exclude_keys:
                items.append(key)
            items.extend(extract_items(value))
    elif isinstance(data, list):
        for item in data:
            items.extend(extract_items(item))
    return items



def find_centroid(contour):
    moments = cv2.moments(contour)
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])
    return centroid_y, centroid_x

def is_boundary_point(mask, point):
    row, col = point
    if row == 0 or col == 0 or row == mask.shape[0] - 1 or col == mask.shape[1] - 1:
        return True
    return False

def find_nearest_inner_point(mask, contour, centroid):
    min_distance = float("inf")
    nearest_inner_point = None

    for point in contour:
        point = tuple(point[0])
        if not is_boundary_point(mask, point):
            distance = np.sqrt((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_inner_point = point

    return nearest_inner_point

def get_center(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    centroid = find_centroid(largest_contour)
    if is_boundary_point(mask, centroid):
        return find_nearest_inner_point(mask, largest_contour, centroid)
    return centroid

def mask_wo_intersection(masks):
    new_masks = masks.copy()  # 求解新的mask，以便获取point所在位置
    for i, mask in enumerate(masks):
        is_contained = False  # 检查是否被其他mask包含
        for j, other_mask in enumerate(masks):
            if i != j and np.all(mask <= other_mask):
                is_contained = True
                break
        if is_contained:
            new_masks[i] = mask
            continue

        # 计算与其他mask的交集
        intersection = mask
        for j, other_mask in enumerate(masks):
            if i != j:
                intersection = np.logical_and(intersection, other_mask)
        # 减去交集部分
        new_masks[i] = mask ^ intersection
    return new_masks


def split_list_into_parts(lst, num_parts):
    avg = len(lst) / float(num_parts)
    out = []
    last = 0.0

    while last < len(lst):
        out.append(lst[int(last):int(last + avg)])
        last += avg

    return out

def show_point(image_path, masks, text_descriptions, output_dir="outputs/visualizations"):
    """Process a single image and generate visualization with labels
    
    Args:
        image_path (str): Path to the input image
        masks (list): List of numpy arrays containing masks
        text_descriptions (list): List of text descriptions corresponding to masks
        output_dir (str): Directory to save output visualizations
        
    Returns:
        str: Path to the saved visualization image, or None if processing failed
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        new_masks = mask_wo_intersection(masks)
        texts = []
        ori_img = pltimg.imread(image_path)
        plt.imshow(ori_img)

        for mask, label in zip(new_masks, text_descriptions):
            center = get_center(mask)
            texts.append(plt.gca().text(center[1], center[0], label, fontsize=6, color='red'))
            plt.plot(center[1], center[0], marker='*', markersize=6)

        adjust_text(texts, lim=200)
        plt.axis('off')
        
        # Save to output directory with original filename
        output_filename = osp.basename(osp.splitext(image_path)[0]) + '_point.jpg'
        output_path = osp.join(output_dir, output_filename)
        plt.savefig(output_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        
        plt.close()
        return output_path
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

if __name__ == "__main__":
    # These would be provided externally in actual use
    example_masks = []  # List of numpy arrays
    example_descriptions = []  # List of strings
    
    success = show_point(image_path, example_masks, example_descriptions)
    if success:
        print("Image processed successfully")
    else:
        print("Failed to process image")

