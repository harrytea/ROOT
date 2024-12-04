import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from adjustText import adjust_text
import numpy as np
from utils.show_point import get_center, mask_wo_intersection  # Reuse existing functions
import json


def parse_json_nodes(data, parent=None, relation=None):
    results = []
    
    if isinstance(data, dict):
        if not data:  # Check for leaf node
            results.append(parent)
        else:
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        for k, v in item.items():
                            results.append([parent, key, k])
                            results.extend(parse_json_nodes(v, k, key))
                else:
                    results.extend(parse_json_nodes(value, key, relation))
    
    return results



def show_relations(image_path, masks, obj_names, relations, output_dir="outputs/visualizations"):
    """Process an image and visualize object relations with arrows
    
    Args:
        image_path (str): Path to the input image
        masks (list): List of numpy arrays containing masks
        obj_names (list): List of object names corresponding to masks
        relations (list): List of [obj1, relation, obj2] relationships
        output_dir (str): Directory to save output visualizations
        
    Returns:
        str: Path to the saved visualization image, or None if processing failed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    ori_img = plt.imread(image_path)
    
    # Get image dimensions
    img_height, img_width = ori_img.shape[:2]
    
    # Add fixed positions for special objects (wall, ceiling, floor)
    special_objects = {
        'wall': (img_height * 0.1, img_width * 0.9),  # Top right
        'ceiling': (img_height * 0.1, img_width * 0.1),  # Top left
        'floor': (img_height * 0.9, img_width * 0.5),  # Bottom center
    }
    
    # Create a mapping from object names to mask indices
    obj_to_idx = {name: idx for idx, name in enumerate(obj_names)}
    
    new_masks = mask_wo_intersection(masks)
    texts = []
    
    # Set a better-looking chart style
    plt.style.use('default')  # Use the default style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.figure(figsize=(12, 8))
    plt.imshow(ori_img)
    
    # Store the centers of all objects
    centers = {}
    
    # Set fixed parameters (moved to the function front)
    ARROW_WIDTH = 2.0  # Reduce arrow width
    ARROW_HEAD_WIDTH = 10.0  # Reduce arrow head width
    ARROW_HEAD_LENGTH = 12.0  # Reduce arrow head length
    ARROW_ALPHA = 0.7  # Slightly increase transparency
    POINT_RADIUS = 8.0  # Increase point radius
    
    # Update color scheme, use softer colors
    colors = [
        '#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6',
        '#1abc9c', '#e67e22', '#34495e', '#27ae60', '#d35400',
        '#8e44ad', '#16a085', '#f39c12', '#2980b9', '#c0392b',
        '#7f8c8d', '#95a5a6', '#bdc3c7', '#e84393', '#fd79a8',
        '#6c5ce7', '#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5',
        '#ff8b94', '#00b894', '#00cec9', '#0984e3', '#6c5ce7',
        '#6c5ce7', '#fd79a8', '#00b894', '#00cec9', '#0984e3',
        '#ffeaa7', '#fab1a0', '#ff7675', '#fd79a8', '#74b9ff'
    ]
    
    # Update arrow colors with more options
    arrow_colors = [
        '#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6',
        '#1abc9c', '#e67e22', '#34495e', '#27ae60', '#d35400',
        '#8e44ad', '#16a085', '#f39c12', '#2980b9', '#c0392b',
        '#7f8c8d', '#95a5a6', '#bdc3c7', '#e84393', '#fd79a8'
    ]
    
    # Create a mapping from object types to colors
    unique_types = set()
    
    # First, add the centers of special objects
    for obj_name, center in special_objects.items():
        centers[obj_name] = center
        # Add special objects to unique_types
        unique_types.add(obj_name)
    
    # Process regular objects
    for mask, obj_name in zip(new_masks, obj_names):
        center = get_center(mask)
        centers[obj_name] = center
        obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name
        unique_types.add(obj_type)
    
    # Update type_to_color mapping
    type_to_color = {t: colors[i % len(colors)] for i, t in enumerate(sorted(unique_types))}
    
    # Collect all unique relationship types and assign colors
    unique_relations = set()
    for relation in relations:
        if len(relation) == 3:
            _, relation_type, _ = relation
            unique_relations.add(relation_type)
    
    relation_to_color = {rel: arrow_colors[i % len(arrow_colors)] 
                        for i, rel in enumerate(sorted(unique_relations))}
    
    # Create a list of legend handles
    legend_elements = []
    
    # First, add object types to the legend
    for obj_type, color in type_to_color.items():
        legend_elements.append(plt.Line2D([0], [0], 
                                        marker='o',
                                        color=color,
                                        markeredgecolor='white',
                                        markeredgewidth=1.5,
                                        markersize=POINT_RADIUS,
                                        linestyle='None',  # Only display points, no lines
                                        label=obj_type))
    
    # Add relationship types to the legend
    for relation_type in unique_relations:
        arrow_color = relation_to_color[relation_type]
        legend_elements.append(plt.Line2D([0], [0], 
                                        color=arrow_color, 
                                        alpha=ARROW_ALPHA,
                                        marker='>',
                                        markersize=12,
                                        markeredgewidth=2,
                                        label=relation_type))
    
    # Draw the centers of all objects (including special objects)
    for obj_name, center in centers.items():
        obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name
        color = type_to_color[obj_type]
        
        # Draw the outer circle
        plt.plot(center[1], center[0], marker='o', 
                markersize=POINT_RADIUS + 2,  # Slightly larger outer circle
                color='white',
                zorder=3)
        # Draw the inner circle
        plt.plot(center[1], center[0], marker='o', 
                markersize=POINT_RADIUS,
                color=color,
                markeredgecolor='white',
                markeredgewidth=1.5,
                zorder=4)
    
    # Define a set of arrow colors
    arrow_colors = [
        '#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6',
        '#1abc9c', '#e67e22', '#34495e', '#27ae60', '#d35400'
    ]
    
    # Set fixed arrow parameters
    ARROW_WIDTH = 2.0  # Reduce arrow width
    ARROW_HEAD_WIDTH = 10.0  # Reduce arrow head width
    ARROW_HEAD_LENGTH = 12.0  # Reduce arrow head length
    ARROW_ALPHA = 0.7  # Slightly increase transparency
    POINT_RADIUS = 8.0  # Increase point radius
    
    # Draw relationship arrows
    for relation in relations:
        if len(relation) == 3:  # Only process valid triples
            obj1, relation_type, obj2 = relation
            if obj1 in centers and obj2 in centers:
                start = centers[obj1]
                end = centers[obj2]
                
                # Calculate arrow direction and length
                dx = end[1] - start[1]
                dy = end[0] - start[0]
                length = np.sqrt(dx**2 + dy**2)
                
                # Calculate unit vector
                unit_dx = dx / length
                unit_dy = dy / length
                
                # Adjust start and end points to ensure the arrow touches the edge of the point
                start_x = start[1] + unit_dx * POINT_RADIUS
                start_y = start[0] + unit_dy * POINT_RADIUS
                end_x = end[1] - unit_dx * (POINT_RADIUS + ARROW_HEAD_LENGTH/2)
                end_y = end[0] - unit_dy * (POINT_RADIUS + ARROW_HEAD_LENGTH/2)
                
                # Calculate new differences
                new_dx = end_x - start_x
                new_dy = end_y - start_y
                
                # Use the color of the relationship
                arrow_color = relation_to_color[relation_type]
                
                # Create arrow
                plt.arrow(start_x, start_y, new_dx, new_dy,
                         color=arrow_color,
                         width=ARROW_WIDTH,
                         head_width=ARROW_HEAD_WIDTH,
                         head_length=ARROW_HEAD_LENGTH,
                         length_includes_head=True,
                         alpha=ARROW_ALPHA,
                         zorder=2)
    
    # Add legend, now containing object types and relationships
    plt.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1, 0.5),
              title='Objects & Relations',
              frameon=True,
              facecolor='white',
              edgecolor='lightgray',
              fontsize=10,  # Increase font size
              title_fontsize=12,  # Increase title font size
              borderpad=1,
              labelspacing=1.2)  # Increase spacing between legend items
    
    # Adjust image size and margins
    plt.gcf().set_size_inches(16, 9)  # Use 16:9 aspect ratio
    
    # Adjust text position to avoid overlap
    adjust_text(texts, lim=200)
    plt.axis('off')
    
    # Save result
    output_filename = osp.basename(osp.splitext(image_path)[0]) + '_relations.jpg'
    print(output_filename)
    output_path = osp.join(output_dir, output_filename)
    plt.savefig(output_path, 
                bbox_inches="tight", 
                dpi=300,  # Reduce DPI to reduce file size but maintain clarity
                pad_inches=0.2)  # Add a little margin
    plt.close()
    
    return output_path
    


if __name__ == "__main__":
    image_path = "asset/3.jpg"
    masks = np.load("outputs/asset/3/masks.npy")

    with open("outputs/asset/3/answer.json", "r") as f:
        relations = json.load(f)
    relations = parse_json_nodes(relations)
    with open("outputs/asset/3/new_pred_phrases.json", "r") as f:
        obj_names = json.load(f)["predictions"]
    success = show_relations(image_path, masks, obj_names, relations)
    if success:
        print("Relations visualization saved successfully")
    else:
        print("Failed to create relations visualization") 