import numpy as np
import open3d as o3d
import cv2
from PIL import Image
import os
import os.path as osp

class IndoorDistanceEstimator:
    def __init__(self, config):
        self.config = config
        self.width = None
        self.height = None
        self.intrinsic_parameters = None

    def _prepare_output_dir(self, image_path):
        output_dir = osp.join(
            self.config.output_dir, 
            osp.basename(osp.dirname(image_path)), 
            osp.basename(image_path).split(".")[0]
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def process_image(self, image_path, masks, text_descriptions, metric_depth):
        self.output_dir = self._prepare_output_dir(image_path)
        img = cv2.imread(image_path)
        self.height, self.width = img.shape[:2]
        self.intrinsic_parameters = {
            'width': self.width, 
            'height': self.height,
            'fx': 1.5 * self.width, 
            'fy': 1.5 * self.height,
            'cx': self.width / 2, 
            'cy': self.height / 2,
        }

        # Get point clouds for each mask
        pcd_canonicalized, canonicalized, transform, point_clouds, pcd_paths = \
            self._get_segment_pcds(image_path, masks, metric_depth, text_descriptions)
        point_clouds = self._post_canonicalize_pcd(point_clouds, canonicalized, transform)
        for pcd_path, each_pcd in zip(pcd_paths, point_clouds):
            o3d.io.write_point_cloud(pcd_path, each_pcd)
            
        # Calculate object properties and distances
        centroids = [self._calculate_centroid(pcd) for pcd in point_clouds]
        colors = [np.mean(np.asarray(pcd.colors), axis=0) for pcd in point_clouds]
        sizes = [pcd.get_axis_aligned_bounding_box().get_extent() for pcd in point_clouds]
        
        assert len(centroids) == len(text_descriptions), "Mismatch between pcd and text description"
        relative_positions = self._calculate_relative_positions(centroids, text_descriptions)
        
        return relative_positions, point_clouds, colors, sizes
    

    def _get_segment_pcds(self, image_path, masks, metric_depth, text_descriptions):
        original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        point_clouds = []
        pcd_paths = []
        # create full scene point cloud
        full_scene_pcd = self._create_point_cloud_indoor(image_path, metric_depth)
        pcd_canonicalized, canonicalized, transform = self._canonicalize_point_cloud(full_scene_pcd)
        
        # create point cloud for each mask
        for i, mask_binary in enumerate(masks):
            masked_rgb = self._apply_mask_to_image(original_image, mask_binary)
            masked_metric_depth = self._apply_mask_to_array(metric_depth, mask_binary)
            masked_rgb_path = f'{self.output_dir}/rgb_{i}_{text_descriptions[i]}.png'
            cv2.imwrite(masked_rgb_path, cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))
            
            # create point cloud
            pcd = self._create_point_cloud_indoor(masked_rgb_path, masked_metric_depth)
            masked_pcd_path = f"{self.output_dir}/pcd_{i}_{text_descriptions[i]}.pcd"
            # o3d.io.write_point_cloud(masked_pcd_path, pcd)
            
            point_clouds.append(pcd)
            pcd_paths.append(masked_pcd_path)
        return pcd_canonicalized, canonicalized, transform,point_clouds, pcd_paths
    
    def _apply_mask_to_image(self, image, mask):
        masked_image = image.copy()
        for c in range(masked_image.shape[2]):
            masked_image[:, :, c] = masked_image[:, :, c] * mask
        return masked_image
    
    def _apply_mask_to_array(self, raw_array, mask_array):
        masked_array = raw_array.copy()
        mask_array = mask_array.astype(masked_array.dtype)
        masked_array = raw_array * mask_array
        return masked_array
    
    def _create_point_cloud_indoor(self, image_path, metric_depth):
        """
        Create point cloud from metric depth
        """
        fx = self.intrinsic_parameters['fx']
        fy = self.intrinsic_parameters['fy']
        cx = self.intrinsic_parameters['cx']
        cy = self.intrinsic_parameters['cy']

        color_image = Image.open(image_path).convert('RGB')
        original_width, original_height = color_image.size

        resized_pred = Image.fromarray(metric_depth).resize((original_width, original_height), Image.NEAREST)

        x, y = np.meshgrid(np.arange(original_width), np.arange(original_height))
        x = (x - cx) / fx
        y = (y - cy) / fy
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return self._filter_point_cloud(pcd)
    
    def _filter_point_cloud(self, point_cloud):
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)

        non_black_indices = np.where(np.any(colors != [0, 0, 0], axis=1))[0]

        filtered_points = points[non_black_indices]
        filtered_colors = colors[non_black_indices]

        filtered_point_cloud = o3d.geometry.PointCloud()
        filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
        return filtered_point_cloud
    
    def _calculate_centroid(self, pcd):
        points = np.asarray(pcd.points)
        return np.mean(points, axis=0)
    
    def _calculate_relative_positions(self, centroids, text_descriptions):
        relative_positions = []
        
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                relative_vector = centroids[j] - centroids[i]
                distance = np.linalg.norm(relative_vector)
                
                relative_positions.append({
                    'object_pair': (text_descriptions[i], text_descriptions[j]),
                    'distance': distance,
                    'relative_vector': relative_vector
                })
                
        return relative_positions 
    
    def _canonicalize_point_cloud(self, pcd, canonicalize_threshold=0.3):
        # Segment the largest plane, assumed to be the floor
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

        canonicalized = False
        if len(inliers) / len(pcd.points) > canonicalize_threshold:
            canonicalized = True

            # Ensure the plane normal points upwards
            if np.dot(plane_model[:3], [0, 1, 0]) < 0:
                plane_model = -plane_model

            # Normalize the plane normal vector
            normal = plane_model[:3] / np.linalg.norm(plane_model[:3])

            # Compute the new basis vectors
            new_y = normal
            new_x = np.cross(new_y, [0, 0, -1])
            new_x /= np.linalg.norm(new_x)
            new_z = np.cross(new_x, new_y)

            # Create the transformation matrix
            transformation = np.identity(4)
            transformation[:3, :3] = np.vstack((new_x, new_y, new_z)).T
            transformation[:3, 3] = -np.dot(transformation[:3, :3], pcd.points[inliers[0]])

            # Apply the transformation
            pcd.transform(transformation)

            # Additional 180-degree rotation around the Z-axis
            rotation_z_180 = np.array([[np.cos(np.pi), -np.sin(np.pi), 0], 
                                       [np.sin(np.pi), np.cos(np.pi), 0], [0, 0, 1]])
            pcd.rotate(rotation_z_180, center=(0, 0, 0))

            return pcd, canonicalized, transformation
        else:
            return pcd, canonicalized, None
    
    def _post_canonicalize_pcd(self, pcds, canonicalization, transformation):
        """
        Post-process point clouds by removing outliers and applying canonicalization
        """
        for idx, pcd in enumerate(pcds):
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            inlier_cloud = pcd.select_by_index(ind)
            if canonicalization:
                inlier_cloud = inlier_cloud.transform(transformation)
            pcds[idx] = inlier_cloud
        return pcds