import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from collections import defaultdict
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Default camera intrinsic matrix (can be overridden)
DEFAULT_K = np.array([
    [1356.29847, 0.0, 360.59406],
    [0.0, 1437.35719, 245.234035],
    [0.0, 0.0, 1.0]
])

DEFAULT_P = np.array([
    [1356.29847, 0.0, 360.59406, 0.0],
    [0.0, 1437.35719, 245.234035, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

# Average dimensions for common objects (height, width, length) in meters
DEFAULT_DIMS = {
    'car': np.array([1.52, 1.64, 3.85]),
    'truck': np.array([3.07, 2.63, 11.17]),
    'bus': np.array([3.07, 2.63, 11.17]),
    'motorcycle': np.array([1.50, 0.90, 2.20]),
    'bicycle': np.array([1.40, 0.70, 1.80]),
    'person': np.array([1.75, 0.60, 0.60]),
    'dog': np.array([0.80, 0.50, 1.10]),
    'cat': np.array([0.40, 0.30, 0.70]),
    'potted plant': np.array([0.80, 0.40, 0.40]),
    'plant': np.array([0.80, 0.40, 0.40]),
    'chair': np.array([0.80, 0.60, 0.60]),
    'sofa': np.array([0.80, 0.85, 2.00]),
    'table': np.array([0.75, 1.20, 1.20]),
    'bed': np.array([0.60, 1.50, 2.00]),
    'tv': np.array([0.80, 0.15, 1.20]),
    'laptop': np.array([0.02, 0.25, 0.35]),
    'keyboard': np.array([0.03, 0.15, 0.45]),
    'mouse': np.array([0.03, 0.06, 0.10]),
    'book': np.array([0.03, 0.20, 0.15]),
    'bottle': np.array([0.25, 0.10, 0.10]),
    'cup': np.array([0.10, 0.08, 0.08]),
    'vase': np.array([0.30, 0.15, 0.15]),
    '2-Piece-Sectional-Sofa': np.array([0.85, 1.50, 2.50]),
    '4-Piece-Sectional-Sofa': np.array([0.85, 2.00, 3.20]),
    'Armchair': np.array([0.90, 0.85, 0.85]),
    'BBQ': np.array([1.20, 0.70, 1.20]),
    'Bahut': np.array([1.00, 0.50, 1.80]),
    'BeanBag-Chair': np.array([0.80, 0.80, 0.80]),
    'Bedside-Table': np.array([0.50, 0.45, 0.45]),
    'Bench-Chair': np.array([0.90, 0.45, 1.20]),
    'Big-Bed': np.array([0.65, 1.80, 2.00]),
    'Bike': np.array([1.20, 0.60, 1.80]),
    'Bunk-Bed': np.array([1.70, 1.00, 2.00]),
    'Carpet': np.array([0.01, 2.00, 3.00]),
    'Chandelier': np.array([0.70, 0.70, 0.70]),
    'Child-Bed': np.array([0.50, 0.80, 1.60]),
    'Coat-Stand': np.array([1.80, 0.40, 0.40]),
    'Desk': np.array([0.75, 0.60, 1.20]),
    'Desktop': np.array([0.45, 0.20, 0.45]),
    'Dishwasher': np.array([0.85, 0.60, 0.60]),
    'Display-Cabinet': np.array([1.80, 0.50, 1.00]),
    'Dressing-Table': np.array([1.50, 0.45, 1.20]),
    'EV-Charger': np.array([1.20, 0.40, 0.40]),
    'Egg-Chair': np.array([1.20, 1.00, 1.00]),
    'Electronic-Piano': np.array([0.90, 0.40, 1.50]),
    'Exercise-Bench': np.array([0.50, 0.30, 1.30]),
    'Exercise-Bike': np.array([1.20, 0.60, 1.20]),
    'Exercise-Treadmill': np.array([1.40, 0.80, 2.00]),
    'FileDrawer-Storage': np.array([1.00, 0.50, 0.50]),
    'Flat-TV': np.array([0.80, 0.15, 1.50]),
    'Floor-Lamp': np.array([1.70, 0.40, 0.40]),
    'Freezer': np.array([1.50, 0.70, 0.70]),
    'Fridge': np.array([1.80, 0.70, 0.70]),
    'Game-Table': np.array([0.75, 1.00, 1.50]),
    'Gueridon': np.array([0.75, 0.60, 0.60]),
    'Guitar': np.array([1.00, 0.40, 0.10]),
    'HI-FI': np.array([1.20, 0.40, 0.40]),
    'Hot-Tub': np.array([0.90, 2.00, 2.00]),
    'Ironing-Board': np.array([0.90, 0.40, 1.40]),
    'Ladder': np.array([2.00, 0.50, 0.10]),
    'Laundry-Basket': np.array([0.60, 0.50, 0.40]),
    'Lawn-Mower': np.array([1.00, 0.60, 1.20]),
    'Luggage': np.array([0.70, 0.40, 0.25]),
    'Microwave': np.array([0.30, 0.50, 0.40]),
    'Mirror': np.array([1.50, 0.60, 0.05]),
    'Office-Chair': np.array([1.00, 0.60, 0.60]),
    'Ottoman-Chair': np.array([0.40, 0.60, 0.60]),
    'Oven': np.array([0.80, 0.60, 0.60]),
    'Painting': np.array([1.00, 0.70, 0.05]),
    'Papasan-Chair': np.array([0.90, 1.00, 1.00]),
    'Piano': np.array([1.20, 1.50, 0.60]),
    'Plants': np.array([1.00, 0.50, 0.50]),
    'Power-Tool': np.array([0.25, 0.20, 0.30]),
    'Printer': np.array([0.30, 0.50, 0.40]),
    'Shelve-Storage': np.array([1.80, 0.40, 0.80]),
    'Shoe-Rack': np.array([0.50, 0.30, 1.00]),
    'Side-Table': np.array([0.55, 0.50, 0.50]),
    'Sideboard-Credenza-Storage': np.array([0.80, 0.50, 1.80]),
    'Standard-Chair': np.array([0.90, 0.45, 0.45]),
    'Standard-Safe': np.array([0.60, 0.50, 0.50]),
    'Standard-Sofa': np.array([0.85, 0.85, 2.00]),
    'Standard-Table': np.array([0.75, 0.90, 1.60]),
    'Stool': np.array([0.45, 0.40, 0.40]),
    'Sun-Lounger': np.array([0.50, 0.70, 2.00]),
    'TV-Stand': np.array([0.50, 0.40, 1.50]),
    'Table-Lamp': np.array([0.60, 0.30, 0.30]),
    'Table-Umbrella': np.array([2.20, 2.00, 2.00]),
    'Tool-Box': np.array([0.25, 0.30, 0.50]),
    'Tractor-Mower': np.array([1.20, 1.00, 2.00]),
    'Tumble-Dryer': np.array([0.85, 0.60, 0.60]),
    'Twin-Bed': np.array([0.60, 1.00, 2.00]),
    'Vacuum-Cleaner': np.array([1.10, 0.40, 0.40]),
    'Wardrobe': np.array([2.00, 0.60, 1.50]),
    'Washing-Machine': np.array([0.85, 0.60, 0.60]),
    'Wine-Rack': np.array([1.20, 0.40, 0.40])
}

class BBox3DEstimator:
    """
    3D bounding box estimation from 2D detections and depth
    """
    def __init__(self, camera_matrix=None, projection_matrix=None, class_dims=None):
        """
        Initialize the 3D bounding box estimator
        
        Args:
            camera_matrix (numpy.ndarray): Camera intrinsic matrix (3x3)
            projection_matrix (numpy.ndarray): Camera projection matrix (3x4)
            class_dims (dict): Dictionary mapping class names to dimensions (height, width, length)
        """
        self.K = camera_matrix if camera_matrix is not None else DEFAULT_K
        self.P = projection_matrix if projection_matrix is not None else DEFAULT_P
        self.dims = class_dims if class_dims is not None else DEFAULT_DIMS
        
        # Initialize Kalman filters for tracking 3D boxes
        self.kf_trackers = {}
        
        # Store history of 3D boxes for filtering
        self.box_history = defaultdict(list)
        self.max_history = 5
    
    def estimate_3d_box(self, bbox_2d, depth_value, class_name, object_id=None):
        """
        Estimate 3D bounding box from 2D bounding box and depth
        
        Args:
            bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
            depth_value (float): Depth value at the center of the bounding box
            class_name (str): Class name of the object
            object_id (int): Object ID for tracking (None for no tracking)
            
        Returns:
            dict: 3D bounding box parameters
        """
        # Get 2D box center and dimensions
        x1, y1, x2, y2 = bbox_2d
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Get dimensions for the class
        if class_name.lower() in self.dims:
            dimensions = self.dims[class_name.lower()].copy()
        else:
            dimensions = self.dims['car'].copy()
        
        # Adjust dimensions based on 2D box aspect ratio and size
        aspect_ratio_2d = width_2d / height_2d if height_2d > 0 else 1.0
        
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            dimensions[0] = height_2d / 120
            dimensions[1] = dimensions[0] * 0.6
            dimensions[2] = dimensions[0] * 0.6
        
        elif 'person' in class_name.lower():
            dimensions[0] = height_2d / 100
            dimensions[1] = dimensions[0] * 0.3
            dimensions[2] = dimensions[0] * 0.3
        
        distance = 1.0 + (1.0 - depth_value) * 9.0
        
        location = self._backproject_point(center_x, center_y, distance)
        
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            bottom_y = y2
            location[1] = self._backproject_point(center_x, bottom_y, distance)[1]
        
        orientation = self._estimate_orientation(bbox_2d, location, class_name)
        
        box_3d = {
            'dimensions': dimensions,
            'location': location,
            'orientation': orientation,
            'bbox_2d': bbox_2d,
            'object_id': object_id,
            'class_name': class_name
        }
        
        if object_id is not None:
            box_3d = self._apply_kalman_filter(box_3d, object_id)
            
            self.box_history[object_id].append(box_3d)
            if len(self.box_history[object_id]) > self.max_history:
                self.box_history[object_id].pop(0)
            
            box_3d = self._apply_temporal_filter(object_id)
        
        return box_3d
    
    def _backproject_point(self, x, y, depth):
        """
        Backproject a 2D point to 3D space
        
        Args:
            x (float): X coordinate in image space
            y (float): Y coordinate in image space
            depth (float): Depth value
            
        Returns:
            numpy.ndarray: 3D point (x, y, z) in camera coordinates
        """
        point_2d = np.array([x, y, 1.0])
        point_3d = np.linalg.inv(self.K) @ point_2d * depth
        point_3d[1] = point_3d[1] * 0.5
        return point_3d
    
    def _estimate_orientation(self, bbox_2d, location, class_name):
        """
        Estimate orientation of the object
        
        Args:
            bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
            location (numpy.ndarray): 3D location of the object
            class_name (str): Class name of the object
            
        Returns:
            float: Orientation angle in radians
        """
        theta_ray = np.arctan2(location[0], location[2])
        
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            return theta_ray
        
        if 'person' in class_name.lower():
            alpha = 0.0
        else:
            x1, y1, x2, y2 = bbox_2d
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 1.0
            
            if aspect_ratio > 1.5:
                image_center_x = self.K[0, 2]
                if (x1 + x2) / 2 < image_center_x:
                    alpha = np.pi / 2
                else:
                    alpha = -np.pi / 2
            else:
                alpha = 0.0
        
        rot_y = alpha + theta_ray
        return rot_y
    
    def _init_kalman_filter(self, box_3d):
        """
        Initialize a Kalman filter for a new object
        
        Args:
            box_3d (dict): 3D bounding box parameters
            
        Returns:
            filterpy.kalman.KalmanFilter: Initialized Kalman filter
        """
        kf = KalmanFilter(dim_x=11, dim_z=7)
        kf.x = np.array([
            box_3d['location'][0],
            box_3d['location'][1],
            box_3d['location'][2],
            box_3d['dimensions'][1],
            box_3d['dimensions'][0],
            box_3d['dimensions'][2],
            box_3d['orientation'],
            0, 0, 0, 0
        ])
        dt = 1.0
        kf.F = np.eye(11)
        kf.F[0, 7] = dt
        kf.F[1, 8] = dt
        kf.F[2, 9] = dt
        kf.F[6, 10] = dt
        kf.H = np.zeros((7, 11))
        kf.H[0, 0] = 1
        kf.H[1, 1] = 1
        kf.H[2, 2] = 1
        kf.H[3, 3] = 1
        kf.H[4, 4] = 1
        kf.H[5, 5] = 1
        kf.H[6, 6] = 1
        kf.R = np.eye(7) * 0.1
        kf.R[0:3, 0:3] *= 1.0
        kf.R[3:6, 3:6] *= 0.1
        kf.R[6, 6] = 0.3
        kf.Q = np.eye(11) * 0.1
        kf.Q[7:11, 7:11] *= 0.5
        kf.P = np.eye(11) * 1.0
        kf.P[7:11, 7:11] *= 10.0
        return kf
    
    def _apply_kalman_filter(self, box_3d, object_id):
        """
        Apply Kalman filtering to smooth 3D box parameters
        
        Args:
            box_3d (dict): 3D bounding box parameters
            object_id (int): Object ID for tracking
            
        Returns:
            dict: Filtered 3D bounding box parameters
        """
        if object_id not in self.kf_trackers:
            self.kf_trackers[object_id] = self._init_kalman_filter(box_3d)
        
        kf = self.kf_trackers[object_id]
        kf.predict()
        measurement = np.array([
            box_3d['location'][0],
            box_3d['location'][1],
            box_3d['location'][2],
            box_3d['dimensions'][1],
            box_3d['dimensions'][0],
            box_3d['dimensions'][2],
            box_3d['orientation']
        ])
        kf.update(measurement)
        filtered_box = box_3d.copy()
        filtered_box['location'] = np.array([kf.x[0], kf.x[1], kf.x[2]])
        filtered_box['dimensions'] = np.array([kf.x[4], kf.x[3], kf.x[5]])
        filtered_box['orientation'] = kf.x[6]
        return filtered_box
    
    def _apply_temporal_filter(self, object_id):
        """
        Apply temporal filtering to smooth 3D box parameters over time
        
        Args:
            object_id (int): Object ID for tracking
            
        Returns:
            dict: Temporally filtered 3D bounding box parameters
        """
        history = self.box_history[object_id]
        if len(history) < 2:
            return history[-1]
        current_box = history[-1]
        filtered_box = current_box.copy()
        alpha = 0.7
        for i in range(len(history) - 2, -1, -1):
            weight = alpha * (1 - alpha) ** (len(history) - i - 2)
            filtered_box['location'] = filtered_box['location'] * (1 - weight) + history[i]['location'] * weight
            angle_diff = history[i]['orientation'] - filtered_box['orientation']
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            filtered_box['orientation'] += angle_diff * weight
        return filtered_box
    
    def project_box_3d_to_2d(self, box_3d):
        """
        Project 3D bounding box corners to 2D image space
        
        Args:
            box_3d (dict): 3D bounding box parameters
            
        Returns:
            numpy.ndarray: 2D points of the 3D box corners (8x2)
        """
        h, w, l = box_3d['dimensions']
        x, y, z = box_3d['location']
        rot_y = box_3d['orientation']
        class_name = box_3d['class_name'].lower()
        x1, y1, x2, y2 = box_3d['bbox_2d']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        R_mat = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)]
        ])
        if 'plant' in class_name or 'potted plant' in class_name:
            x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
            y_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])
            z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        else:
            x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
            y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
            z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = R_mat @ corners_3d
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])
        corners_2d_homo = self.P @ corners_3d_homo
        corners_2d = corners_2d_homo[:2, :] / corners_2d_homo[2, :]
        mean_x = np.mean(corners_2d[0, :])
        mean_y = np.mean(corners_2d[1, :])
        if abs(mean_x - center_x) > width_2d or abs(mean_y - center_y) > height_2d:
            shift_x = center_x - mean_x
            shift_y = center_y - mean_y
            corners_2d[0, :] += shift_x
            corners_2d[1, :] += shift_y
        return corners_2d.T
    
    def draw_box_3d(self, image, box_3d, color=(0, 255, 0), thickness=2):
        """
        Draw enhanced 3D bounding box on image with better depth perception
        
        Args:
            image (numpy.ndarray): Image to draw on
            box_3d (dict): 3D bounding box parameters
            color (tuple): Color in BGR format
            thickness (int): Line thickness
            
        Returns:
            numpy.ndarray: Image with 3D box drawn
        """
        x1, y1, x2, y2 = [int(coord) for coord in box_3d['bbox_2d']]
        depth_value = box_3d.get('depth_value', 0.5)
        width = x2 - x1
        height = y2 - y1
        offset_factor = 1.0 - depth_value
        offset_x = int(width * 0.3 * offset_factor)
        offset_y = int(height * 0.3 * offset_factor)
        offset_x = max(15, min(offset_x, 50))
        offset_y = max(15, min(offset_y, 50))
        front_tl = (x1, y1)
        front_tr = (x2, y1)
        front_br = (x2, y2)
        front_bl = (x1, y2)
        back_tl = (x1 + offset_x, y1 - offset_y)
        back_tr = (x2 + offset_x, y1 - offset_y)
        back_br = (x2 + offset_x, y2 - offset_y)
        back_bl = (x1 + offset_x, y2 - offset_y)
        overlay = image.copy()
        cv2.rectangle(image, front_tl, front_br, color, thickness)
        cv2.line(image, front_tl, back_tl, color, thickness)
        cv2.line(image, front_tr, back_tr, color, thickness)
        cv2.line(image, front_br, back_br, color, thickness)
        cv2.line(image, front_bl, back_bl, color, thickness)
        cv2.line(image, back_tl, back_tr, color, thickness)
        cv2.line(image, back_tr, back_br, color, thickness)
        cv2.line(image, back_br, back_bl, color, thickness)
        cv2.line(image, back_bl, back_tl, color, thickness)
        pts_top = np.array([front_tl, front_tr, back_tr, back_tl], np.int32)
        pts_top = pts_top.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_top], color)
        pts_right = np.array([front_tr, front_br, back_br, back_tr], np.int32)
        pts_right = pts_right.reshape((-1, 1, 2))
        right_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
        cv2.fillPoly(overlay, [pts_right], right_color)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        class_name = box_3d['class_name']
        obj_id = box_3d['object_id'] if 'object_id' in box_3d else None
        text_y = y1 - 10
        if obj_id is not None:
            cv2.putText(image, f"ID:{obj_id}", (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            text_y -= 15
        cv2.putText(image, class_name, (x1, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        text_y -= 15
        if 'depth_value' in box_3d:
            depth_value = box_3d['depth_value']
            depth_method = box_3d.get('depth_method', 'unknown')
            depth_text = f"D:{depth_value:.2f} ({depth_method})"
            cv2.putText(image, depth_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            text_y -= 15
        if 'score' in box_3d:
            score = box_3d['score']
            score_text = f"S:{score:.2f}"
            cv2.putText(image, score_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        ground_y = y2 + int(height * 0.2)
        cv2.line(image, (int((x1 + x2) / 2), y2), (int((x1 + x2) / 2), ground_y), color, thickness)
        cv2.circle(image, (int((x1 + x2) / 2), ground_y), thickness * 2, color, -1)
        return image
    
    def cleanup_trackers(self, active_ids):
        """
        Clean up Kalman filters and history for objects that are no longer tracked
        
        Args:
            active_ids (list): List of active object IDs
        """
        active_ids_set = set(active_ids)
        for obj_id in list(self.kf_trackers.keys()):
            if obj_id not in active_ids_set:
                del self.kf_trackers[obj_id]
        for obj_id in list(self.box_history.keys()):
            if obj_id not in active_ids_set:
                del self.box_history[obj_id]

class BirdEyeView:
    def __init__(self, size=(600, 600), scale=40, camera_height=1.2, bg_color=(30, 30, 30)):
        """
        Initialize the Bird's Eye View visualizer
        
        Args:
            size (tuple): Size of the BEV image (width, height)
            scale (float): Scale factor (pixels per meter)
            camera_height (float): Height of the camera above ground (meters)
            bg_color (tuple): Background color in BGR format
        """
        self.width, self.height = size
        self.scale = scale
        self.camera_height = camera_height
        self.bg_color = bg_color
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        
        # Create empty BEV image
        self.bev_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Set origin at the bottom center (for upside-down view)
        self.origin_x = self.width // 2
        self.origin_y = self.height - 60  # Near the bottom for upside-down
        
    def reset(self):
        """
        Reset the BEV image with enhanced grid and axes, flipped upside down
        """
        # Set background
        self.bev_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.bev_image[:, :] = self.bg_color
        
        # Draw grid with gradient opacity
        grid_spacing = max(int(self.scale), 20)
        for y in range(self.height - self.origin_y, self.height, grid_spacing):
            flipped_y = self.height - y
            alpha = 1.0 - (y - (self.height - self.origin_y)) / (self.origin_y)
            color = (60, 60, 60)
            overlay = self.bev_image.copy()
            cv2.line(overlay, (0, flipped_y), (self.width, flipped_y), color, 1)
            cv2.addWeighted(overlay, alpha * 0.7, self.bev_image, 1 - alpha * 0.7, 0, self.bev_image)
        for x in range(0, self.width, grid_spacing):
            cv2.line(self.bev_image, (x, 0), (x, self.height), (60, 60, 60), 1)
        
        # Draw coordinate system (X-axis upward, Y-axis rightward)
        axis_length = 100
        cv2.line(self.bev_image, 
                 (self.origin_x, self.origin_y), 
                 (self.origin_x, self.origin_y - axis_length),
                 (0, 180, 0), 2)
        cv2.line(self.bev_image, 
                 (self.origin_x, self.origin_y), 
                 (self.origin_x + axis_length, self.origin_y), 
                 (180, 0, 0), 2)
        
        def draw_text_with_shadow(text, pos, color):
            cv2.putText(self.bev_image, text, (pos[0] + 1, pos[1] + 1), 
                        self.font, self.font_scale, (0, 0, 0), self.font_thickness)
            cv2.putText(self.bev_image, text, pos, 
                        self.font, self.font_scale, color, self.font_thickness)
        
        draw_text_with_shadow("X", 
                              (self.origin_x - 15, self.origin_y - axis_length - 5), 
                              (0, 180, 0))
        draw_text_with_shadow("Y", 
                              (self.origin_x + axis_length - 15, self.origin_y + 20), 
                              (180, 0, 0))
        
        for dist in [1, 2, 3, 4, 5]:
            y = self.origin_y - int(dist * self.scale)
            if y < 30:
                continue
            thickness = 2 if dist == int(dist) else 1
            cv2.line(self.bev_image, 
                     (self.origin_x - 10, y), 
                     (self.origin_x + 10, y), 
                     (150, 150, 150), thickness)
            draw_text_with_shadow(f"{int(dist)}m", 
                                  (self.origin_x + 15, y + 5), 
                                  (200, 200, 200))
        
        draw_text_with_shadow("Bird's Eye View", 
                              (self.width // 2 - 50, 30), 
                              (255, 255, 255))
    
    def draw_box(self, box_3d, color=None):
        """
        Draw a realistic object on the BEV image with depth cues, flipped upside down
        
        Args:
            box_3d (dict): 3D bounding box parameters
            color (tuple): Color in BGR format (None for automatic)
        """
        try:
            class_name = box_3d['class_name'].lower()
            depth_value = box_3d.get('depth_value', 0.5)
            depth = 1.0 + (1.0 - depth_value) * 4.0
            
            if 'bbox_2d' in box_3d:
                x1, y1, x2, y2 = box_3d['bbox_2d']
                width_2d = x2 - x1
                size_factor = width_2d / 100
                size_factor = max(0.5, min(size_factor, 2.0))
            else:
                size_factor = 1.0
            
            depth_scale = 1.0 - (depth - 1.0) / 5.0
            size_factor *= max(0.6, depth_scale)
            
            if color is None:
                color_map = {
                    'car': (0, 0, 200), 'vehicle': (0, 0, 200),
                    'truck': (0, 140, 200), 'bus': (0, 140, 200),
                    'person': (0, 200, 0),
                    'bicycle': (200, 0, 0), 'motorcycle': (200, 0, 0),
                    'potted plant': (0, 200, 200), 'plant': (0, 200, 200)
                }
                color = color_map.get(class_name, (200, 200, 200))
            
            bev_y = self.origin_y - int(depth * self.scale)
            if 'bbox_2d' in box_3d:
                center_x_2d = (x1 + x2) / 2
                image_width = self.bev_image.shape[1]
                rel_x = (center_x_2d / image_width) - 0.5
                bev_x = self.origin_x + int(rel_x * self.width * 0.6)
            else:
                bev_x = self.origin_x
            
            bev_x = max(20, min(bev_x, self.width - 20))
            bev_y = max(30, min(bev_y, self.height - 20))
            
            def draw_with_outline(center, shape_fn, size, fill_color, outline_color=(50, 50, 50)):
                overlay = self.bev_image.copy()
                shape_fn(overlay, fill_color, -1)
                cv2.addWeighted(overlay, 0.8, self.bev_image, 0.2, 0, self.bev_image)
                shape_fn(self.bev_image, outline_color, 1)
            
            if 'person' in class_name:
                radius = int(5 * size_factor)
                draw_with_outline(
                    (bev_x, bev_y),
                    lambda img, c, t: cv2.circle(img, (bev_x, bev_y), radius, c, t),
                    radius,
                    color
                )
            elif 'car' in class_name or 'vehicle' in class_name or 'truck' in class_name or 'bus' in class_name:
                rect_width = int(14 * size_factor)
                rect_length = int(20 * size_factor)
                if 'truck' in class_name or 'bus' in class_name:
                    rect_length = int(28 * size_factor)
                draw_with_outline(
                    (bev_x, bev_y),
                    lambda img, c, t: cv2.rectangle(img,
                        (bev_x - rect_width//2, bev_y - rect_length//2),
                        (bev_x + rect_width//2, bev_y + rect_length//2),
                        c, t),
                    (rect_width, rect_length),
                    color
                )
            elif 'plant' in class_name or 'potted plant' in class_name:
                radius = int(10 * size_factor)
                draw_with_outline(
                    (bev_x, bev_y),
                    lambda img, c, t: cv2.circle(img, (bev_x, bev_y), radius, c, t),
                    radius,
                    color
                )
            else:
                size = int(10 * size_factor)
                draw_with_outline(
                    (bev_x, bev_y),
                    lambda img, c, t: cv2.rectangle(img,
                        (bev_x - size, bev_y - size),
                        (bev_x + size, bev_y + size),
                        c, t),
                    size,
                    color
                )
            
            obj_id = box_3d.get('object_id', None)
            score = box_3d.get('score', None)
            label_parts = []
            if obj_id is not None:
                label_parts.append(f"ID:{obj_id}")
            label_parts.append(class_name.title())
            if score is not None:
                label_parts.append(f"{score:.2f}")
            label = " ".join(label_parts)
            
            text_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            text_x = bev_x - text_size[0] // 2
            text_y = bev_y + 15
            if text_y > self.height - 20:
                text_y = bev_y - 20
            
            bg_color = (50, 50, 50)
            cv2.rectangle(self.bev_image,
                          (text_x - 2, text_y - text_size[1] - 2),
                          (text_x + text_size[0] + 2, text_y + 2),
                          bg_color, -1)
            cv2.putText(self.bev_image, label, (text_x, text_y), 
                        self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            
            overlay = self.bev_image.copy()
            cv2.line(overlay, (self.origin_x, self.origin_y), (bev_x, bev_y), 
                     (100, 100, 100), 1)
            alpha = max(0.4, depth_scale)
            cv2.addWeighted(overlay, alpha, self.bev_image, 1 - alpha, 0, self.bev_image)
        except Exception as e:
            print(f"Error drawing box in BEV: {e}")
    
    def get_image(self):
        """
        Get the BEV image
        Returns:
            numpy.ndarray: BEV image
        """
        return self.bev_image