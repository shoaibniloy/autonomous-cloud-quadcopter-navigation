import datetime
import random
import numpy as np
import json
import pandas as pd

# Class mappings for YOLOv11
CLASS_MAPPINGS = {
    'yolov11': {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
        55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
        65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
        70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
        75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
}

# Realistic dimensions for YOLO classes (in meters) as provided
DIMENSIONS = {
    'car': [1.52, 1.64, 3.85],
    'truck': [3.07, 2.63, 11.17],
    'bus': [3.07, 2.63, 11.17],
    'motorcycle': [1.50, 0.90, 2.20],
    'bicycle': [1.40, 0.70, 1.80],
    'person': [1.75, 0.60, 0.60],
    'dog': [0.80, 0.50, 1.10],
    'cat': [0.40, 0.30, 0.70],
    'potted plant': [0.80, 0.40, 0.40],
    'chair': [0.80, 0.60, 0.60],
    'couch': [0.80, 0.85, 2.00],
    'dining table': [0.75, 1.20, 1.20],
    'bed': [0.60, 1.50, 2.00],
    'tv': [0.80, 0.15, 1.20],
    'laptop': [0.02, 0.25, 0.35],
    'keyboard': [0.03, 0.15, 0.45],
    'mouse': [0.03, 0.06, 0.10],
    'book': [0.03, 0.20, 0.15],
    'bottle': [0.25, 0.10, 0.10],
    'cup': [0.10, 0.08, 0.08],
    'vase': [0.30, 0.15, 0.15],
    '2-Piece-Sectional-Sofa': [0.85, 1.50, 2.50],
    '4-Piece-Sectional-Sofa': [0.85, 2.00, 3.20],
    'Armchair': [0.90, 0.85, 0.85],
    'BBQ': [1.20, 0.70, 1.20],
    'Bahut': [1.00, 0.50, 1.80],
    'BeanBag-Chair': [0.80, 0.80, 0.80],
    'Bedside-Table': [0.50, 0.45, 0.45],
    'Bench-Chair': [0.90, 0.45, 1.20],
    'Big-Bed': [0.65, 1.80, 2.00],
    'Bike': [1.20, 0.60, 1.80],
    'Bunk-Bed': [1.70, 1.00, 2.00],
    'Carpet': [0.01, 2.00, 3.00],
    'Chandelier': [0.70, 0.70, 0.70],
    'Child-Bed': [0.50, 0.80, 1.60],
    'Coat-Stand': [1.80, 0.40, 0.40],
    'Desk': [0.75, 0.60, 1.20],
    'Desktop': [0.45, 0.20, 0.45],
    'Dishwasher': [0.85, 0.60, 0.60],
    'Display-Cabinet': [1.80, 0.50, 1.00],
    'Dressing-Table': [1.50, 0.45, 1.20],
    'EV-Charger': [1.20, 0.40, 0.40],
    'Egg-Chair': [1.20, 1.00, 1.00],
    'Electronic-Piano': [0.90, 0.40, 1.50],
    'Exercise-Bench': [0.50, 0.30, 1.30],
    'Exercise-Bike': [1.20, 0.60, 1.20],
    'Exercise-Treadmill': [1.40, 0.80, 2.00],
    'FileDrawer-Storage': [1.00, 0.50, 0.50],
    'Flat-TV': [0.80, 0.15, 1.50],
    'Floor-Lamp': [1.70, 0.40, 0.40],
    'Freezer': [1.50, 0.70, 0.70],
    'Fridge': [1.80, 0.70, 0.70],
    'Game-Table': [0.75, 1.00, 1.50],
    'Gueridon': [0.75, 0.60, 0.60],
    'Guitar': [1.00, 0.40, 0.10],
    'HI-FI': [1.20, 0.40, 0.40],
    'Hot-Tub': [0.90, 2.00, 2.00],
    'Ironing-Board': [0.90, 0.40, 1.40],
    'Ladder': [2.00, 0.50, 0.10],
    'Laundry-Basket': [0.60, 0.50, 0.40],
    'Lawn-Mower': [1.00, 0.60, 1.20],
    'Luggage': [0.70, 0.40, 0.25],
    'Microwave': [0.30, 0.50, 0.40],
    'Mirror': [1.50, 0.60, 0.05],
    'Office-Chair': [1.00, 0.60, 0.60],
    'Ottoman-Chair': [0.40, 0.60, 0.60],
    'Oven': [0.80, 0.60, 0.60],
    'Painting': [1.00, 0.70, 0.05],
    'Papasan-Chair': [0.90, 1.00, 1.00],
    'Piano': [1.20, 1.50, 0.60],
    'Plants': [1.00, 0.50, 0.50],
    'Power-Tool': [0.25, 0.20, 0.30],
    'Printer': [0.30, 0.50, 0.40],
    'Shelve-Storage': [1.80, 0.40, 0.80],
    'Shoe-Rack': [0.50, 0.30, 1.00],
    'Side-Table': [0.55, 0.50, 0.50],
    'Sideboard-Credenza-Storage': [0.80, 0.50, 1.80],
    'Standard-Chair': [0.90, 0.45, 0.45],
    'Standard-Safe': [0.60, 0.50, 0.50],
    'Standard-Sofa': [0.85, 0.85, 2.00],
    'Standard-Table': [0.75, 0.90, 1.60],
    'Stool': [0.45, 0.40, 0.40],
    'Sun-Lounger': [0.50, 0.70, 2.00],
    'TV-Stand': [0.50, 0.40, 1.50],
    'Table-Lamp': [0.60, 0.30, 0.30],
    'Table-Umbrella': [2.20, 2.00, 2.00],
    'Tool-Box': [0.25, 0.30, 0.50],
    'Tractor-Mower': [1.20, 1.00, 2.00],
    'Tumble-Dryer': [0.85, 0.60, 0.60],
    'Twin-Bed': [0.60, 1.00, 2.00],
    'Vacuum-Cleaner': [1.10, 0.40, 0.40],
    'Wardrobe': [2.00, 0.60, 1.50],
    'Washing-Machine': [0.85, 0.60, 0.60],
    'Wine-Rack': [1.20, 0.40, 0.40]
}

# Default dimensions for unlisted YOLO classes
DEFAULT_DIMENSIONS = [0.5, 0.5, 0.5]

# Sensor key mapping
SENSOR_KEY_MAPPING = {
    "Left Clearance": "Free space in the left direction (X)",
    "Right Clearance": "Free space in the right direction (-X)",
    "Front Clearance": "Free space in the front direction (-Y)",
    "Back Clearance": "Free space in the back direction (Y)",
    "Up Clearance": "Free space in the up direction (Z)",
    "Bottom Clearance": "Free space in the bottom dimension (-Z)",
    "AccelX": "Acceleration in the X direction (m/s²)",
    "AccelY": "Acceleration in the Y direction (m/s²)",
    "AccelZ": "Acceleration in the Z direction (m/s²)",
    "GyroX": "Angular velocity around the X axis (rad/s)",
    "GyroY": "Angular velocity around the Y axis (rad/s)",
    "GyroZ": "Angular velocity around the Z axis (rad/s)"
}

# Detailed scene descriptions (VLM responses) - all 175 provided responses
SCENE_DESCRIPTIONS = [
    "I see a doorway leading to a dining area with a table, chairs, and a bathroom in the background, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I need to carefully navigate through the doorway and avoid the furniture.",
    "I observe a doorway opening to a dining space with a wooden table, chairs, and a bathroom visible, while a bedroom with a messy dressing table, mirror, and bed is on my right; I must proceed cautiously through the doorway and steer clear of the furniture.",
    "I detect an open doorway leading to a dining area with a table and chairs, a bathroom in the background, and a bedroom with a cluttered dressing table, mirror, and bed on the right; I need to navigate carefully through the doorway and avoid obstacles.",
    "I see a doorway ahead connecting to a dining room with a table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on my right; I should move through the doorway and dodge the furniture.",
    "I notice a doorway leading to a dining area with a wooden table, chairs, and a bathroom in the distance, while a bedroom with a cluttered dressing table, mirror, and bed is on my right; I must fly through the doorway and avoid collisions.",
    "I spot an open doorway to a dining space with a table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on the right; I need to carefully navigate through the doorway and avoid the objects.",
    "I observe a doorway opening to a dining area with a table, chairs, and a bathroom in the background, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I should proceed through the doorway and steer clear of obstacles.",
    "I detect a doorway leading to a dining room with a table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on my right; I must navigate through the doorway and avoid the furniture.",
    "I see a doorway ahead connecting to a dining space with a table, chairs, and a bathroom in the background, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I need to fly through the doorway and avoid collisions.",
    "I notice an open doorway to a dining area with a wooden table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on my right; I should carefully navigate through the doorway and dodge obstacles.",
    "I spot a doorway leading to a dining room with a table, chairs, and a bathroom in the distance, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I must proceed through the doorway and avoid the furniture.",
    "I observe a doorway opening to a dining space with a table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on my right; I need to navigate through the doorway and steer clear of objects.",
    "I detect an open doorway leading to a dining area with a table, chairs, and a bathroom in the background, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I should fly through the doorway and avoid obstacles.",
    "I see a doorway ahead connecting to a dining room with a wooden table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on my right; I must carefully navigate through the doorway and dodge the furniture.",
    "I notice a doorway leading to a dining space with a table, chairs, and a bathroom in the distance, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I need to proceed through the doorway and avoid collisions.",
    "I spot an open doorway to a dining area with a table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on the right; I should navigate through the doorway and steer clear of obstacles.",
    "I observe a doorway opening to a dining room with a table, chairs, and a bathroom in the background, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I must fly through the doorway and avoid the furniture.",
    "I detect a doorway leading to a dining space with a wooden table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on my right; I need to carefully navigate through the doorway and dodge objects.",
    "I see a doorway ahead connecting to a dining area with a table, chairs, and a bathroom in the distance, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I should proceed through the doorway and avoid collisions.",
    "I notice an open doorway to a dining room with a table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on my right; I must navigate through the doorway and steer clear of the furniture.",
    "I spot a doorway leading to a dining space with a table, chairs, and a bathroom in the background, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I need to fly through the doorway and avoid obstacles.",
    "I observe a doorway opening to a dining area with a wooden table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on my right; I should carefully navigate through the doorway and dodge the furniture.",
    "I detect an open doorway leading to a dining room with a table, chairs, and a bathroom in the distance, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I must proceed through the doorway and avoid collisions.",
    "I see a doorway ahead connecting to a dining space with a table, chairs, and a bathroom, while a bedroom with a dressing table, mirror, and bed is on my right; I need to navigate through the doorway and steer clear of objects.",
    "I notice a doorway leading to a dining area with a table, chairs, and a bathroom in the background, while a bedroom with a cluttered dressing table, mirror, and bed is on the right; I should fly through the doorway and avoid the furniture.",
    "I am hovering in a room with a bed, two chairs, a window with curtains, a door, and various objects on a dresser; I need to navigate safely around obstacles.",
    "I am flying in a room with a bed, two chairs, a window with blue curtains, a door, and a dresser with items; I must carefully navigate around these obstacles.",
    "I’m in a space with a bed, two red chairs, a window with patterned curtains, an open door, and a dresser; I need to avoid collisions while moving.",
    "I’m inside a room containing a bed with a laptop, two chairs, a window with curtains, a door, and a dresser with objects; I must maneuver safely.",
    "I am navigating a room with a bed, two cushioned chairs, a window with blue curtains, a door, and a dresser with items; I need to ensure I don’t hit anything.",
    "I’m flying through a room with a bed, two chairs with pillows, a window with curtains, a door, and a dresser; I must avoid obstacles to proceed.",
    "I am in a room with a bed holding a laptop, two chairs, a window with blue curtains, a door, and a dresser with items; I need to fly carefully.",
    "I’m hovering in a space with a bed, two red chairs, a window with patterned curtains, an open door, and a dresser; I must navigate around objects.",
    "I am in a room with a bed, two cushioned chairs, a window with curtains, a door leading to another area, and a dresser; I need to navigate safely.",
    "I’m flying in a room with a bed, two chairs with red cushions, a window with blue curtains, a door, and a dresser with items; I must avoid obstacles.",
    "I am navigating a space with a bed, two chairs, a window with patterned curtains, a door, and a dresser with various objects; I need to ensure I don’t collide.",
    "I’m in a room with a bed holding a laptop, two chairs, a window with blue curtains, an open door, and a dresser; I must fly cautiously.",
    "I am hovering in a room with a bed, two red chairs, a window with curtains, a door, and a dresser with items; I need to navigate carefully.",
    "I’m flying through a space with a bed, two cushioned chairs, a window with blue curtains, a door, and a dresser; I must avoid hitting objects.",
    "I am in a room with a bed, two chairs with pillows, a window with patterned curtains, a door, and a dresser with items; I need to maneuver safely.",
    "I’m navigating a room with a bed, two red chairs, a window with blue curtains, an open door, and a dresser; I must avoid obstacles.",
    "I am flying in a space with a bed holding a laptop, two chairs, a window with curtains, a door, and a dresser with objects; I need to ensure safe movement.",
    "I’m hovering in a room with a bed, two cushioned chairs, a window with blue curtains, a door, and a dresser with items; I must navigate carefully.",
    "I am in a room with a bed, two chairs with red cushions, a window with patterned curtains, a door, and a dresser; I need to avoid collisions.",
    "I’m flying through a space with a bed, two chairs, a window with blue curtains, an open door, and a dresser with items; I must move cautiously.",
    "I am navigating a room with a bed, two red chairs, a window with curtains, a door, and a dresser with objects; I need to ensure I don’t hit anything.",
    "I’m in a room with a bed holding a laptop, two chairs, a window with patterned curtains, a door, and a dresser; I must fly safely around obstacles.",
    "I am hovering in a space with a bed, two cushioned chairs, a window with blue curtains, a door, and a dresser with items; I need to navigate carefully.",
    "I’m flying in a room with a bed, two chairs with pillows, a window with curtains, an open door, and a dresser; I must avoid obstacles to proceed.",
    "I am in a room with a bed, two red chairs, a window with blue curtains, a door, and a dresser with objects; I need to maneuver safely.",
    "I am an autonomous drone navigating a bedroom with a bed, a small table with books and flowers, a desk with a laptop, and a window with curtains; I need to avoid obstacles.",
    "I’m an autonomous drone in a bedroom with a bed, a small table with books and flowers, a desk with a laptop, and a window with blue curtains; I need to avoid obstacles.",
    "As a drone, I’m flying through a room with a bed, a table with flowers and books, a desk with a laptop, and a window with patterned curtains; I must navigate carefully.",
    "I’m in a bedroom with a bed, a small table with books and a vase, a desk with a laptop, and a window with blue and white curtains; I need to steer clear of obstacles.",
    "I am navigating a bedroom with a bed, a table with flowers and books, a desk with a laptop, and a window with curtains; I must avoid these obstacles.",
    "I’m an autonomous drone in a room with a bed, a small table with a vase and books, a desk with a laptop, and a window with sheer curtains; I need to navigate around them.",
    "As a drone, I’m in a bedroom with a bed, a table with flowers, a desk with a laptop, and a window with blue curtains; I must avoid collisions.",
    "I’m flying in a bedroom with a bed, a small table with books and a vase, a desk with a laptop, and a window with patterned curtains; I need to avoid obstacles.",
    "I am an autonomous drone navigating a bedroom with a bed, a table with flowers, a desk with a laptop, and a window with curtains; I must fly carefully.",
    "I’m in a room with a bed, a small table with books and flowers, a desk with a laptop, and a window with blue curtains; I need to avoid obstacles.",
    "I am flying in a bedroom with a bed, a table with a vase and books, a desk with a laptop, and a window with sheer curtains; I must navigate safely.",
    "I’m an autonomous drone in a bedroom with a bed, a small table with flowers, a desk with a laptop, and a window with blue and white curtains; I need to avoid obstacles.",
    "As a drone, I’m navigating a room with a bed, a table with books and a vase, a desk with a laptop, and a window with patterned curtains; I must avoid collisions.",
    "I’m in a bedroom with a bed, a small table with flowers and books, a desk with a laptop, and a window with blue curtains; I need to navigate around obstacles.",
    "I am an autonomous drone in a room with a bed, a table with a vase, a desk with a laptop, and a window with sheer curtains; I must fly carefully.",
    "I’m flying in a bedroom with a bed, a small table with books and flowers, a desk with a laptop, and a window with blue and white curtains; I need to avoid obstacles.",
    "As a drone, I’m in a bedroom with a bed, a table with a vase and books, a desk with a laptop, and a window with patterned curtains; I must navigate safely.",
    "I’m an autonomous drone in a room with a bed, a small table with flowers and books, a desk with a laptop, and a window with blue curtains; I need to avoid obstacles.",
    "I am flying in a bedroom with a bed, a table with a vase, a desk with a laptop, and a window with sheer curtains; I must avoid collisions.",
    "I’m in a bedroom with a bed, a small table with books and flowers, a desk with a laptop, and a window with blue and white curtains; I need to steer clear of obstacles.",
    "I am an autonomous drone navigating a room with a bed, a table with a vase and books, a desk with a laptop, and a window with patterned curtains; I must fly carefully.",
    "I’m in a bedroom with a bed, a small table with flowers, a desk with a laptop, and a window with blue curtains; I need to avoid obstacles.",
    "As a drone, I’m flying in a room with a bed, a table with books and a vase, a desk with a laptop, and a window with sheer curtains; I must navigate around them.",
    "I’m an autonomous drone in a bedroom with a bed, a small table with flowers and books, a desk with a laptop, and a window with blue and white curtains; I need to avoid collisions.",
    "I am navigating a bedroom with a bed, a table with a vase, a desk with a laptop, and a window with patterned curtains; I must avoid obstacles.",
    "I’m in a room with a bed, a small table with books and flowers, a desk with a laptop, and a window with blue curtains; I need to navigate safely.",
    "I am hovering in a small indoor room with a red sofa, a dining table with chairs, a sink, a window, a closed door, and various hanging items on the wall; I need to navigate carefully to avoid obstacles.",
    "I am flying in a compact room with a red sofa, a dining table with chairs, a sink, a window, and a closed door; I must carefully maneuver around the furniture and hanging items on the wall.",
    "I’m hovering in a small space with a red sofa, a dining set, a sink, a window with bars, and a door; I need to avoid the chairs and wall decorations.",
    "I am navigating a tight indoor area with a red couch, a glass-top dining table, a sink, a window, and a wooden door; I must steer clear of the hanging bags and furniture.",
    "I’m in a room with a red sofa, a dining table surrounded by chairs, a sink, a window with curtains, and a closed door; I need to avoid obstacles like the hanging scarf and chairs.",
    "I am flying through a small room with a red couch, a dining table with four chairs, a sink, a window, and a door; I must dodge the wall-mounted items and furniture.",
    "I’m in a confined space with a red sofa, a glass dining table, a sink, a window with bars, and a wooden door; I need to navigate around the chairs and hanging objects.",
    "I am hovering in a small room with a red couch, a dining table with chairs, a sink, a window, and a closed door; I must avoid the hanging items and furniture.",
    "I’m flying in a compact area with a red sofa, a dining table, a sink, a window with curtains, and a door; I need to carefully avoid the chairs and wall decorations.",
    "I am navigating a tight indoor space with a red couch, a dining set, a sink, a window, and a wooden door; I must steer clear of the hanging scarf and furniture.",
    "I’m in a small room with a red sofa, a dining table with chairs, a sink, a window with bars, and a closed door; I need to avoid obstacles like the hanging bags.",
    "I am flying through a compact space with a red couch, a glass-top dining table, a sink, a window, and a door; I must dodge the chairs and wall-mounted items.",
    "I’m hovering in a small indoor area with a red sofa, a dining table, a sink, a window with curtains, and a wooden door; I need to navigate around the hanging objects.",
    "I am in a confined room with a red couch, a dining table with chairs, a sink, a window, and a closed door; I must avoid the hanging scarf and furniture.",
    "I’m flying in a tight space with a red sofa, a dining set, a sink, a window with bars, and a door; I need to carefully steer clear of the chairs and wall decorations.",
    "I am hovering in a small room with a red couch, a glass dining table, a sink, a window, and a wooden door; I must avoid the hanging items and furniture.",
    "I’m navigating a compact indoor area with a red sofa, a dining table with chairs, a sink, a window with curtains, and a closed door; I need to dodge obstacles.",
    "I am in a small space with a red couch, a dining table, a sink, a window, and a door; I must navigate around the hanging bags and furniture carefully.",
    "I’m flying through a tight room with a red sofa, a glass-top dining table, a sink, a window with bars, and a wooden door; I need to avoid the chairs and wall-mounted items.",
    "I am hovering in a confined area with a red couch, a dining table with chairs, a sink, a window, and a closed door; I must steer clear of the hanging scarf.",
    "I’m navigating a small indoor space with a red sofa, a dining set, a sink, a window with curtains, and a door; I need to avoid the hanging objects and furniture.",
    "I am in a compact room with a red couch, a glass dining table, a sink, a window, and a wooden door; I must carefully dodge the chairs and wall decorations.",
    "I’m flying through a tight space with a red sofa, a dining table with chairs, a sink, a window with bars, and a closed door; I need to navigate around the hanging items.",
    "I am hovering in a small room with a red couch, a dining table, a sink, a window, and a door; I must avoid obstacles like the hanging bags and furniture.",
    "I’m in a confined indoor area with a red sofa, a glass-top dining table, a sink, a window with curtains, and a wooden door; I need to steer clear of the chairs and wall-mounted items.",
    "I see a living room with two red sofas, a wooden door in the center, a backpack hanging on the wall, two small tables with decorative items, and a refrigerator in the background near a window; I need to navigate around obstacles.",
    "I observe a living room with two red sofas, a wooden door in the middle, a backpack on the wall, two small tables with items, and a refrigerator near a window; I must navigate around obstacles.",
    "I’m in a room with two red couches, a central wooden door, a backpack hanging above, two side tables with decor, and a fridge by the window; I need to avoid obstacles while moving.",
    "I detect a living space with two red sofas, a wooden door in the center, a backpack on the wall, two tables with objects, and a refrigerator near a window; I must carefully fly around obstacles.",
    "I’m scanning a room with two red chairs, a wooden door in the middle, a backpack hanging up, two tables with items, and a fridge by the window; I need to steer clear of obstacles.",
    "I see a living area with two red sofas, a wooden door at the center, a backpack on the wall, two small tables with decor, and a refrigerator near a window; I must avoid obstacles to navigate safely.",
    "I’m flying through a room with two red couches, a central wooden door, a backpack above, two tables with items, and a fridge by the window; I need to maneuver around obstacles carefully.",
    "I notice a living room with two red sofas, a wooden door in the middle, a backpack on the wall, two tables with decorative items, and a refrigerator near a window; I must avoid obstacles to proceed.",
    "I’m in a space with two red chairs, a wooden door at the center, a backpack hanging, two side tables with objects, and a fridge by the window; I need to navigate around obstacles.",
    "I observe a room with two red sofas, a central wooden door, a backpack on the wall, two small tables with decor, and a refrigerator near a window; I must fly carefully to avoid obstacles.",
    "I’m scanning a living area with two red couches, a wooden door in the middle, a backpack hanging up, two tables with items, and a fridge by the window; I need to steer clear of obstacles.",
    "I see a living room with two red sofas, a wooden door at the center, a backpack on the wall, two small tables with decor, and a refrigerator near a window; I must navigate around obstacles safely.",
    "I’m in a room with two red chairs, a central wooden door, a backpack above, two tables with objects, and a fridge by the window; I need to avoid obstacles while moving forward.",
    "I detect a living space with two red sofas, a wooden door in the middle, a backpack hanging, two side tables with items, and a refrigerator near a window; I must carefully fly around obstacles.",
    "I’m flying through a room with two red couches, a wooden door at the center, a backpack on the wall, two tables with decor, and a fridge by the window; I need to avoid obstacles to proceed.",
    "I notice a living area with two red sofas, a central wooden door, a backpack hanging up, two small tables with items, and a refrigerator near a window; I must navigate around obstacles carefully.",
    "I’m in a space with two red chairs, a wooden door in the middle, a backpack on the wall, two tables with objects, and a fridge by the window; I need to steer clear of obstacles.",
    "I observe a room with two red sofas, a wooden door at the center, a backpack hanging, two side tables with decor, and a refrigerator near a window; I must fly carefully to avoid obstacles.",
    "I’m scanning a living room with two red couches, a central wooden door, a backpack above, two tables with items, and a fridge by the window; I need to avoid obstacles while navigating.",
    "I see a living area with two red sofas, a wooden door in the middle, a backpack on the wall, two small tables with decor, and a refrigerator near a window; I must navigate around obstacles safely.",
    "I’m in a room with two red chairs, a central wooden door, a backpack hanging up, two tables with objects, and a fridge by the window; I need to steer clear of obstacles to proceed.",
    "I detect a living space with two red sofas, a wooden door in the center, a backpack on the wall, two side tables with items, and a refrigerator near a window; I must fly carefully around obstacles.",
    "I’m flying through a room with two red couches, a wooden door in the middle, a backpack hanging, two tables with decor, and a fridge by the window; I need to avoid obstacles to move forward.",
    "I notice a living room with two red sofas, a central wooden door, a backpack on the wall, two small tables with objects, and a refrigerator near a window; I must navigate around obstacles safely.",
    "I’m in a space with two red chairs, a wooden door at the center, a backpack above, two tables with items, and a fridge by the window; I need to steer clear of obstacles while proceeding.",
    "I am an autonomous drone navigating a living room with two red chairs, a small table, a wooden door, and a cluttered dressing area; I need to avoid obstacles and exit safely through the door.",
    "I am flying through a living room with two red chairs, a small table, a wooden door, and a cluttered dressing area; I must carefully navigate around obstacles to exit through the door.",
    "I’m in a room with two red chairs, a table, a wooden door, and a messy dressing area; I need to avoid obstacles to safely fly out through the door.",
    "I’m an autonomous drone in a living room with two red chairs, a small table, a wooden door, and a cluttered dressing area; I must maneuver around obstacles to exit via the door.",
    "I see two red chairs, a small table, a wooden door, and a cluttered dressing area in this room; I need to avoid obstacles to reach the door safely.",
    "I am navigating a living room with two red chairs, a table, a wooden door, and a cluttered dressing area; I must exit through the door without hitting obstacles.",
    "I’m flying in a room with two red chairs, a small table, a wooden door, and a messy dressing area; I need to steer clear of obstacles to fly out the door.",
    "I’m in a living room with two red chairs, a table, a wooden door, and a cluttered dressing area; I must avoid obstacles to safely exit through the door.",
    "I am in a room with two red chairs, a small table, a wooden door, and a cluttered dressing area; I need to navigate around obstacles to reach the door.",
    "I’m an autonomous drone in a living room with two red chairs, a table, a wooden door, and a cluttered dressing area; I must avoid obstacles to fly out through the door safely.",
    "I am navigating a room with two red chairs, a small table, a wooden door, and a messy dressing area; I need to carefully fly around obstacles to exit via the door.",
    "I’m in a living room with two red chairs, a table, a wooden door, and a cluttered dressing area; I must steer clear of obstacles to reach the door safely.",
    "I see two red chairs, a small table, a wooden door, and a cluttered dressing area; I need to avoid obstacles to exit through the door.",
    "I am in a room with two red chairs, a table, a wooden door, and a messy dressing area; I must navigate carefully to fly out the door without hitting obstacles.",
    "I’m flying in a living room with two red chairs, a small table, a wooden door, and a cluttered dressing area; I need to avoid obstacles to safely reach the door.",
    "I’m an autonomous drone in a room with two red chairs, a table, a wooden door, and a cluttered dressing area; I must maneuver around obstacles to exit through the door.",
    "I am in a living room with two red chairs, a small table, a wooden door, and a cluttered dressing area; I need to navigate around obstacles to reach the door safely.",
    "I’m navigating a room with two red chairs, a table, a wooden door, and a cluttered dressing area; I must avoid obstacles to fly out through the door.",
    "I am in a living room with two red chairs, a small table, a wooden door, and a cluttered dressing area; I need to steer clear of obstacles to exit via the door.",
    "I’m flying in a room with two red chairs, a table, a wooden door, and a messy dressing area; I must carefully avoid obstacles to reach the door safely.",
    "I see two red chairs, a small table, a wooden door, and a cluttered dressing area; I need to navigate around obstacles to exit through the door.",
    "I’m an autonomous drone in a living room with two red chairs, a table, a wooden door, and a cluttered dressing area; I must avoid obstacles to fly out the door safely.",
    "I am in a room with two red chairs, a small table, a wooden door, and a messy dressing area; I need to carefully maneuver around obstacles to reach the door.",
    "I’m navigating a living room with two red chairs, a table, a wooden door, and a cluttered dressing area; I must steer clear of obstacles to safely exit through the door.",
    "I am flying in a room with two red chairs, a small table, a wooden door, and a cluttered dressing area; I need to avoid obstacles to reach the door safely.",
    "I am an autonomous drone navigating a room with a bed, a laptop, two red chairs, a desk, and various objects; I need to avoid obstacles and find a clear path.",
    "I am flying in a room with a bed, a laptop, two red chairs, a desk, and various objects; I must carefully navigate around obstacles to find a clear path.",
    "I’m an autonomous drone in a space with a bed, a laptop, two red chairs, a desk, and items; I need to avoid obstacles to chart a safe route.",
    "I’m in a room with a bed, a laptop, two red chairs, a desk, and various objects; I must maneuver around obstacles to find a clear path.",
    "I am navigating a room with a bed, a laptop, two red chairs, a desk, and items; I need to avoid obstacles to proceed safely.",
    "I’m flying through a space with a bed, a laptop, two red chairs, a desk, and various objects; I must find a clear path while avoiding collisions.",
    "I am in a room with a bed, a laptop, two red chairs, a desk, and items; I need to navigate carefully to avoid obstacles.",
    "I’m an autonomous drone in a room with a bed, a laptop, two red chairs, a desk, and various objects; I must steer clear of obstacles to find a safe path.",
    "I am flying in a space with a bed, a laptop, two red chairs, a desk, and items; I need to avoid obstacles to navigate safely.",
    "I’m in a room with a bed, a laptop, two red chairs, a desk, and various objects; I must find a clear path without hitting anything.",
    "I am navigating a room with a bed, a laptop, two red chairs, a desk, and items; I need to carefully avoid obstacles to proceed.",
    "I’m an autonomous drone in a space with a bed, a laptop, two red chairs, a desk, and various objects; I must navigate around obstacles to find a clear route.",
    "I am in a room with a bed, a laptop, two red chairs, a desk, and items; I need to steer clear of obstacles to chart a safe path.",
    "I’m flying through a room with a bed, a laptop, two red chairs, a desk, and various objects; I must avoid obstacles to find a clear path.",
    "I am navigating a space with a bed, a laptop, two red chairs, a desk, and items; I need to carefully maneuver around obstacles.",
    "I’m in a room with a bed, a laptop, two red chairs, a desk, and various objects; I must avoid collisions to find a safe route.",
    "I am an autonomous drone in a room with a bed, a laptop, two red chairs, a desk, and items; I need to navigate safely around obstacles.",
    "I’m flying in a space with a bed, a laptop, two red chairs, a desk, and various objects; I must steer clear of obstacles to proceed.",
    "I am in a room with a bed, a laptop, two red chairs, a desk, and items; I need to find a clear path while avoiding obstacles.",
    "I’m navigating a room with a bed, a laptop, two red chairs, a desk, and various objects; I must carefully avoid obstacles to chart a safe path.",
    "I am an autonomous drone in a space with a bed, a laptop, two red chairs, a desk, and items; I need to navigate around obstacles to find a clear route.",
    "I’m in a room with a bed, a laptop, two red chairs, a desk, and various objects; I must avoid obstacles to proceed safely.",
    "I am flying through a room with a bed, a laptop, two red chairs, a desk, and items; I need to steer clear of obstacles to find a clear path.",
    "I’m navigating a space with a bed, a laptop, two red chairs, a desk, and various objects; I must carefully maneuver around obstacles.",
    "I am in a room with a bed, a laptop, two red chairs, a desk, and items; I need to avoid obstacles to navigate safely."
]

def generate_timestamp():
    return datetime.datetime.now().isoformat()

def generate_yolo_detections():
    num_objects = random.randint(0, 5)
    detections = []
    for _ in range(num_objects):
        obj_class = random.choice(list(CLASS_MAPPINGS['yolov11'].values()))
        distance = round(random.uniform(0.5, 10.0) + random.gauss(0, 0.1), 2)
        dimensions = DIMENSIONS.get(obj_class, DEFAULT_DIMENSIONS)
        dimensions = [round(d + random.gauss(0, 0.05), 2) for d in dimensions]
        orientation = round(random.uniform(0, 2 * np.pi) + random.gauss(0, 0.01), 2)
        detections.append({
            "Name of the detected object": obj_class,
            "Distance of the object from you (in meters)": distance,
            "Dimensions of the object (in meters)": dimensions,
            "orientation": orientation
        })
    return detections

def generate_scene_description(yolo_detections):
    if not yolo_detections:
        return "clear path"
    num_objects = len(yolo_detections)
    if num_objects > 3:
        return random.choice(SCENE_DESCRIPTIONS)
    elif num_objects == 1:
        obj = yolo_detections[0]
        return f"{obj['Name of the detected object']} at {obj['Distance of the object from you (in meters)']}m ahead"
    else:
        return "several objects in vicinity"

def generate_sensor_readings(tof_bias=None):
    tof_keys = ["Left Clearance", "Right Clearance", "Front Clearance", "Back Clearance", "Up Clearance", "Bottom Clearance"]
    tof_values = {}
    for key in tof_keys:
        if tof_bias and key in tof_bias:
            value = tof_bias[key] + random.gauss(0, 10)
        else:
            value = random.uniform(300, 2000) + random.gauss(0, 10)
        tof_values[key] = max(50, round(value))
    imu_keys = ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"]
    imu_values = {key: round(random.gauss(0, 0.1), 2) for key in imu_keys}
    return {**tof_values, **imu_values}

def determine_ideal_output(sensors, yolo_detections, is_yaw_instance=True):
    tof_directions = {
        "Left Clearance": (90, 0.5, 0),   # yaw (deg), vx, vy
        "Right Clearance": (-90, -0.5, 0),
        "Front Clearance": (0, 0, -0.5),
        "Back Clearance": (180, 0, 0.5)
    }
    lateral_tof = ["Left Clearance", "Right Clearance", "Front Clearance", "Back Clearance"]
    vertical_tof = ["Up Clearance", "Bottom Clearance"]

    # Drone parameters
    mass = 0.5  # kg
    drag_coeff = 0.1  # drag coefficient
    dt = 0.1  # time step (s)
    k_repulse = 1e6  # repulsive force constant
    max_v = 0.5  # max lateral velocity (m/s)
    max_vz = 0.3  # max vertical velocity (m/s)
    max_yaw_rate = 0.5  # max yaw rate (rad/s)

    # Current velocities (estimated from IMU accelerations)
    current_vx = sensors["AccelX"] * dt  # Approximate from acceleration
    current_vy = sensors["AccelY"] * dt
    current_vz = sensors["AccelZ"] * dt
    current_yaw_rate = sensors["GyroZ"]  # rad/s

    # Initialize forces
    fx, fy, fz = 0.0, 0.0, 0.0

    # Repulsive forces from ToF sensors
    for key in lateral_tof:
        dist = sensors[key] / 1000.0  # Convert mm to m
        if dist < 0.3:  # Close obstacle
            force = k_repulse / (dist ** 2 + 0.01)  # Avoid division by zero
            if key == "Left Clearance":
                fx -= force  # Push right
            elif key == "Right Clearance":
                fx += force  # Push left
            elif key == "Front Clearance":
                fy += force  # Push back
            elif key == "Back Clearance":
                fy -= force  # Push forward
        elif dist > 1.0 and is_yaw_instance:  # Attract to clear path
            force = k_repulse / (dist ** 0.5)  # Weaker attraction
            if key == "Left Clearance":
                fx += force * 0.1
            elif key == "Right Clearance":
                fx -= force * 0.1
            elif key == "Front Clearance":
                fy -= force * 0.1
            elif key == "Back Clearance":
                fy += force * 0.1

    for key in vertical_tof:
        dist = sensors[key] / 1000.0
        if dist < 0.3:
            force = k_repulse / (dist ** 2 + 0.01)
            if key == "Up Clearance":
                fz -= force  # Push down
            elif key == "Bottom Clearance":
                fz += force  # Push up

    # Repulsive forces from YOLO detections
    for obj in yolo_detections:
        dist = obj["Distance of the object from you (in meters)"]
        if dist < 2.0:
            force = k_repulse / (dist ** 2 + 0.01)
            # Assume object is in the direction of min ToF for simplicity
            min_tof_key = min(lateral_tof, key=lambda k: sensors[k])
            if min_tof_key == "Left Clearance":
                fx -= force
            elif min_tof_key == "Right Clearance":
                fx += force
            elif min_tof_key == "Front Clearance":
                fy += force
            elif min_tof_key == "Back Clearance":
                fy -= force

    # YOLO influence: Stop if person is too close
    for obj in yolo_detections:
        if obj["Name of the detected object"] == "person" and obj["Distance of the object from you (in meters)"] < 2:
            return {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.0}

    # Compute accelerations
    ax = (fx / mass) - drag_coeff * current_vx
    ay = (fy / mass) - drag_coeff * current_vy
    az = (fz / mass) - drag_coeff * current_vz

    # Update velocities
    vx = current_vx + ax * dt
    vy = current_vy + ay * dt
    vz = current_vz + az * dt

    # Cap velocities
    vx = max(min(vx, max_v), -max_v)
    vy = max(min(vy, max_v), -max_v)
    vz = max(min(vz, max_vz), -max_vz)

    # Determine yaw based on max ToF distance
    tof_values = [sensors[key] for key in lateral_tof]
    max_tof_idx = np.argmax(tof_values)
    max_tof_key = lateral_tof[max_tof_idx]
    target_yaw = np.deg2rad(tof_directions[max_tof_key][0])

    # Smooth yaw adjustment
    yaw_rate = min(max((target_yaw - current_yaw_rate) / dt, -max_yaw_rate), max_yaw_rate)
    yaw = current_yaw_rate + yaw_rate * dt

    return {"vx": float(vx), "vy": float(vy), "vz": float(vz), "yaw": float(yaw)}

def generate_instance(tof_bias=None, is_yaw_instance=True):
    timestamp = generate_timestamp()
    yolo = generate_yolo_detections()
    vlm = generate_scene_description(yolo)
    sensors = generate_sensor_readings(tof_bias)
    input_data = {
        "timestamp": timestamp,
        "yolo": yolo,
        "vlm": vlm,
        "sensors": sensors
    }
    output = determine_ideal_output(sensors, yolo, is_yaw_instance)
    return {
        "input": input_data,
        "output": output
    }

# Generate dataset
dataset = []

# 80 yaw movement instances
yaw_configs = [
    ("Left Clearance", 1000, 2000),
    ("Right Clearance", 1000, 2000),
    ("Front Clearance", 1000, 2000),
    ("Back Clearance", 1000, 2000)
]
for direction, min_dist, max_dist in yaw_configs:
    for _ in range(20):
        tof_bias = {direction: random.uniform(min_dist, max_dist)}
        dataset.append(generate_instance(tof_bias, is_yaw_instance=True))

# 60 vy/vz obstacle avoidance instances
obstacle_configs = [
    (["Left Clearance", "Right Clearance"], "vx"),
    (["Front Clearance", "Back Clearance"], "vy"),
    (["Up Clearance", "Bottom Clearance"], "vz")
]
for directions, _ in obstacle_configs:
    for _ in range(20):
        direction = random.choice(directions)
        tof_bias = {direction: random.uniform(50, 299)}
        dataset.append(generate_instance(tof_bias, is_yaw_instance=False))

# Save to JSON
with open("drone_navigation_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} instances and saved to 'drone_navigation_dataset.json'")