import numpy as np
import cv2
import torch
from PIL import Image
import serial
import re
import time
import logging
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import pipeline
import threading
import queue

# Setup logging
logging.basicConfig(filename='/home/shoaib/Drone/YOLO+LLM/depth_comparison.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# DepthEstimator class (adapted from your second script)
class DepthEstimator:
    def __init__(self, model_size='small', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.pipe_device = 'cpu' if device == 'mps' else device
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        model_name = model_map.get(model_size.lower(), model_map['small'])
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            logging.info(f"Loaded Depth Anything V2 {model_size} on {self.pipe_device}")
        except Exception as e:
            logging.error(f"Error loading model on {self.pipe_device}: {e}")
            self.pipe_device = 'cpu'
            self.pipe = pipeline(task="depth-estimation", model=model_name, device='cpu')
            logging.info(f"Fell back to CPU for Depth Anything V2 {model_size}")

    def estimate_depth(self, image):
        start_time = time.time()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        depth_result = self.pipe(pil_image)
        depth_map = depth_result["depth"]
        if isinstance(depth_map, Image.Image):
            depth_map = np.array(depth_map)
        elif isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        latency = time.time() - start_time
        logging.info(f"Depth estimation latency: {latency:.4f} seconds")
        return depth_map

# Serial port configuration
SERIAL_PORT = "/dev/ttyUSB0"  # Update if different
BAUD_RATE = 115200
TIMEOUT = 0.02
INVALID_READINGS = {8190, 65535}

# Regex pattern for front ToF sensor
pattern = re.compile(r"Front: (\d+) mm")

# Thread-safe queue for ToF readings
tof_queue = queue.Queue()

def initialize_serial():
    """Initialize serial connection for ToF sensor."""
    try:
        ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=TIMEOUT)
        logging.info(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
        return ser
    except serial.SerialException as e:
        logging.error(f"Failed to connect to {SERIAL_PORT}: {e}")
        raise

def read_tof_serial(ser):
    """Read front ToF sensor data in a separate thread."""
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                match = pattern.search(line)
                if match:
                    front_depth_mm = int(match.group(1))
                    if front_depth_mm not in INVALID_READINGS:
                        tof_queue.put(front_depth_mm / 1000.0)  # Convert to meters
                    else:
                        logging.warning(f"Invalid front ToF reading: {front_depth_mm} mm")
                else:
                    logging.debug(f"No match for line: {line}")
        except Exception as e:
            logging.error(f"Error reading serial: {e}")
            time.sleep(0.1)

def normalize_depth(depth, min_depth=0.2, max_depth=5.0):
    """Normalize depth to [0, 1] for visualization."""
    depth = np.clip(depth, min_depth, max_depth)
    return (depth - min_depth) / (max_depth - min_depth)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open webcam")
        raise RuntimeError("Failed to open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize ToF sensor
    try:
        ser = initialize_serial()
    except serial.SerialException:
        logging.error("Exiting due to serial connection failure")
        cap.release()
        return

    tof_thread = threading.Thread(target=read_tof_serial, args=(ser,), daemon=True)
    tof_thread.start()

    # Initialize depth estimator
    model = DepthEstimator(model_size='small')
    transform = Compose([Resize((384, 384)), ToTensor()])

    # Parameters
    max_depth = 5.0  # Adjust based on your ToF sensor's range (e.g., 2m for VL53L0X)
    roi = None  # [x1, y1, x2, y2] or None for center
    frame_count = 0

    while True:
        start_time = time.time()

        # Capture webcam frame
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame")
            continue
        frame = cv2.resize(frame, (384, 384))

        # Skip every other frame for performance
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # Estimate monocular depth
        depth_map = model.estimate_depth(frame)

        # Get latest ToF depth
        tof_depth = None
        try:
            while not tof_queue.empty():
                tof_depth = tof_queue.get_nowait()  # Get latest reading
        except queue.Empty:
            pass

        # Align monocular depth to ToF
        h, w = depth_map.shape
        if roi is None:
            cx, cy = w // 2, h // 2
            mono_depth_at_point = depth_map[cy, cx]
        else:
            x1, y1, x2, y2 = [int(coord) for coord in roi]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            region = depth_map[y1:y2, x1:x2]
            mono_depth_at_point = np.median(region) if region.size > 0 else depth_map[cy, cx]

        # Scale monocular depth
        depth_map_scaled = depth_map
        if tof_depth is not None and mono_depth_at_point > 0:
            scale = tof_depth / mono_depth_at_point
            depth_map_scaled = depth_map * scale
            abs_rel = abs(depth_map_scaled[cy, cx] - tof_depth) / tof_depth if tof_depth > 0 else 0
        else:
            abs_rel = 0
            logging.warning("No valid ToF depth or monocular depth for scaling")

        # Normalize for visualization
        depth_map_vis = normalize_depth(depth_map_scaled, max_depth=max_depth)
        tof_depth_vis = np.ones_like(depth_map_vis) * (normalize_depth(tof_depth, max_depth=max_depth) if tof_depth else 0.5)
        error_map = np.abs(depth_map_scaled - (tof_depth if tof_depth else max_depth/2))
        error_map_vis = normalize_depth(error_map, min_depth=0, max_depth=2)

        # Convert to displayable formats
        depth_map_vis = (depth_map_vis * 255).astype(np.uint8)
        tof_depth_vis = (tof_depth_vis * 255).astype(np.uint8)
        error_map_vis = (error_map_vis * 255).astype(np.uint8)
        depth_map_color = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_INFERNO)
        tof_depth_color = cv2.applyColorMap(tof_depth_vis, cv2.COLORMAP_INFERNO)
        error_map_color = cv2.applyColorMap(error_map_vis, cv2.COLORMAP_COOL)

        # Annotate depths
        cv2.putText(frame, f"ToF: {tof_depth:.2f}m" if tof_depth else "ToF: N/A",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(depth_map_color, f"ToF: {tof_depth:.2f}m" if tof_depth else "ToF: N/A",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(error_map_color, f"AbsRel: {abs_rel:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if roi is None:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(depth_map_color, (cx, cy), 5, (0, 0, 255), -1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(depth_map_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Combine for display
        top_row = np.hstack((frame, depth_map_color))
        bottom_row = np.hstack((tof_depth_color, error_map_color))
        display = np.vstack((top_row, bottom_row))

        # Add titles
        cv2.putText(display, "RGB Frame", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "Monocular Depth", (w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "ToF Depth", (10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "Absolute Error", (w + 10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display
        cv2.imshow("Depth Comparison", display)
        latency = time.time() - start_time
        logging.info(f"Frame latency: {latency:.4f} seconds, AbsRel: {abs_rel:.3f}")

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program terminated by user")
    except Exception as e:
        logging.error(f"Program failed: {e}")