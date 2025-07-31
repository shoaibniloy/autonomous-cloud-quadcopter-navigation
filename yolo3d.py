#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path
import json
import zmq
import msgpack
import zlib
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QGridLayout
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
from threading import Lock

# Define devices
cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

# Set up logging
logging.basicConfig(filename='yolo3d.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Log PyTorch and CUDA status
logging.info(f"PyTorch version: {torch.__version__}")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
logging.info(f"CUDA device: {cuda_device}")
logging.info(f"CPU device: {cpu_device}")

# Set MPS fallback for Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules (assumed to be available)
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator

def setup_zmq_publisher(port="5555", retries=3, delay=1):
    """Initialize ZMQ PUB socket with retries."""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    for attempt in range(retries):
        try:
            socket.bind(f"tcp://*:{port}")
            time.sleep(3)
            logging.info(f"ZMQ publisher bound to tcp://*:{port}")
            return context, socket
        except zmq.error.ZMQError as e:
            logging.error(f"ZMQ bind attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
    raise zmq.error.ZMQError("Failed to bind ZMQ socket after retries")

def send_zmq_message(socket, data):
    """Send compressed msgpack data via ZMQ."""
    try:
        packed = msgpack.packb(data, use_bin_type=True)
        compressed = zlib.compress(packed)
        socket.send(compressed)
        logging.debug(f"Sent ZMQ message: {data}")
    except zmq.error.ZMQError as e:
        logging.error(f"Error sending ZMQ message: {e}")
        print(f"Error sending ZMQ message: {e}")

class CameraSubscriber(QThread):
    """Thread to receive camera frames from ZMQ."""
    frame_received = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, port="5556"):
        super().__init__()
        self.port = port
        self.running = False
        self.context = None
        self.socket = None

    def initialize(self):
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(f"tcp://localhost:{self.port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            time.sleep(1)
            logging.info(f"ZMQ subscriber connected to tcp://localhost:{self.port}")
            return True
        except zmq.error.ZMQError as e:
            self.error_occurred.emit(f"Failed to set up ZMQ subscriber: {e}")
            logging.error(f"ZMQ subscriber setup failed: {e}")
            return False

    def run(self):
        self.running = True
        while self.running:
            try:
                frame_data = self.socket.recv()
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self.frame_received.emit(frame)
                    logging.debug("Frame received")
                time.sleep(0.033)
            except zmq.error.ZMQError as e:
                logging.error(f"Error receiving ZMQ frame: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        logging.info("CameraSubscriber stopped")

class LLMSubscriber(QThread):
    """Thread to receive LLM velocity commands from ZMQ."""
    velocity_received = pyqtSignal(list)  # Emits [vx, vy, vz]
    error_occurred = pyqtSignal(str)

    def __init__(self, port="5557"):
        super().__init__()
        self.port = port
        self.running = False
        self.context = None
        self.socket = None

    def initialize(self):
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(f"tcp://localhost:{self.port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            time.sleep(1)
            logging.info(f"ZMQ LLM subscriber connected to tcp://localhost:{self.port}")
            return True
        except zmq.error.ZMQError as e:
            self.error_occurred.emit(f"Failed to set up ZMQ LLM subscriber: {e}")
            logging.error(f"ZMQ LLM subscriber setup failed: {e}")
            return False

    def run(self):
        self.running = True
        while self.running:
            try:
                message = self.socket.recv_string(flags=zmq.NOBLOCK)
                if message == "acknowledged":
                    continue
                try:
                    data = json.loads(message)
                    response = data.get("response", "")
                    lines = response.split("\n")
                    action = next((line for line in lines if line.startswith("Action:")), None)
                    if action:
                        action_text = action.replace("Action: ", "").strip()
                        if action_text == "Maintain current velocity":
                            velocity = [0.0, 0.0, 0.0]
                        else:
                            parts = [p.strip() for p in action_text.replace("Set velocity ", "").split(",")]
                            vx = float(parts[0].split("=")[1])
                            vy = float(parts[1].split("=")[1])
                            vz = float(parts[2].split("=")[1])
                            velocity = [vx, vy, vz]
                        self.velocity_received.emit(velocity)
                        logging.debug(f"Emitted LLM velocity: {velocity}")
                except (json.JSONDecodeError, ValueError, IndexError) as e:
                    logging.error(f"Error parsing LLM response: {e}, message: {message}")
            except zmq.error.Again:
                time.sleep(0.05)  # No message available
            except zmq.error.ZMQError as e:
                logging.error(f"Error receiving ZMQ LLM message: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        logging.info("LLMSubscriber stopped")

class VideoProcessor(QThread):
    frame_processed = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    fps_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, output_path="output.mp4", yolo_model_size="nano", depth_model_size="small", conf_threshold=0.25, iou_threshold=0.45, enable_tracking=True, enable_bev=True):
        super().__init__()
        self.output_path = output_path
        self.yolo_model_size = yolo_model_size
        self.depth_model_size = depth_model_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_tracking = enable_tracking
        self.enable_bev = enable_bev
        self.cuda_device = cuda_device  # For detection and depth
        self.cpu_device = cpu_device    # For BEV
        self.running = False
        self.zmq_context = None
        self.zmq_socket = None
        self.out = None
        self.frame = None
        self.width = 240
        self.height = 180

    def set_frame(self, frame):
        self.frame = frame

    def initialize(self):
        try:
            self.zmq_context, self.zmq_socket = setup_zmq_publisher(port="5555")
            logging.info("ZMQ publisher initialized")
        except zmq.error.ZMQError as e:
            self.error_occurred.emit(f"Failed to set up ZMQ publisher: {e}")
            logging.error(f"ZMQ setup failed: {e}")
            return False

        print(f"Using CUDA device: {self.cuda_device} for detection and depth estimation")
        print(f"Using CPU device: {self.cpu_device} for BEV")
        logging.info(f"Using CUDA device: {self.cuda_device} for detection and depth estimation")
        logging.info(f"Using CPU device: {self.cpu_device} for BEV")

        print("Initializing models...")
        logging.info("Initializing models...")
        try:
            self.detector = ObjectDetector(
                model_size="best.pt",
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                classes=None,
                device=self.cuda_device
            )
            logging.info("Object detector initialized with epoch26.pt on CUDA")
        except Exception as e:
            print(f"Error initializing object detector with epoch26.pt: {e}")
            logging.error(f"Error initializing object detector with epoch26.pt: {e}")
            print("Falling back to CPU for object detection")
            logging.info("Falling back to CPU for object detection")
            try:
                self.detector = ObjectDetector(
                    model_size="best.pt",
                    conf_thres=self.conf_threshold,
                    iou_thres=self.iou_threshold,
                    classes=None,
                    device='cpu'
                )
                logging.info("Object detector initialized with epoch26.pt on CPU")
            except Exception as e:
                self.error_occurred.emit(f"Failed to initialize object detector with epoch26.pt on CPU: {e}")
                logging.error(f"Failed to initialize object detector with epoch26.pt on CPU: {e}")
                return False

        try:
            self.depth_estimator = DepthEstimator(
                model_size=self.depth_model_size,
                device=self.cuda_device
            )
            logging.info("Depth estimator initialized on CUDA")
        except Exception as e:
            print(f"Error initializing depth estimator: {e}")
            logging.error(f"Error initializing depth estimator: {e}")
            print("Falling back to CPU for depth estimation")
            logging.info("Falling back to CPU for depth estimation")
            try:
                self.depth_estimator = DepthEstimator(
                    model_size=self.depth_model_size,
                    device='cpu'
                )
            except Exception as e:
                self.error_occurred.emit(f"Failed to initialize depth estimator on CPU: {e}")
                logging.error(f"Failed to initialize depth estimator on CPU: {e}")
                return False

        try:
            self.bbox3d_estimator = BBox3DEstimator()
            logging.info("BBox3DEstimator initialized")
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize BBox3DEstimator: {e}")
            logging.error(f"Failed to initialize BBox3DEstimator: {e}")
            return False

        if self.enable_bev:
            try:
                # Try initializing BirdEyeView with device parameter
                self.bev = BirdEyeView(scale=60, size=(300, 300), device=self.cpu_device)
                logging.info("BirdEyeView initialized with device parameter")
            except TypeError:
                # Fallback if device parameter is not supported
                print("BirdEyeView does not support device parameter, initializing without it")
                logging.warning("BirdEyeView does not support device parameter")
                try:
                    self.bev = BirdEyeView(scale=60, size=(300, 300))
                    logging.info("BirdEyeView initialized without device parameter")
                except Exception as e:
                    self.error_occurred.emit(f"Failed to initialize BirdEyeView: {e}")
                    logging.error(f"Failed to initialize BirdEyeView: {e}")
                    return False

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, 30, (self.width, self.height))
            logging.info("Video writer initialized")
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize video writer: {e}")
            logging.error(f"Failed to initialize video writer: {e}")
            return False

        return True

    def run(self):
        self.running = True
        frame_count = 0
        start_time = time.time()
        fps_display = "FPS: --"

        while self.running:
            if self.frame is None:
                time.sleep(0.033)
                continue

            try:
                # Start total processing timer
                start_total = time.time()

                frame = self.frame.copy()
                original_frame = frame.copy()
                detection_frame = frame.copy()
                depth_frame = frame.copy()
                result_frame = frame.copy()

                # Measure object detection latency
                start_detection = time.time()
                detection_frame, detections = self.detector.detect(detection_frame, track=self.enable_tracking)
                if self.cuda_device.type == 'cuda':
                    torch.cuda.synchronize()
                latency_detection = time.time() - start_detection
                logging.debug(f"Detected {len(detections)} objects")

                # Measure depth estimation latency
                start_depth = time.time()
                depth_map = self.depth_estimator.estimate_depth(original_frame)
                if self.cuda_device.type == 'cuda':
                    torch.cuda.synchronize()
                latency_depth = time.time() - start_depth
                depth_colored = self.depth_estimator.colorize_depth(depth_map)

                # Measure 3D bounding box processing latency
                start_bbox3d = time.time()
                boxes_3d = []
                active_ids = []
                for detection in detections:
                    try:
                        bbox, score, class_id, obj_id = detection
                        class_name = self.detector.get_class_names()[class_id]
                        if class_name.lower() in ['person', 'cat', 'dog']:
                            center_x = int((bbox[0] + bbox[2]) / 2)
                            center_y = int((bbox[1] + bbox[3]) / 2)
                            depth_value = self.depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                            depth_method = 'center'
                        else:
                            depth_value = self.depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                            depth_method = 'median'

                        # Ensure CPU-based data
                        if isinstance(bbox, torch.Tensor):
                            bbox = bbox.cpu().numpy()
                        if isinstance(score, torch.Tensor):
                            score = score.cpu().numpy()
                        if isinstance(depth_value, torch.Tensor):
                            depth_value = depth_value.cpu().numpy()

                        box_3d = {
                            'bbox_2d': bbox,
                            'depth_value': float(depth_value),
                            'depth_method': depth_method,
                            'class_name': class_name,
                            'object_id': obj_id,
                            'score': float(score)
                        }
                        boxes_3d.append(box_3d)
                        if obj_id is not None:
                            active_ids.append(obj_id)
                    except Exception as e:
                        logging.error(f"Error processing detection: {e}")
                        continue

                self.bbox3d_estimator.cleanup_trackers(active_ids)

                if frame_count % 10 == 0:
                    detection_data = {
                        "frame_number": frame_count,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
                        "boxes_3d": [
                            {
                                "bbox_2d": [float(box['bbox_2d'][0]), float(box['bbox_2d'][1]), 
                                            float(box['bbox_2d'][2]), float(box['bbox_2d'][3])],
                                "depth_value": float(box['depth_value']),
                                "depth_method": box['depth_method'],
                                "class_name": box['class_name'].lower(),
                                "object_id": box['object_id'],
                                "score": float(box['score']),
                                "3d_center": box.get('3d_center', [0.0, 0.0, 0.0]),
                                "dimensions": box.get('dimensions', [0.0, 0.0, 0.0]),
                                "orientation": box.get('orientation', 0.0)
                            } for box in boxes_3d
                        ]
                    }
                    print(json.dumps(detection_data, indent=2))
                    logging.info(f"Generated detection data with {len(detection_data['boxes_3d'])} detections")
                    send_zmq_message(self.zmq_socket, detection_data)

                    try:
                        with open("detection_log.jsonl", "a") as f:
                            f.write(json.dumps(detection_data) + "\n")
                    except Exception as e:
                        logging.error(f"Error appending to log file: {e}")

                for box_3d in boxes_3d:
                    try:
                        class_name = box_3d['class_name'].lower()
                        color = {
                            'car': (0, 0, 255), 'vehicle': (0, 0, 255),
                            'person': (0, 255, 0),
                            'bicycle': (255, 0, 0), 'motorcycle': (255, 0, 0),
                            'potted plant': (0, 255, 255), 'plant': (0, 255, 255)
                        }.get(class_name, (255, 255, 255))
                        result_frame = self.bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                    except Exception as e:
                        logging.error(f"Error drawing box: {e}")
                        continue
                latency_bbox3d = time.time() - start_bbox3d

                # Measure BEV generation latency
                bev_image = np.zeros((300, 300, 3), dtype=np.uint8)
                if self.enable_bev:
                    start_bev = time.time()
                    try:
                        self.bev.reset()
                        for box_3d in boxes_3d:
                            # Ensure box_3d values are CPU-based
                            for key, value in box_3d.items():
                                if isinstance(value, torch.Tensor):
                                    logging.warning(f"GPU tensor detected in BEV box_3d: {key}")
                                    box_3d[key] = value.cpu().numpy()
                            self.bev.draw_box(box_3d)
                        bev_image = self.bev.get_image()
                        logging.debug("BEV image generated on CPU")
                    except Exception as e:
                        logging.error(f"Error drawing BEV: {e}")
                    latency_bev = time.time() - start_bev
                else:
                    latency_bev = 0.0

                frame_count += 1
                if frame_count % 10 == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fps_value = frame_count / elapsed_time
                    fps_display = f"FPS: {fps_value:.1f}"
                    self.fps_updated.emit(fps_display)

                cv2.putText(result_frame, f"{fps_display} | Device: {self.cuda_device}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                try:
                    depth_height = self.height // 4
                    depth_width = depth_height * self.width // self.height
                    depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                    result_frame[0:depth_height, 0:depth_width] = depth_resized
                except Exception as e:
                    logging.error(f"Error adding depth map to result: {e}")

                # Calculate total latency before writing to video or emitting signal
                latency_total = time.time() - start_total

                # Write latencies to JSON file
                latency_data = {
                    'detection': latency_detection,
                    'depth': latency_depth,
                    'bbox3d': latency_bbox3d,
                    'bev': latency_bev,
                    'total': latency_total
                }
                with open('yolo3d_latency.json', 'w') as f:
                    json.dump(latency_data, f)

                self.out.write(result_frame)
                self.frame_processed.emit(result_frame, depth_colored, detection_frame, bev_image)

            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                continue

    def stop(self):
        self.running = False
        if self.out:
            self.out.release()
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
        logging.info("VideoProcessor stopped")

class HoverWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.normal_border = "border: 1px solid #CED4DA;"
        self.hover_border = "border: 1px solid #66B0FF;"
        self.setStyleSheet(self.normal_border)

    def enterEvent(self, event):
        self.setStyleSheet(self.hover_border)

    def leaveEvent(self, event):
        self.setStyleSheet(self.normal_border)

class StartStopButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.normal_style = """
            QPushButton {
                background: #FFFFFF;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px 10px;
                font-size: 11px;
                color: #343A40;
            }
        """
        self.pressed_style = """
            QPushButton {
                background: #E8ECEF;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px 10px;
                font-size: 11px;
                color: #343A40;
            }
        """
        self.start_style = """
            QPushButton {
                background: #28A745;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px 10px;
                font-size: 11px;
                color: #FFFFFF;
            }
        """
        self.stop_style = """
            QPushButton {
                background: #DC3545;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px 10px;
                font-size: 11px;
                color: #FFFFFF;
            }
        """
        self.setStyleSheet(self.normal_style)

    def mousePressEvent(self, event):
        self.setStyleSheet(self.pressed_style)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.text() == "Start":
            self.setStyleSheet(self.start_style)
        else:
            self.setStyleSheet(self.stop_style)
        super().mouseReleaseEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Perception Module")
        self.setMinimumSize(620, 620)

        try:
            print("Initializing CameraSubscriber...")
            logging.info("Initializing CameraSubscriber")
            self.camera_subscriber = CameraSubscriber(port="5556")
            self.camera_subscriber.frame_received.connect(self.on_frame_received)
            self.camera_subscriber.error_occurred.connect(self.show_error)
        except Exception as e:
            self.show_error(f"Failed to initialize CameraSubscriber: \'e")
            logging.error(f"Failed to initialize CameraSubscriber: {e}")
            return

        try:
            print("Initializing LLMSubscriber...")
            logging.info("Initializing LLMSubscriber")
            self.llm_subscriber = LLMSubscriber(port="5557")
            self.llm_subscriber.velocity_received.connect(self.on_velocity_received)
            self.llm_subscriber.error_occurred.connect(self.show_error)
        except Exception as e:
            self.show_error(f"Failed to initialize LLMSubscriber: {e}")
            logging.error(f"Failed to initialize LLMSubscriber: {e}")
            return

        try:
            print("Initializing VideoProcessor...")
            logging.info("Initializing VideoProcessor")
            self.processor = VideoProcessor(
                output_path="output.mp4",
                yolo_model_size="nano",
                depth_model_size="small",
                conf_threshold=0.25,
                iou_threshold=0.45,
                enable_tracking=True,
                enable_bev=True
            )
            self.processor.frame_processed.connect(self.update_frames)
            self.processor.fps_updated.connect(self.update_fps)
            self.processor.error_occurred.connect(self.show_error)
        except Exception as e:
            self.show_error(f"Failed to initialize VideoProcessor: {e}")
            logging.error(f"Failed to initialize VideoProcessor: {e}")
            return

        try:
            print("Setting up UI...")
            logging.info("Setting up UI")
            self.setup_ui()
        except Exception as e:
            self.show_error(f"Failed to set up UI: {e}")
            logging.error(f"Failed to set up UI: {e}")
            return

        try:
            print("Initializing processor...")
            logging.info("Initializing processor")
            if not self.processor.initialize():
                self.show_error("Failed to initialize video processor")
                return
        except Exception as e:
            self.show_error(f"Processor initialization failed: {e}")
            logging.error(f"Processor initialization failed: {e}")
            return

        try:
            print("Initializing camera subscriber...")
            logging.info("Initializing camera subscriber")
            if not self.camera_subscriber.initialize():
                self.show_error("Failed to initialize camera subscriber")
                return
        except Exception as e:
            self.show_error(f"Camera subscriber initialization failed: {e}")
            logging.error(f"Camera subscriber initialization failed: {e}")
            return

        try:
            print("Initializing LLM subscriber...")
            logging.info("Initializing LLM subscriber")
            if not self.llm_subscriber.initialize():
                self.show_error("Failed to initialize LLM subscriber")
                return
        except Exception as e:
            self.show_error(f"LLM subscriber initialization failed: {e}")
            logging.error(f"LLM subscriber initialization failed: {e}")
            return

        try:
            print("Starting subscribers...")
            logging.info("Starting subscribers")
            self.camera_subscriber.start()
            self.llm_subscriber.start()
        except Exception as e:
            self.show_error(f"Failed to start subscribers: {e}")
            logging.error(f"Failed to start subscribers: {e}")
            return

        try:
            self.resize(620, 620)
            logging.info("MainWindow initialized successfully")
        except Exception as e:
            self.show_error(f"Failed to resize window: {e}")
            logging.error(f"Failed to resize window: {e}")
            return

    def on_frame_received(self, frame):
        self.processor.set_frame(frame)

    def on_velocity_received(self, velocity):
        pass  # No plot widget to update

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        central_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F5F7FA, stop:1 #E8ECEF);
                font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
                color: #343A40;
            }
        """)
        self.setStyleSheet("""
            QMainWindow {
                border: 1px solid #D3D8DE;
                border-radius: 10px;
            }
        """)

        sections_layout = QGridLayout()
        sections_layout.setSpacing(5)

        webcam_widget = HoverWidget()
        webcam_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #F9FAFB);
                border: 1px solid #CED4DA;
                border-radius: 5px;
            }
        """)
        webcam_widget.setMinimumSize(300, 260)
        webcam_layout = QVBoxLayout(webcam_widget)
        webcam_layout.setSpacing(2)
        webcam_layout.setContentsMargins(5, 5, 5, 5)
        webcam_title = QLabel("Object Detection(3D Bounding Boxes)")
        webcam_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529; text-align: center;")
        webcam_layout.addWidget(webcam_title)
        self.webcam_label = QLabel("Loading...")
        self.webcam_label.setMinimumSize(280, 220)
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setStyleSheet("color: #6C757D;")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(4)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.webcam_label.setGraphicsEffect(shadow)
        webcam_layout.addWidget(self.webcam_label, stretch=1)
        sections_layout.addWidget(webcam_widget, 0, 0)

        depth_widget = HoverWidget()
        depth_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #F9FAFB);
                border: 1px solid #CED4DA;
                border-radius: 5px;
            }
        """)
        depth_widget.setMinimumSize(300, 260)
        depth_layout = QVBoxLayout(depth_widget)
        depth_layout.setSpacing(2)
        depth_layout.setContentsMargins(5, 5, 5, 5)
        depth_title = QLabel("Depth Estimation (Colored Map)")
        depth_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529; text-align: center;")
        depth_layout.addWidget(depth_title)
        self.depth_label = QLabel("Loading...")
        self.depth_label.setMinimumSize(280, 220)
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_label.setStyleSheet("color: #6C757D;")
        depth_layout.addWidget(self.depth_label, stretch=1)
        sections_layout.addWidget(depth_widget, 0, 1)

        detection_widget = HoverWidget()
        detection_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #F9FAFB);
                border: 1px solid #CED4DA;
                border-radius: 5px;
            }
        """)
        detection_widget.setMinimumSize(300, 260)
        detection_layout = QVBoxLayout(detection_widget)
        detection_layout.setSpacing(2)
        detection_layout.setContentsMargins(5, 5, 5, 5)
        detection_title = QLabel("Object Detection(2D Bounding Boxes)")
        detection_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529; text-align: center;")
        detection_layout.addWidget(detection_title)
        self.detection_label = QLabel("Loading...")
        self.detection_label.setMinimumSize(280, 220)
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_label.setStyleSheet("color: #6C757D;")
        detection_layout.addWidget(self.detection_label, stretch=1)
        sections_layout.addWidget(detection_widget, 1, 0)

        bev_widget = HoverWidget()
        bev_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #F9FAFB);
                border: 1px solid #CED4DA;
                border-radius: 5px;
            }
        """)
        bev_widget.setMinimumSize(300, 260)
        bev_layout = QVBoxLayout(bev_widget)
        bev_layout.setSpacing(2)
        bev_layout.setContentsMargins(5, 5, 5, 5)
        bev_title = QLabel("Bird's Eye View")
        bev_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529; text-align: center;")
        bev_layout.addWidget(bev_title)
        self.bev_label = QLabel("Loading...")
        self.bev_label.setMinimumSize(280, 220)
        self.bev_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bev_label.setStyleSheet("color: #6C757D;")
        bev_layout.addWidget(self.bev_label, stretch=1)
        sections_layout.addWidget(bev_widget, 1, 1)

        main_layout.addLayout(sections_layout, stretch=1)

        controls_widget = QWidget()
        controls_widget.setStyleSheet("""
            QWidget {
                background: #FFFFFF;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px;
            }
        """)
        controls_widget.setFixedHeight(40)
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setSpacing(5)

        self.start_button = StartStopButton("Start")
        self.start_button.clicked.connect(self.toggle_processing)
        controls_layout.addWidget(self.start_button)

        main_layout.addWidget(controls_widget)

    def toggle_processing(self):
        if self.processor.running:
            self.processor.stop()
            self.start_button.setText("Start")
            self.start_button.setStyleSheet(self.start_button.start_style)
        else:
            self.processor.start()
            self.start_button.setText("Stop")
            self.start_button.setStyleSheet(self.start_button.stop_style)
        QApplication.processEvents()
        time.sleep(0.01)

    def update_frames(self, result_frame, depth_colored, detection_frame, bev_image):
        try:
            result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = result_rgb.shape
            bytes_per_line = ch * w
            image = QImage(result_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(self.webcam_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.webcam_label.setPixmap(pixmap)

            depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            h, w, ch = depth_rgb.shape
            bytes_per_line = ch * w
            image = QImage(depth_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(self.depth_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.depth_label.setPixmap(pixmap)

            detection_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = detection_rgb.shape
            bytes_per_line = ch * w
            image = QImage(detection_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(self.detection_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.detection_label.setPixmap(pixmap)

            bev_rgb = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
            h, w, ch = bev_rgb.shape
            bytes_per_line = ch * w
            image = QImage(bev_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(self.bev_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.bev_label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"Error updating frames: {e}")
            self.show_error(f"Error updating frames: {e}")

    def update_fps(self, fps_display):
        pass

    def show_error(self, message):
        print(f"Error: {message}")
        logging.error(message)

    def closeEvent(self, event):
        self.processor.stop()
        self.camera_subscriber.stop()
        self.llm_subscriber.stop()
        event.accept()

def main():
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error in main: {e}")
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        logging.info("Program interrupted by user (Ctrl+C)")