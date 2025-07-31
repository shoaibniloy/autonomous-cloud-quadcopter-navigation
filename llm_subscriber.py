#!/usr/bin/env python3
import sys
import zmq
import time
import json
import msgpack
import zlib
import threading
import langchain
from langchain_community.llms import Ollama
import logging
import os
import subprocess
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ModuleNotFoundError:
    logging.warning("keyboard module not found. Falling back to input-based termination.")
    KEYBOARD_AVAILABLE = False
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QPushButton, QSplitter
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas

# Set up logging
logging.basicConfig(filename='llm.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Logging initialized with DEBUG level.")

# Class mappings for YOLOv11 (COCO dataset)
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

# Sensor key mapping for descriptive names
SENSOR_KEY_MAPPING = {
    "Left Clearance": "Free space in the left direction (X)",
    "Right Clearance": "Free space in the right direction (-X)",
    "Front Clearance": "Free space in the front direction (-Y)",
    "Back Clearance": "Free space in the back direction (Y)",
    "Up Clearance": "Free space in the up direction (Z)",
    "Bottom Clearance": "Free space in the bottom direction (-Z)",
    "AccelX": "Acceleration in the X direction (m/s²)",
    "AccelY": "Acceleration in the Y direction (m/s²)",
    "AccelZ": "Acceleration in the Z direction (m/s²)",
    "GyroX": "Angular velocity around the X axis (rad/s)",
    "GyroY": "Angular velocity around the Y axis (rad/s)",
    "GyroZ": "Angular buckling around the Z axis (rad/s)"
}

# Updated context prompt with new data format
context_prompt = """ You are a drone navigating an environment. Your task is to decide your next movement based on the following inputs:

Input Data: A JSON object with the following structure:





"timestamp": Current time in ISO format.



"yolo": List of detected objects, each with:





"Name of the detected object": Name of the object (e.g., 'person', 'car')



"Distance of the object from you (in meters)": Distance to the object



"Dimensions of the object (in meters)": [width, height, length]



"orientation": Orientation of the object in radians



"Context or scene inference where the detected object is located": A description of the scene



"sensors": Dictionary with sensor readings:





"Free space in the left direction (X)": Distance in mm



"Free space in the right direction (-X)": Distance in mm



"Free space in the front direction (-Y)": Distance in mm



"Free space in the back direction (Y)": Distance in mm



"Free space in the up direction (Z)": Distance in mm



"Free space in the bottom direction (-Z)": Distance in mm



"Acceleration in the X direction (m/s²)": Acceleration



"Acceleration in the Y direction (m/s²)": Acceleration



"Acceleration in the Z direction (m/s²)": Acceleration



"Angular velocity around the X axis (rad/s)": Angular velocity



"Angular velocity around the Y axis (rad/s)": Angular velocity



"Angular buckling around the Z axis (rad/s)": Angular velocity

Golden Rule: No ToF sensor reading must ever be less than 300mm after the movement.

Movement Rules:





Yaw and vx: The drone yaws towards the direction with the most distance as indicated by ToF sensors (front, back, left, right) and then moves forward with positive vx. Yaw angles correspond to: front (-Y) = 0°, right (-X) = 90°, back (Y) = 180°, left (X) = 270°.



vy and vz: Used only for responsive obstacle avoidance. If a ToF reading < 300mm, set vy or vz to move away from that direction to restore clearance >= 300mm. vy adjusts left/right movement, vz adjusts up/down movement.

Decision Guidelines with Priorities:





Sensors (50% priority): Highest priority. Yaw towards the direction with the largest ToF reading among left, right, front, back, and set vx > 0. If any ToF < 300mm (e.g., left, right, up, bottom), adjust vy or vz to move away immediately. Use IMU data for stability.



YOLO (30% priority): Consider detected objects for context. Adjust path if objects are obstacles or targets, but only if ToF clearances allow it.



VLM (20% priority): Use scene description for situational awareness (e.g., 'narrow corridor' may suggest caution), influencing speed or yaw if sensor data permits.

Output Format: Provide your decision strictly in the following JSON format, with no additional text: { "output": { "vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.0 } }

Important Notes:





Velocities in cm/s, yaw in degrees.



Ensure all ToF readings remain >= 300mm after movement.



Main movement is via yaw and vx; vy and vz are for obstacle avoidance only.



IMU data ensures stable motion. """

# Initialize Ollama client
try:
    ollama_client = Ollama(model="deepseek-r1:1.5b", base_url="http://127.0.0.1:11434", timeout=5.0)
    logging.debug("Ollama client initialized with model deepseek-r1:1.5b, timeout=5.0s")
except Exception as e:
    logging.error(f"Failed to initialize Ollama client: {e}")
    print(f"Error: Failed to initialize Ollama client: {e}")
    exit(1)

# Set up ZMQ subscriber sockets
context = zmq.Context()
context.setsockopt(zmq.MAX_SOCKETS, 10)
poller = zmq.Poller()

yolo_sub = context.socket(zmq.SUB)
yolo_sub.connect("tcp://127.0.0.1:5555")
yolo_sub.setsockopt_string(zmq.SUBSCRIBE, "")
yolo_sub.setsockopt(zmq.RCVBUF, 1024 * 1024)
poller.register(yolo_sub, zmq.POLLIN)
logging.debug("YOLO subscriber connected to tcp://127.0.0.1:5555")

vlm_sub = context.socket(zmq.SUB)
vlm_sub.connect("tcp://127.0.0.1:5558")
vlm_sub.setsockopt_string(zmq.SUBSCRIBE, "")
vlm_sub.setsockopt(zmq.RCVBUF, 1024 * 1024)
poller.register(vlm_sub, zmq.POLLIN)
logging.debug("VLM subscriber connected to tcp://127.0.0.1:5558")

sensor_sub = context.socket(zmq.SUB)
sensor_sub.connect("tcp://127.0.0.1:5550")
sensor_sub.setsockopt_string(zmq.SUBSCRIBE, "")
sensor_sub.setsockopt(zmq.RCVBUF, 1024 * 1024)
poller.register(sensor_sub, zmq.POLLIN)
logging.debug("Sensor subscriber connected to tcp://127.0.0.1:5550")

# Set up ZMQ acknowledgment publisher
ack_context = zmq.Context()
ack_context.setsockopt(zmq.MAX_SOCKETS, 10)
ack_pub = ack_context.socket(zmq.PUB)
ack_pub.setsockopt(zmq.LINGER, 0)
ack_pub.setsockopt(zmq.SNDBUF, 512 * 1024)
try:
    ack_pub.bind("tcp://127.0.0.1:5557")
    logging.debug("Acknowledgment publisher bound to tcp://127.0.0.1:5557")
except zmq.error.ZMQError as e:
    logging.error(f"Failed to bind acknowledgment socket: {e}")
    print(f"Error: Failed to bind acknowledgment socket: {e}")
    exit(1)

# Initialize LLM with context prompt
try:
    logging.debug(f"Sending context prompt to LLM: {context_prompt[:100]}...")
    ollama_client.invoke(context_prompt)
    logging.debug("Context prompt sent to LLM")
    time.sleep(0.1)
    ack_pub.send_string("acknowledged")
    logging.debug("Acknowledgment sent")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}")
    print(f"Error: Failed to initialize LLM: {e}")
    exit(1)

# Event for shutdown
stop_event = threading.Event()

# GUI class
class LLMResponseWindow(QMainWindow):
    update_response = pyqtSignal(str)
    update_log = pyqtSignal(str)
    update_velocity = pyqtSignal(str)
    update_plot_signal = pyqtSignal(float)

    def __init__(self, model_name):
        super().__init__()
        self.setWindowTitle(f"{model_name} MAVLink Responses")
        self.setMinimumSize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        central_widget.setStyleSheet("background: #F5F7FA; font-family: 'Arial', sans-serif; color: #343A40;")
        self.setStyleSheet("QMainWindow { border: 1px solid #D3D8DE; }")

        # Horizontal splitter for text widgets
        text_splitter = QSplitter(Qt.Orientation.Horizontal)
        text_splitter.setHandleWidth(5)
        text_splitter.setStyleSheet("QSplitter::handle { background: #CED4DA; }")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background: #FFFFFF; border: 1px solid #CED4DA; padding: 5px; font-size: 14px; color: #343A40;")
        self.log_text.setMinimumWidth(350)
        text_splitter.addWidget(self.log_text)

        self.velocity_text = QTextEdit()
        self.velocity_text.setReadOnly(True)
        self.velocity_text.setStyleSheet("background: #FFFFFF; border: 1px solid #CED4DA; padding: 5px; font-size: 14px; color: #343A40;")
        self.velocity_text.setMinimumWidth(350)
        text_splitter.addWidget(self.velocity_text)

        # Vertical splitter for text and plot
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setHandleWidth(5)
        main_splitter.setStyleSheet("QSplitter::handle { background: #CED4DA; }")
        main_splitter.addWidget(text_splitter)

        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("LLM Response Times", fontsize=14)
        self.ax.set_xlabel("Response Number", fontsize=12)
        self.ax.set_ylabel("Time (s)", fontsize=12)
        self.ax.tick_params(axis='both', labelsize=10)
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.2)
        self.response_times = []
        self.line, = self.ax.plot([], [], 'r-')
        main_splitter.addWidget(self.canvas)

        main_layout.addWidget(main_splitter)

        log_button = QPushButton("View Drone Perception Logs")
        log_button.setStyleSheet("background: #007BFF; border: 1px solid #CED4DA; padding: 5px; font-size: 13px; color: #FFFFFF; QPushButton:pressed { background: #0056b3; }")
        log_button.clicked.connect(self.open_log_file)
        main_layout.addWidget(log_button)

        clear_log_button = QPushButton("Clear Drone Perception Logs")
        clear_log_button.setStyleSheet("background: #DC3545; border: 1px solid #CED4DA; padding: 5px; font-size: 13px; color: #FFFFFF; QPushButton:pressed { background: #b02a37; }")
        clear_log_button.clicked.connect(self.clear_log_file)
        main_layout.addWidget(clear_log_button)

        self.update_response.connect(self.append_response)
        self.update_log.connect(self.append_log)
        self.update_velocity.connect(self.append_velocity)
        self.update_plot_signal.connect(self.update_plot)

        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.check_log_file)
        self.log_timer.start(500)

        self.log_file_position = 0
        self.last_log_lines = []

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.start(200)

    def append_response(self, response):
        self.update_velocity.emit(response)

    def append_log(self, log_line):
        if self.log_text.document().blockCount() > 1000:
            self.log_text.clear()
        self.log_text.append(log_line)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def append_velocity(self, response):
        if "Action: Set velocity" in response:
            if self.velocity_text.document().blockCount() > 1000:
                self.velocity_text.clear()
            timestamp = time.strftime("%H:%M:%S")
            action_line = response.split("Action: ")[1] if "Action: " in response else response
            self.velocity_text.append(f"[{timestamp}] {action_line}")
            self.velocity_text.verticalScrollBar().setValue(self.velocity_text.verticalScrollBar().maximum())

    def update_plot(self, response_time):
        self.response_times.append(response_time)
        self.line.set_data(range(len(self.response_times)), self.response_times)
        self.ax.relim()
        self.ax.autoscale_view()
        # Calculate and display average response time
        avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
        # Remove previous text annotations to avoid overlap
        for text in self.ax.texts:
            text.remove()
        self.ax.text(0.05, 0.95, f'Avg Response Time: {avg_time:.2f}s', transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        self.canvas.draw()

    def check_log_file(self):
        log_file = os.path.join(os.getcwd(), "llm.log")
        if not os.path.exists(log_file):
            return
        try:
            with open(log_file, 'r') as f:
                f.seek(self.log_file_position)
                new_lines = f.readlines()
                self.log_file_position = f.tell()
                for line in new_lines:
                    line = line.strip()
                    if line:
                        self.update_log.emit(line)
                        if "Action: Set velocity" in line:
                            timestamp = line.split(" - ")[0] if " - " in line else time.strftime("%H:%M:%S")
                            action = line.split("Action: ")[1] if "Action: " in line else line
                            self.update_velocity.emit(f"[{timestamp}] Action: {action}")
        except Exception as e:
            logging.error(f"Error reading llm.log: {e}")

    def open_log_file(self):
        log_file = os.path.join(os.getcwd(), "llm.log")
        if os.path.exists(log_file):
            try:
                if os.name == 'nt':
                    os.startfile(log_file)
                elif os.name == 'posix':
                    subprocess.run(['xdg-open', log_file], check=True)
                else:
                    logging.error("Unsupported OS for opening log file")
            except Exception as e:
                logging.error(f"Failed to open llm.log: {e}")
        else:
            logging.error("llm.log file not found")

    def clear_log_file(self):
        log_file = os.path.join(os.getcwd(), "llm.log")
        try:
            with open(log_file, 'w') as f:
                f.truncate(0)
            self.log_file_position = 0
            self.log_text.clear()
            self.velocity_text.clear()
            logging.debug("Cleared llm.log file")
            self.update_log.emit("Log file (llm.log) cleared successfully.")
        except Exception as e:
            logging.error(f"Failed to clear llm.log: {e}")
            self.update_log.emit(f"Error: Failed to clear llm.log: {e}")

# Termination check
def check_for_quit():
    if KEYBOARD_AVAILABLE:
        keyboard.wait('q')
        logging.debug("Received 'q' keypress for termination")
    else:
        input("Press Enter to stop the LLM script...\n")
        logging.debug("Received Enter key for termination")
    stop_event.set()

quit_thread = threading.Thread(target=check_for_quit)
quit_thread.daemon = True
quit_thread.start()

# Map YOLO detections to new format
def map_detections_to_labels(detections):
    labeled_detections = []
    for det in detections:
        if isinstance(det, dict) and 'class_name' in det:
            label = det['class_name'].lower()
            depth = det.get('depth_value', None)
            dimensions = det.get('dimensions', [0.0, 0.0, 0.0])
            orientation = det.get('orientation', 0.0)
            
            det_data = {
                "Name of the detected object": label,
                "Distance of the object from you (in meters)": depth if depth is not None else 1.0,
                "Dimensions of the object (in meters)": dimensions,
                "orientation": orientation
            }
            labeled_detections.append(det_data)
            logging.debug(f"Mapped YOLO detection: {det_data}")
        else:
            logging.warning(f"Invalid detection dictionary format: {det}")
    return labeled_detections

# Parse LLM output and enforce priorities
def format_llm_output(response, yolo_detections, vlm_description, sensor_data):
    try:
        data = json.loads(response)
        output = data.get("output", {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.0})
        vx, vy, vz, yaw = output["vx"], output["vy"], output["vz"], output["yaw"]
        desc = "Description: LLM provided velocities based on input data."
        formatted = f"{desc}\nAction: Set velocity vx={vx}, vy={vy}, vz={vz}, yaw={yaw}"
        logging.debug(f"Parsed JSON response: {formatted}")
        return formatted
    except json.JSONDecodeError:
        logging.warning("LLM response is not valid JSON. Falling back to priority-based decision-making.")
        vx, vy, vz, yaw = 0.0, 0.0, 0.0, 0.0
        desc = "Description: Fallback decision based on priorities."

        # Sensors (50% priority)
        if sensor_data:
            tof_sensors = ["Left Clearance", "Right Clearance", "Front Clearance", "Back Clearance", "Up Clearance", "Bottom Clearance"]
            for sensor in tof_sensors:
                distance = sensor_data.get(sensor)
                if distance is not None and distance < 300:
                    desc = f"Description: Obstacle too close {sensor.lower()} at {distance}mm."
                    if sensor == "Left Clearance":
                        vx = 0.5
                    elif sensor == "Right Clearance":
                        vx = -0.5
                    elif sensor == "Front Clearance":
                        vy = -0.5
                    elif sensor == "Back Clearance":
                        vy = 0.5
                    elif sensor == "Up Clearance":
                        vz = -0.5
                    elif sensor == "Bottom Clearance":
                        vz = 0.5
                    formatted = f"{desc}\nAction: Set velocity vx={vx}, vy={vy}, vz={vz}, yaw={yaw}"
                    logging.debug(f"Fallback sensor response: {formatted}")
                    return formatted

        # YOLO (30% priority)
        if yolo_detections and (vx == 0.0 and vy == 0.0 and vz == 0.0):
            closest = min(yolo_detections, key=lambda x: x["Distance of the object from you (in meters)"] if x["Distance of the object from you (in meters)"] is not None else float('inf'))
            depth = closest["Distance of the object from you (in meters)"] if closest["Distance of the object from you (in meters)"] is not None else 1.0
            label = closest["Name of the detected object"]
            desc = f"Description: {label.capitalize()} at {depth:.1f}m away."
            if depth < 0.6:
                vy = -0.5  # Move back to avoid

        # VLM (20% priority)
        if vlm_description and (vx == 0.0 and vy == 0.0 and vz == 0.0):
            desc = f"Description: VLM context - {vlm_description[:20]}..." if len(vlm_description) > 20 else f"Description: {vlm_description}"
            if "crowded" in vlm_description.lower():
                desc += " Moving cautiously."
                vx = 0.2
            elif "open" in vlm_description.lower():
                desc += " Moving freely."
                vy = 0.5

        if vx == 0.0 and vy == 0.0 and vz == 0.0:
            desc = "Description: No immediate action required."
        
        formatted = f"{desc}\nAction: Set velocity vx={vx}, vy={vy}, vz={vz}, yaw={yaw}"
        logging.debug(f"Fallback response: {formatted}")
        return formatted

# Main loop with GUI
def main():
    app = QApplication(sys.argv)
    window = LLMResponseWindow(model_name="gemma3:1b")

    def processing_loop():
        try:
            logging.debug("Starting processing loop")
            last_data_time = time.time()
            timeout = 30
            latest_yolo = None
            latest_vlm = None
            latest_sensors = None
            
            # Initialization phase: Wait for all ZMQ sockets to receive initial data
            yolo_received = False
            vlm_received = False
            sensors_received = False
            init_start_time = time.time()
            init_timeout = 30  # seconds

            while not (yolo_received and vlm_received and sensors_received) and not stop_event.is_set():
                if time.time() - init_start_time > init_timeout:
                    logging.error("Initialization timeout: Not all data sources received initial data within 30 seconds.")
                    break
                events = dict(poller.poll(200))
                if yolo_sub in events:
                    try:
                        compressed = yolo_sub.recv(zmq.NOBLOCK)
                        packed = zlib.decompress(compressed)
                        yolo_data = msgpack.unpackb(packed, raw=False)
                        latest_yolo = yolo_data
                        if not yolo_received:
                            yolo_received = True
                            logging.info("Received first YOLO data")
                    except Exception as e:
                        logging.error(f"Error receiving YOLO data: {e}")
                if vlm_sub in events:
                    try:
                        packed = vlm_sub.recv(zmq.NOBLOCK)
                        vlm_data = msgpack.unpackb(packed, raw=False)
                        latest_vlm = vlm_data
                        if not vlm_received:
                            vlm_received = True
                            logging.info("Received first VLM data")
                    except Exception as e:
                        logging.error(f"Error receiving VLM data: {e}")
                if sensor_sub in events:
                    try:
                        compressed = sensor_sub.recv(zmq.NOBLOCK)
                        packed = zlib.decompress(compressed)
                        sensor_data = msgpack.unpackb(packed, raw=False)
                        latest_sensors = sensor_data
                        if not sensors_received:
                            sensors_received = True
                            logging.info("Received first sensor data")
                    except Exception as e:
                        logging.error(f"Error receiving sensor data: {e}")
                time.sleep(0.1)

            if not (yolo_received and vlm_received and sensors_received):
                logging.warning("Proceeding with available data sources.")
            else:
                logging.info("All data sources initialized successfully.")

            # Main processing loop
            while not stop_event.is_set():
                events = dict(poller.poll(200))
                
                if yolo_sub in events:
                    try:
                        compressed = yolo_sub.recv(zmq.NOBLOCK)
                        packed = zlib.decompress(compressed)
                        yolo_data = msgpack.unpackb(packed, raw=False)
                        latest_yolo = yolo_data
                        last_data_time = time.time()
                        logging.debug(f"Received YOLO data: {json.dumps(yolo_data, indent=None)}")
                    except Exception as e:
                        logging.error(f"Error receiving YOLO data: {e}")
                
                if vlm_sub in events:
                    try:
                        packed = vlm_sub.recv(zmq.NOBLOCK)
                        vlm_data = msgpack.unpackb(packed, raw=False)
                        latest_vlm = vlm_data
                        last_data_time = time.time()
                        logging.debug(f"Received VLM data: {json.dumps(vlm_data, indent=None)}")
                    except Exception as e:
                        logging.error(f"Error receiving VLM data: {e}")
                
                if sensor_sub in events:
                    try:
                        compressed = sensor_sub.recv(zmq.NOBLOCK)
                        packed = zlib.decompress(compressed)
                        sensor_data = msgpack.unpackb(packed, raw=False)
                        latest_sensors = sensor_data
                        last_data_time = time.time()
                        logging.debug(f"Received sensor data: {json.dumps(sensor_data, indent=None)}")
                    except Exception as e:
                        logging.error(f"Error receiving sensor data: {e}")
                
                if latest_yolo or latest_vlm or latest_sensors:
                    yolo_detections = map_detections_to_labels(latest_yolo.get('boxes_3d', []) if latest_yolo else [])
                    vlm_description = latest_vlm.get('description', '') if latest_vlm else ''
                    sensor_data = latest_sensors if latest_sensors else {}
                    
                    # Apply sensor key mapping
                    modified_sensors = {SENSOR_KEY_MAPPING.get(key, key): value for key, value in sensor_data.items()}
                    
                    analysis_data = {
                        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
                        'yolo': yolo_detections,
                        "Context or scene inference where the detected object is located": vlm_description,
                        'sensors': modified_sensors
                    }
                    logging.debug(f"Prepared LLM input: {json.dumps(analysis_data, indent=None)}")
                    
                    try:
                        start_time = time.time()
                        full_prompt = context_prompt + "\n\n**Input Data**:\n" + json.dumps(analysis_data, indent=2)
                        response = ollama_client.invoke(full_prompt)
                        end_time = time.time()
                        response_time = end_time - start_time
                        logging.debug(f"LLM inference completed in {response_time:.2f} seconds")
                        with open('llm_latency.log', 'w') as f:
                            f.write(f"{response_time}\n")
                        formatted_response = format_llm_output(response, yolo_detections, vlm_description, sensor_data)
                        window.update_response.emit(f"[LLM Response at {time.strftime('%H:%M:%S')}]: {formatted_response}")
                        window.update_plot_signal.emit(response_time)
                        with open("responses.jsonl", "a") as f:
                            f.write(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()), "response": formatted_response}) + "\n")
                        logging.debug(f"LLM response written: {formatted_response}")
                    except Exception as e:
                        logging.error(f"LLM invocation failed: {e}")
                        response = format_llm_output("", yolo_detections, vlm_description, sensor_data)
                        window.update_response.emit(f"[LLM Response at {time.strftime('%H:%M:%S')}]: {response}")
                        with open("responses.jsonl", "a") as f:
                            f.write(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()), "response": response}) + "\n")
                        logging.debug(f"Fallback response written: {response}")
                
                if time.time() - last_data_time > timeout:
                    logging.error("No data received for 30 seconds. Exiting.")
                    window.update_response.emit("Error: No data received for 30 seconds. Exiting.")
                    break
                time.sleep(0.2)

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            window.update_response.emit(f"Error in main loop: {e}")
        finally:
            yolo_sub.close()
            vlm_sub.close()
            sensor_sub.close()
            context.term()
            ack_pub.close()
            ack_context.term()
            logging.debug("Processing loop terminated, sockets closed")

    processing_thread = threading.Thread(target=processing_loop)
    processing_thread.daemon = True
    processing_thread.start()

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        logging.debug("Program interrupted by user (Ctrl+C)")