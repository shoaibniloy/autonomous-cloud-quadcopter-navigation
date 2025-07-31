import subprocess
import time
import json
import os
import logging
import sys
import cv2
import requests
import numpy as np
from threading import Thread, Lock
import base64
from io import BytesIO
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse
import psutil
import zmq
import msgpack
import zlib
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTextEdit, QComboBox, QPushButton, QGraphicsDropShadowEffect
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt6.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, QThread, pyqtSignal

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask system monitoring script (unchanged)
FLASK_SCRIPT = """from flask import Flask, jsonify
import psutil
import subprocess
import platform
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage

def get_ram_usage():
    mem = psutil.virtual_memory()
    return mem.percent

def get_gpu_usage():
    if platform.system() == 'Linux':
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            gpu_usage = result.stdout.decode('utf-8').strip()
            if gpu_usage:
                return gpu_usage
            else:
                return "GPU not in use"
        except FileNotFoundError:
            return "No NVIDIA GPU or nvidia-smi not installed"
    else:
        return "Not a Linux system or no GPU"

@app.route('/system-usage', methods=['GET'])
def system_usage():
    cpu_usage = get_cpu_usage()
    ram_usage = get_ram_usage()
    gpu_usage = get_gpu_usage()

    return jsonify({
        'cpu_usage': cpu_usage,
        'ram_usage': ram_usage,
        'gpu_usage': gpu_usage
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
"""

def save_flask_script():
    """Save the Flask script to a file."""
    logger.debug("Saving Flask script...")
    flask_file_path = os.path.join(os.getcwd(), "system_monitor.py")
    try:
        with open(flask_file_path, "w") as f:
            f.write(FLASK_SCRIPT)
        logger.info(f"Flask script saved to {flask_file_path}")
        return flask_file_path
    except Exception as e:
        logger.error(f"Error saving Flask script: {e}")
        raise

def start_vlm_server():
    """Start the VLM server in a new terminal."""
    logger.debug("Starting VLM server...")
    try:
        vlm_command = (
            "cd ~/Drone/YOLO+LLM/llama.cpp/build && "
            "./bin/llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -ngl 99"
        )
        subprocess.Popen([
            "xterm", "-hold", "-e", f"bash -c '{vlm_command}'"
        ])
        time.sleep(10)  # Wait for VLM server initialization
        logger.info("VLM server terminal opened.")
    except Exception as e:
        logger.error(f"Error starting VLM server: {e}")
        raise

def start_flask_server(flask_file_path):
    """Start the Flask system monitoring server in a new terminal."""
    logger.debug("Starting Flask server...")
    try:
        flask_command = f"python3 {flask_file_path}"
        subprocess.Popen([
            "xterm", "-hold", "-e", f"bash -c 'source ~/.bashrc && {flask_command}'"
        ])
        time.sleep(2)  # Wait for the Flask server to start
        logger.info("Flask server terminal opened.")
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}")
        raise

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
            time.sleep(1)  # Allow connection to stabilize
            logging.info(f"ZMQ subscriber connected to tcp://localhost:{self.port}")
            return True
        except zmq.error.ZMQError as e:
            self.error_occurred.emit(f"Failed to set up ZMQ subscriber: {e}")
            return False

    def run(self):
        self.running = True
        while self.running:
            try:
                frame_data = self.socket.recv()
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self.frame_received.emit(frame)
                time.sleep(0.033)  # ~30 FPS
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

class YoloSubscriber(QThread):
    """Thread to receive YOLO data from ZMQ."""
    yolo_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, port="5555"):
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
            time.sleep(1)  # Allow connection to stabilize
            logging.info(f"ZMQ YOLO subscriber connected to tcp://localhost:{self.port}")
            return True
        except zmq.error.ZMQError as e:
            self.error_occurred.emit(f"Failed to set up ZMQ YOLO subscriber: {e}")
            return False

    def run(self):
        self.running = True
        while self.running:
            try:
                compressed = self.socket.recv()
                packed = zlib.decompress(compressed)
                yolo_data = msgpack.unpackb(packed, raw=False)
                self.yolo_received.emit(yolo_data)
                time.sleep(0.033)  # ~30 FPS
            except zmq.error.ZMQError as e:
                logging.error(f"Error receiving ZMQ YOLO data: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        logging.info("YoloSubscriber stopped")

class VLMClient:
    """Manage VLM API interactions with retries."""
    def __init__(self, endpoint, instruction="You are a drone, decide what you need to do now"):
        logger.debug(f"Initializing VLM client with endpoint: {endpoint}")
        self.endpoint = endpoint
        self.instruction = instruction
        self.latest_response = "Waiting for VLM response..."
        self.lock = Lock()
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def process_frame(self, frame):
        """Send frame to VLM and get response with latency logging."""
        try:
            logger.debug("Encoding frame for VLM...")
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{img_base64}"
            payload = {
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.instruction},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ]
            }
            logger.debug(f"Sending frame to VLM at {self.endpoint}")
            start_time = time.time()  # Start timing
            response = self.session.post(self.endpoint, json=payload, timeout=3)
            end_time = time.time()  # End timing
            latency = end_time - start_time
            if response.status_code == 200:
                logger.debug(f"VLM inference latency: {latency:.4f} seconds")
                data = response.json()
                vlm_text = data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            else:
                logger.debug(f"VLM inference failed with HTTP {response.status_code} in {latency:.4f} seconds")
                vlm_text = f"Error: HTTP {response.status_code}"
        except requests.RequestException as e:
            end_time = time.time()  # End timing on exception
            latency = end_time - start_time
            logger.debug(f"VLM inference failed after {latency:.4f} seconds: {e}")
            vlm_text = f"Error: {e}"
        with self.lock:
            self.latest_response = vlm_text
            logger.debug(f"VLM response: {vlm_text}")

    def get_response(self):
        """Get the latest VLM response."""
        with self.lock:
            return self.latest_response

class SystemUsageWorker(QThread):
    """Worker thread to fetch system usage data asynchronously."""
    usage_updated = pyqtSignal(dict, str, str)  # Signal to emit usage data, error code, error message

    def run(self):
        try:
            response = requests.get("http://localhost:5000/system-usage", timeout=2)
            if response.status_code == 200:
                data = response.json()
                self.usage_updated.emit(data, "", "")
            else:
                self.usage_updated.emit({}, str(response.status_code), f"HTTP {response.status_code}")
        except requests.RequestException as e:
            self.usage_updated.emit({}, "RequestException", str(e))

class HoverWidget(QWidget):
    """Custom widget to handle hover effects."""
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
    """Custom button with pressed state styling."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.normal_style = """
            QPushButton {
                background: #FFFFFF;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px 10px;
                font-size: 14px;
                color: #343A40;
            }
        """
        self.pressed_style = """
            QPushButton {
                background: #E8ECEF;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px 10px;
                font-size: 14px;
                color: #343A40;
            }
        """
        self.start_style = """
            QPushButton {
                background: #28A745;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px 10px;
                font-size: 14px;
                color: #FFFFFF;
            }
        """
        self.stop_style = """
            QPushButton {
                background: #DC3545;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px 10px;
                font-size: 14px;
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
    def __init__(self, vlm_endpoint, instruction, inference_interval):
        super().__init__()
        self.setWindowTitle("High Level Scenic Understanding")
        self.setMinimumSize(900, 400)  # Minimum size to ensure usability

        # Initialize camera subscriber
        self.camera_subscriber = CameraSubscriber(port="5556")
        self.camera_subscriber.frame_received.connect(self.on_frame_received)
        self.camera_subscriber.error_occurred.connect(self.show_error)

        # Initialize YOLO subscriber
        self.yolo_subscriber = YoloSubscriber(port="5555")
        self.yolo_subscriber.yolo_received.connect(self.on_yolo_received)
        self.yolo_subscriber.error_occurred.connect(self.show_error)

        # Set up ZMQ publisher for sending VLM responses
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        try:
            self.zmq_socket.bind("tcp://127.0.0.1:5558")
            logging.info("ZMQ publisher for VLM responses bound to tcp://127.0.0.1:5558")
        except zmq.error.ZMQError as e:
            logger.error(f"Failed to bind ZMQ publisher for VLM responses: {e}")
            raise

        # Initialize VLM client
        self.vlm_client = VLMClient(vlm_endpoint, instruction)
        self.inference_interval = inference_interval
        self.is_running = False  # Track processing state
        self.frame = None
        self.latest_yolo = None

        # Setup UI
        self.setup_ui()

        # Setup timers for updates (initially stopped)
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feed)
        self.camera_timer.setInterval(50)  # Reduced to ~20 FPS for smoother performance

        self.vlm_timer = QTimer()
        self.vlm_timer.timeout.connect(self.update_vlm_response)
        self.vlm_timer.setInterval(inference_interval)

        self.sys_usage_timer = QTimer()
        self.sys_usage_timer.timeout.connect(self.start_system_usage_update)
        self.sys_usage_timer.setInterval(1000)  # Every 1 second

        # Setup system usage worker thread
        self.sys_usage_worker = SystemUsageWorker()
        self.sys_usage_worker.usage_updated.connect(self.update_system_usage)

        # Start camera and YOLO subscribers
        if not self.camera_subscriber.initialize():
            self.show_error("Failed to initialize camera subscriber")
            return

        if not self.yolo_subscriber.initialize():
            self.show_error("Failed to initialize YOLO subscriber")
            return

        self.camera_subscriber.start()
        self.yolo_subscriber.start()

        # Set initial size after UI setup
        self.resize(900, 400)

    def on_frame_received(self, frame):
        self.frame = frame

    def on_yolo_received(self, yolo_data):
        self.latest_yolo = yolo_data

    def setup_ui(self):
        # Central widget and main layout (vertical to include controls at the bottom)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Apply gradient background to the main window
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

        # Sections layout (horizontal)
        sections_layout = QHBoxLayout()
        sections_layout.setSpacing(5)
        sections_layout.setStretch(0, 1)  # Camera feed section stretches
        sections_layout.setStretch(1, 0)  # VLM response section fixed
        sections_layout.setStretch(2, 0)  # System usage section fixed

        # Camera Feed Section
        camera_widget = HoverWidget()
        camera_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #F9FAFB);
                border: 1px solid #CED4DA;
                border-radius: 5px;
            }
        """)
        camera_widget.setMinimumSize(360, 290)  # Increased initial size
        camera_layout = QVBoxLayout(camera_widget)
        camera_layout.setSpacing(2)
        camera_layout.setContentsMargins(5, 5, 5, 5)
        camera_title = QLabel("Camera Feed")
        camera_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529; text-align: center;")
        camera_layout.addWidget(camera_title)
        self.camera_label = QLabel("Loading...")
        self.camera_label.setMinimumSize(340, 255)  # Increased initial size, 4:3 ratio
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("color: #6C757D; font-size: 16px;")
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(4)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.camera_label.setGraphicsEffect(shadow)
        camera_layout.addWidget(self.camera_label, stretch=1)
        sections_layout.addWidget(camera_widget)

        # VLM Response Section
        vlm_widget = HoverWidget()
        vlm_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #F9FAFB);
                border: 1px solid #CED4DA;
                border-radius: 5px;
            }
        """)
        vlm_widget.setFixedSize(260, 220)
        vlm_layout = QVBoxLayout(vlm_widget)
        vlm_layout.setSpacing(2)
        vlm_layout.setContentsMargins(5, 5, 5, 5)
        vlm_title = QLabel("VLM Response")
        vlm_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529; text-align: center;")
        vlm_layout.addWidget(vlm_title)
        self.vlm_text = QTextEdit()
        self.vlm_text.setReadOnly(True)
        self.vlm_text.setFixedSize(240, 180)
        self.vlm_text.setText("Waiting for VLM response...")
        self.vlm_text.setStyleSheet("""
            QTextEdit {
                background: transparent;
                border: none;
                color: #343A40;
                font-size: 18px;
                padding: 3px;
            }
            QTextEdit QScrollBar:vertical {
                border: none;
                background: #F9FAFB;
                width: 6px;
                margin: 0px;
            }
            QTextEdit QScrollBar::handle:vertical {
                background: #CED4DA;
                border-radius: 3px;
            }
            QTextEdit QScrollBar::add-line:vertical, QTextEdit QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        vlm_layout.addWidget(self.vlm_text)
        sections_layout.addWidget(vlm_widget)

        # System Usage Section
        sys_usage_widget = HoverWidget()
        sys_usage_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #F9FAFB);
                border: 1px solid #CED4DA;
                border-radius: 5px;
            }
        """)
        sys_usage_widget.setFixedSize(260, 220)
        sys_usage_layout = QVBoxLayout(sys_usage_widget)
        sys_usage_layout.setSpacing(2)
        sys_usage_layout.setContentsMargins(5, 5, 5, 5)
        sys_usage_title = QLabel("System Usage")
        sys_usage_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529; text-align: center;")
        sys_usage_layout.addWidget(sys_usage_title)
        self.cpu_label = QLabel("CPU Usage: -- %")
        self.cpu_label.setStyleSheet("font-size: 16px; color: #343A40; margin-left: 5px;")
        self.cpu_progress = QWidget()
        self.cpu_progress.setFixedHeight(2)
        self.cpu_progress.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #28A745, stop:1 #34C759);
        """)  # Green gradient
        self.ram_label = QLabel("RAM Usage: -- %")
        self.ram_label.setStyleSheet("font-size: 16px; color: #343A40; margin-left: 5px;")
        self.ram_progress = QWidget()
        self.ram_progress.setFixedHeight(2)
        self.ram_progress.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007BFF, stop:1 #66B0FF);
        """)  # Blue gradient
        self.gpu_label = QLabel("GPU Usage: --")
        self.gpu_label.setStyleSheet("font-size: 16px; color: #343A40; margin-left: 5px;")
        self.gpu_progress = QWidget()
        self.gpu_progress.setFixedHeight(2)
        self.gpu_progress.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #DC3545, stop:1 #FF6B6B);
        """)  # Red gradient
        sys_usage_layout.addWidget(self.cpu_label)
        sys_usage_layout.addWidget(self.cpu_progress)
        sys_usage_layout.addWidget(self.ram_label)
        sys_usage_layout.addWidget(self.ram_progress)
        sys_usage_layout.addWidget(self.gpu_label)
        sys_usage_layout.addWidget(self.gpu_progress)
        sections_layout.addWidget(sys_usage_widget)

        # Add sections layout to main layout
        main_layout.addLayout(sections_layout, stretch=1)

        # Controls Section
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

        # Inference Speed Control
        interval_label = QLabel("Inference Speed:")
        interval_label.setStyleSheet("font-size: 14px; color: #343A40;")
        controls_layout.addWidget(interval_label)

        self.interval_select = QComboBox()
        self.interval_select.addItem("Flash (100ms)", 100)
        self.interval_select.addItem("Cheetah (250ms)", 250)
        self.interval_select.addItem("Ideal (500ms)", 500)
        self.interval_select.addItem("Normal (1000ms)", 1000)
        self.interval_select.addItem("Lethargic (1500ms)", 1500)
        self.interval_select.addItem("Sloth (2000ms)", 2000)
        self.interval_select.setCurrentIndex(2)  # Default to Ideal (500ms)
        self.interval_select.currentIndexChanged.connect(self.update_inference_interval)
        self.interval_select.setStyleSheet("""
            QComboBox {
                background: #FFFFFF;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px;
                font-size: 14px;
                color: #343A40;
            }
            QComboBox QAbstractItemView {
                background: #FFFFFF;
                border: 1px solid #CED4DA;
                selection-background-color: #2ECC71;
                color: #343A40;
                font-size: 14px;
            }
        """)
        controls_layout.addWidget(self.interval_select)

        # Start/Stop Button
        self.start_button = StartStopButton("Start")
        self.start_button.clicked.connect(self.toggle_processing)
        controls_layout.addWidget(self.start_button)

        main_layout.addWidget(controls_widget)

    def update_inference_interval(self):
        """Update the VLM timer interval based on the selected inference speed."""
        interval = self.interval_select.currentData()
        self.inference_interval = interval
        self.vlm_timer.setInterval(interval)
        logger.debug(f"Inference interval updated to {interval}ms")

    def toggle_processing(self):
        """Toggle between starting and stopping the processing."""
        if self.is_running:
            self.is_running = False
            self.camera_timer.stop()
            self.vlm_timer.stop()
            self.sys_usage_timer.stop()
            self.start_button.setText("Start")
            self.start_button.setStyleSheet(self.start_button.start_style)
            logger.debug("Processing stopped")
        else:
            self.is_running = True
            self.camera_timer.start()
            self.vlm_timer.start()
            self.sys_usage_timer.start()
            self.start_button.setText("Stop")
            self.start_button.setStyleSheet(self.start_button.stop_style)
            logger.debug("Processing started")
        # Force GUI update to ensure responsiveness
        QApplication.processEvents()
        time.sleep(0.01)  # Brief delay to allow GUI refresh

    def update_camera_feed(self):
        if not self.is_running:
            return
        if self.frame is None:
            return
        frame = self.frame.copy()
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_label.setPixmap(pixmap)

    def update_vlm_response(self):
        if not self.is_running:
            return
        if self.frame is None:
            return
        frame = self.frame.copy()
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        thread = Thread(target=self.vlm_client.process_frame, args=(frame,), daemon=True)
        thread.start()
        vlm_response = self.vlm_client.get_response()
        logger.debug(f"VLM response: {vlm_response}")
        self.vlm_text.setText(vlm_response)

        # Prepare combined data with YOLO and VLM
        combined_data = {
            "description": vlm_response
        }
        if self.latest_yolo:
            combined_data.update(self.latest_yolo)

        # Send combined data via ZMQ
        try:
            packed = msgpack.packb(combined_data, use_bin_type=True)
            self.zmq_socket.send(packed)
            logger.debug(f"Sent combined VLM and YOLO data: {combined_data}")
        except zmq.error.ZMQError as e:
            logger.error(f"Error sending combined VLM and YOLO data via ZMQ: {e}")

    def start_system_usage_update(self):
        if not self.is_running:
            return
        # Start the worker thread to fetch system usage data
        self.sys_usage_worker.start()

    def update_system_usage(self, data, error_code, error_message):
        if not self.is_running:
            return
        if error_code:
            logger.error(f"System usage error: {error_code} - {error_message}")
            self.cpu_label.setText("CPU Usage: Error")
            self.ram_label.setText("RAM Usage: Error")
            self.gpu_label.setText("GPU Usage: Error")
            self.cpu_progress.setFixedWidth(0)
            self.ram_progress.setFixedWidth(0)
            self.gpu_progress.setFixedWidth(0)
        else:
            logger.debug(f"System usage: CPU {data['cpu_usage']}%, RAM {data['ram_usage']}%, GPU {data['gpu_usage']}")
            self.cpu_label.setText(f"CPU Usage: {data['cpu_usage']}%")
            self.ram_label.setText(f"RAM Usage: {data['ram_usage']}%")
            self.gpu_label.setText(f"GPU Usage: {data['gpu_usage']}")
            # Update progress bars (adjusted for smaller width)
            cpu_width = int(240 * (float(data['cpu_usage']) / 100))
            ram_width = int(240 * (float(data['ram_usage']) / 100))
            gpu_width = int(240 * (float(data['gpu_usage']) / 100)) if isinstance(data['gpu_usage'], (int, float)) else 0
            self.cpu_progress.setFixedWidth(cpu_width)
            self.ram_progress.setFixedWidth(ram_width)
            self.gpu_progress.setFixedWidth(gpu_width)

    def show_error(self, message):
        print(f"Error: {message}")
        logging.error(message)

    def closeEvent(self, event):
        self.camera_subscriber.stop()
        self.yolo_subscriber.stop()
        self.zmq_socket.close()
        self.zmq_context.term()
        event.accept()

def setup_gui_fallback(vlm_endpoint, instruction, inference_interval):
    """Fallback using OpenCV for camera feed with on-screen text, styled with a whitish theme."""
    logger.debug("Starting fallback GUI...")

    # Initialize camera subscriber
    camera_subscriber = CameraSubscriber(port="5556")
    if not camera_subscriber.initialize():
        logger.error("Failed to initialize camera subscriber")
        print("Error: Failed to initialize camera subscriber")
        return

    camera_subscriber.start()

    # Initialize YOLO subscriber
    yolo_subscriber = YoloSubscriber(port="5555")
    if not yolo_subscriber.initialize():
        logger.error("Failed to initialize YOLO subscriber")
        print("Error: Failed to initialize YOLO subscriber")
        return

    yolo_subscriber.start()

    # Set up ZMQ publisher for sending VLM responses
    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    try:
        zmq_socket.bind("tcp://127.0.0.1:5558")
        logging.info("ZMQ publisher for VLM responses bound to tcp://127.0.0.1:5558")
    except zmq.error.ZMQError as e:
        logger.error(f"Failed to bind ZMQ publisher for VLM responses: {e}")
        raise

    vlm_client = VLMClient(vlm_endpoint, instruction)

    # Initialize processing state
    is_running = False
    interval_name = {
        100: "Flash (100ms)",
        250: "Cheetah (250ms)",
        500: "Ideal (500ms)",
        1000: "Normal (1000ms)",
        1500: "Lethargic (1500ms)",
        2000: "Sloth (2000ms)"
    }.get(inference_interval, "Ideal (500ms)")

    frame = None
    latest_yolo = None
    def on_frame_received(frame_data):
        nonlocal frame
        frame = frame_data

    def on_yolo_received(yolo_data):
        nonlocal latest_yolo
        latest_yolo = yolo_data

    camera_subscriber.frame_received.connect(on_frame_received)
    yolo_subscriber.yolo_received.connect(on_yolo_received)

    last_vlm_update = 0
    last_sys_update = 0
    state_flash = 0  # For flashing effect
    flash_duration = 0.5  # Flash for 0.5 seconds
    flash_timer = time.time()
    while True:
        if frame is None:
            time.sleep(0.033)
            continue

        frame_copy = frame.copy()
        if frame_copy.dtype != np.uint8:
            frame_copy = frame_copy.astype(np.uint8)

        if is_running:
            # Update VLM response
            current_time = time.time()
            if current_time - last_vlm_update >= (inference_interval / 1000.0):
                vlm_client.process_frame(frame_copy)
                last_vlm_update = current_time
            vlm_text = vlm_client.get_response()[:50]

            # Prepare combined data with YOLO and VLM
            combined_data = {
                "description": vlm_text
            }
            if latest_yolo:
                combined_data.update(latest_yolo)

            # Send combined data via ZMQ
            try:
                packed = msgpack.packb(combined_data, use_bin_type=True)
                zmq_socket.send(packed)
                logger.debug(f"Sent combined VLM and YOLO data: {combined_data}")
            except zmq.error.ZMQError as e:
                logger.error(f"Error sending combined VLM and YOLO data via ZMQ: {e}")

            # Update system usage
            if current_time - last_sys_update >= 1.0:
                try:
                    response = requests.get("http://localhost:5000/system-usage", timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        sys_text = f"CPU: {data['cpu_usage']}%, RAM: {data['ram_usage']}%, GPU: {data['gpu_usage']}"
                    else:
                        sys_text = "System Usage: Error"
                except requests.RequestException as e:
                    sys_text = f"System Usage: Error - {e}"
                last_sys_update = current_time
        else:
            vlm_text = "Processing Stopped"
            sys_text = "System Usage: N/A"

        # Add white background rectangles for text (adjusted for smaller frame)
        cv2.rectangle(frame_copy, (8, 8), (232, 72), (245, 247, 250), -1)  # White background (#F5F7FA)
        cv2.rectangle(frame_copy, (8, 8), (232, 72), (206, 212, 218), 2)  # Border (#CED4DA)

        # Flashing effect for state change
        current_time = time.time()
        if current_time - flash_timer < flash_duration:
            state_flash = 1 - state_flash  # Toggle between 0 and 1
            flash_color = (255, 255, 255) if state_flash else (52, 58, 64)  # White or dark gray
        else:
            flash_color = (52, 58, 64)  # Default to dark gray

        # Overlay text with dark gray color for readability (increased font size for VLM response)
        state_text = f"State: {'Running' if is_running else 'Stopped'} (Press 's' to toggle)"
        interval_text = f"Speed: {interval_name}"
        cv2.putText(frame_copy, state_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, flash_color, 1)
        cv2.putText(frame_copy, interval_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (52, 58, 64), 1)
        cv2.putText(frame_copy, vlm_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 58, 64), 1)

        cv2.imshow("Camera Feed - Fallback", frame_copy)

        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            is_running = not is_running
            flash_timer = time.time()  # Reset flash timer
            state_flash = 0
            logger.debug(f"Processing toggled to {'Running' if is_running else 'Stopped'} in fallback GUI")

        time.sleep(0.033)  # ~30 FPS

    camera_subscriber.stop()
    yolo_subscriber.stop()
    zmq_socket.close()
    zmq_context.term()
    cv2.destroyAllWindows()
    logger.info("Fallback GUI closed.")

def main():
    """Main function to orchestrate the server startups and GUI."""
    parser = argparse.ArgumentParser(description="Drone GUI with VLM integration")
    parser.add_argument("--vlm-endpoint", default="http://localhost:8080/v1/chat/completions", help="VLM server endpoint")
    parser.add_argument("--fallback", action="store_true", help="Use OpenCV fallback GUI")
    parser.add_argument("--instruction", default="Describe the scene", help="VLM instruction")
    parser.add_argument("--inference-interval", type=int, default=500, choices=[100, 250, 500, 1000, 1500, 2000], help="Inference interval in milliseconds")
    args = parser.parse_args()

    try:
        # Save the Flask script
        flask_file_path = save_flask_script()

        # Start the VLM server
        start_vlm_server()

        # Start the Flask server
        start_flask_server(flask_file_path)

        # Start the GUI
        if args.fallback:
            setup_gui_fallback(args.vlm_endpoint, args.instruction, args.inference_interval)
        else:
            app = QApplication(sys.argv)
            window = MainWindow(args.vlm_endpoint, args.instruction, args.inference_interval)
            window.show()
            sys.exit(app.exec())

        logger.info("Servers and GUI started successfully.")
    except Exception as e:
        logger.error(f"Failed to start servers or GUI: {e}")
        raise

if __name__ == "__main__":
    main()