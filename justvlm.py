import subprocess
import time
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
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTextEdit, QComboBox, QPushButton, QGraphicsDropShadowEffect, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt6.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, QThread, pyqtSignal
import csv

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

def check_webcam_free(device=0):
    """Check if webcam is free by attempting to open and release it."""
    logger.debug(f"Checking if webcam device {device} is free...")
    try:
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            cap.release()
            logger.debug("Webcam is free.")
            return True
        else:
            logger.error("Webcam could not be opened.")
            return False
    except Exception as e:
        logger.error(f"Error checking webcam: {e}")
        return False

class CameraFeed:
    """Manage camera feed in a separate thread."""
    def __init__(self, device=0, resolution=(240, 180)):
        logger.debug(f"Initializing camera with device {device} and resolution {resolution}")
        if not check_webcam_free(device):
            raise RuntimeError("Webcam is busy or unavailable")
        self.cap = cv2.VideoCapture(device)
        self.resolution = resolution
        self.frame = None
        self.running = True
        self.lock = Lock()
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            self.cap.release()
            raise RuntimeError("Camera not found")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def start(self):
        """Start capturing frames in a separate thread."""
        logger.debug("Starting camera thread...")
        Thread(target=self.update, daemon=True).start()

    def update(self):
        """Capture frames continuously."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = cv2.resize(frame, self.resolution)
            time.sleep(0.033)  # ~30 FPS

    def get_frame(self):
        """Get the latest frame."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            else:
                return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

    def stop(self):
        """Stop capturing and release camera."""
        logger.debug("Stopping camera...")
        self.running = False
        time.sleep(0.1)  # Brief delay to ensure thread stops
        if self.cap and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
        self.cap = None

class VLMClient:
    """Manage VLM API interactions with retries."""
    def __init__(self, endpoint, instruction="Describe the scene", csv_file_path="vlm_responses.csv"):
        logger.debug(f"Initializing VLM client with endpoint: {endpoint}")
        self.endpoint = endpoint
        self.instruction = instruction
        self.latest_response = "Waiting for VLM response..."
        self.lock = Lock()
        self.session = requests.Session()
        self.csv_file_path = csv_file_path
        self._initialize_csv()  # Initialize CSV file
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def _initialize_csv(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "response"])  # Headers for the CSV

    def _log_to_csv(self, response):
        """Log the VLM response to CSV."""
        with open(self.csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), response])

    def process_frame(self, frame):
        """Send frame to VLM and get response."""
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
            response = self.session.post(self.endpoint, json=payload, timeout=3)
            if response.status_code == 200:
                data = response.json()
                vlm_text = data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            else:
                vlm_text = f"Error: HTTP {response.status_code}"
        except requests.RequestException as e:
            vlm_text = f"Error: {e}"
        
        with self.lock:
            self.latest_response = vlm_text
            logger.debug(f"VLM response: {vlm_text}")

        # Log the VLM response to CSV
        self._log_to_csv(vlm_text)

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
        # Allow widget to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

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
        self.setWindowTitle("Zenitsu V1")
        self.setMinimumSize(600, 300)  # Set minimum size to prevent overly small window
        self.resize(800, 400)  # Initial size

        # Initialize camera and VLM client
        self.camera = CameraFeed()
        self.camera.start()
        self.vlm_client = VLMClient(vlm_endpoint, instruction)
        self.inference_interval = inference_interval
        self.is_running = False

        # Setup UI
        self.setup_ui()

        # Setup timers for updates
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feed)
        self.camera_timer.setInterval(50)

        self.vlm_timer = QTimer()
        self.vlm_timer.timeout.connect(self.update_vlm_response)
        self.vlm_timer.setInterval(inference_interval)

        self.sys_usage_timer = QTimer()
        self.sys_usage_timer.timeout.connect(self.start_system_usage_update)
        self.sys_usage_timer.setInterval(1000)

        # Setup system usage worker thread
        self.sys_usage_worker = SystemUsageWorker()
        self.sys_usage_worker.usage_updated.connect(self.update_system_usage)

    def setup_ui(self):
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Apply gradient background
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
        sections_layout.setSpacing(10)
        sections_layout.setStretch(0, 1)
        sections_layout.setStretch(1, 1)
        sections_layout.setStretch(2, 1)  # Equal stretch for all sections

        # Camera Feed Section
        camera_widget = HoverWidget()
        camera_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #F9FAFB);
                border: 1px solid #CED4DA;
                border-radius: 5px;
            }
        """)
        camera_layout = QVBoxLayout(camera_widget)
        camera_layout.setSpacing(5)
        camera_layout.setContentsMargins(10, 10, 10, 10)
        camera_title = QLabel("Camera Feed")
        camera_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #212529; text-align: center;")
        camera_layout.addWidget(camera_title)
        self.camera_label = QLabel("Loading...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("color: #6C757D; background: #FFFFFF;")
        self.camera_label.setScaledContents(True)  # Allow pixmap to scale
        self.camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(4)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.camera_label.setGraphicsEffect(shadow)
        camera_layout.addWidget(self.camera_label)
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
        vlm_layout = QVBoxLayout(vlm_widget)
        vlm_layout.setSpacing(5)
        vlm_layout.setContentsMargins(10, 10, 10, 10)
        vlm_title = QLabel("VLM Response")
        vlm_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #212529; text-align: center;")
        vlm_layout.addWidget(vlm_title)
        self.vlm_text = QTextEdit()
        self.vlm_text.setReadOnly(True)
        self.vlm_text.setText("Waiting for VLM response...")
        self.vlm_text.setStyleSheet("""
            QTextEdit {
                background: transparent;
                border: none;
                color: #343A40;
                font-size: 14px;
                padding: 5px;
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
        self.vlm_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
        sys_usage_layout = QVBoxLayout(sys_usage_widget)
        sys_usage_layout.setSpacing(5)
        sys_usage_layout.setContentsMargins(10, 10, 10, 10)
        sys_usage_title = QLabel("System Usage")
        sys_usage_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #212529; text-align: center;")
        sys_usage_layout.addWidget(sys_usage_title)
        self.cpu_label = QLabel("CPU Usage: -- %")
        self.cpu_label.setStyleSheet("font-size: 12px; color: #343A40; margin-left: 5px;")
        self.cpu_progress = QWidget()
        self.cpu_progress.setFixedHeight(2)
        self.cpu_progress.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #28A745, stop:1 #34C759);
        """)
        self.ram_label = QLabel("RAM Usage: -- %")
        self.ram_label.setStyleSheet("font-size: 12px; color: #343A40; margin-left: 5px;")
        self.ram_progress = QWidget()
        self.ram_progress.setFixedHeight(2)
        self.ram_progress.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007BFF, stop:1 #66B0FF);
        """)
        self.gpu_label = QLabel("GPU Usage: --")
        self.gpu_label.setStyleSheet("font-size: 12px; color: #343A40; margin-left: 5px;")
        self.gpu_progress = QWidget()
        self.gpu_progress.setFixedHeight(2)
        self.gpu_progress.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #DC3545, stop:1 #FF6B6B);
        """)
        sys_usage_layout.addWidget(self.cpu_label)
        sys_usage_layout.addWidget(self.cpu_progress)
        sys_usage_layout.addWidget(self.ram_label)
        sys_usage_layout.addWidget(self.ram_progress)
        sys_usage_layout.addWidget(self.gpu_label)
        sys_usage_layout.addWidget(self.gpu_progress)
        sys_usage_layout.addStretch()  # Push content to top
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
                padding: 5px;
            }
        """)
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setSpacing(10)

        # Inference Speed Control
        interval_label = QLabel("Inference Speed:")
        interval_label.setStyleSheet("font-size: 11px; color: #343A40;")
        controls_layout.addWidget(interval_label)

        self.interval_select = QComboBox()
        self.interval_select.addItem("Flash (100ms)", 100)
        self.interval_select.addItem("Cheetah (250ms)", 250)
        self.interval_select.addItem("Ideal (500ms)", 500)
        self.interval_select.addItem("Normal (1000ms)", 1000)
        self.interval_select.addItem("Lethargic (1500ms)", 1500)
        self.interval_select.addItem("Sloth (2000ms)", 2000)
        self.interval_select.setCurrentIndex(2)
        self.interval_select.currentIndexChanged.connect(self.update_inference_interval)
        self.interval_select.setStyleSheet("""
            QComboBox {
                background: #FFFFFF;
                border: 1px solid #CED4DA;
                border-radius: 5px;
                padding: 3px;
                font-size: 11px;
                color: #343A40;
            }
            QComboBox QAbstractItemView {
                background: #FFFFFF;
                border: 1px solid #CED4DA;
                selection-background-color: #2ECC71;
                color: #343A40;
            }
        """)
        self.interval_select.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        controls_layout.addWidget(self.interval_select)

        # Start/Stop Button
        self.start_button = StartStopButton("Start")
        self.start_button.clicked.connect(self.toggle_processing)
        controls_layout.addWidget(self.start_button)

        main_layout.addWidget(controls_widget, stretch=0)

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
        # Force GUI update
        QApplication.processEvents()
        time.sleep(0.01)

    def update_camera_feed(self):
        if not self.is_running:
            return
        frame = self.camera.get_frame()
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # Scale pixmap to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(image).scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

    def update_vlm_response(self):
        if not self.is_running:
            return
        frame = self.camera.get_frame()
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        thread = Thread(target=self.vlm_client.process_frame, args=(frame,), daemon=True)
        thread.start()
        vlm_response = self.vlm_client.get_response()
        logger.debug(f"VLM response: {vlm_response}")
        self.vlm_text.setText(vlm_response)
        # Fade-in effect
        self.fade_animation = QPropertyAnimation(self.vlm_text, b"opacity")
        self.fade_animation.setDuration(500)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.fade_animation.start()

    def start_system_usage_update(self):
        if not self.is_running:
            return
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
            # Calculate progress bar widths based on parent widget width
            parent_width = self.cpu_progress.parent().width() - 20  # Account for margins
            cpu_width = int(parent_width * (float(data['cpu_usage']) / 100))
            ram_width = int(parent_width * (float(data['ram_usage']) / 100))
            gpu_width = int(parent_width * (float(data['gpu_usage']) / 100)) if isinstance(data['gpu_usage'], (int, float)) else 0
            self.cpu_progress.setFixedWidth(max(0, cpu_width))
            self.ram_progress.setFixedWidth(max(0, ram_width))
            self.gpu_progress.setFixedWidth(max(0, gpu_width))

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()

def setup_gui_fallback(vlm_endpoint, instruction, inference_interval):
    """Fallback using OpenCV for camera feed with on-screen text, styled with a whitish theme."""
    logger.debug("Starting fallback GUI...")
    try:
        camera = CameraFeed()
        camera.start()
        vlm_client = VLMClient(vlm_endpoint, instruction)
    except RuntimeError as e:
        logger.error(str(e))
        print(f"Error: {e}")
        return

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

    last_vlm_update = 0
    last_sys_update = 0
    state_flash = 0  # For flashing effect
    flash_duration = 0.5  # Flash for 0.5 seconds
    flash_timer = time.time()
    while True:
        frame = camera.get_frame()
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        if is_running:
            # Update VLM response
            current_time = time.time()
            if current_time - last_vlm_update >= (inference_interval / 1000.0):
                vlm_client.process_frame(frame)
                last_vlm_update = current_time
            vlm_text = vlm_client.get_response()[:50]

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

        # Add white background rectangles for text
        cv2.rectangle(frame, (8, 8), (232, 72), (245, 247, 250), -1)  # White background (#F5F7FA)
        cv2.rectangle(frame, (8, 8), (232, 72), (206, 212, 218), 2)  # Border (#CED4DA)

        # Flashing effect for state change
        current_time = time.time()
        if current_time - flash_timer < flash_duration:
            state_flash = 1 - state_flash  # Toggle between 0 and 1
            flash_color = (255, 255, 255) if state_flash else (52, 58, 64)  # White or dark gray
        else:
            flash_color = (52, 58, 64)  # Default to dark gray

        # Overlay text with dark gray color
        state_text = f"State: {'Running' if is_running else 'Stopped'} (Press 's' to toggle)"
        interval_text = f"Speed: {interval_name}"
        cv2.putText(frame, state_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, flash_color, 1)
        cv2.putText(frame, interval_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (52, 58, 64), 1)
        cv2.putText(frame, vlm_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 58, 64), 1)

        cv2.imshow("Camera Feed - Fallback", frame)

        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            is_running = not is_running
            flash_timer = time.time()  # Reset flash timer
            state_flash = 0
            logger.debug(f"Processing toggled to {'Running' if is_running else 'Stopped'} in fallback GUI")

        time.sleep(0.033)  # ~30 FPS

    camera.stop()
    cv2.destroyAllWindows()
    logger.info("Fallback GUI closed.")

def main():
    """Main function to orchestrate the server startups and GUI."""
    parser = argparse.ArgumentParser(description="Drone GUI with VLM integration")
    parser.add_argument("--vlm-endpoint", default="http://localhost:8080/v1/chat/completions", help="VLM server endpoint")
    parser.add_argument("--fallback", action="store_true", help="Use OpenCV fallback GUI")
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device index")
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
