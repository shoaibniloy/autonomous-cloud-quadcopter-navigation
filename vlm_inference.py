import subprocess
import time
import os
import logging
import sys
import cv2
import requests
import numpy as np
from threading import Thread
import base64
from io import BytesIO
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import csv
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTextEdit, QComboBox, QPushButton, QGraphicsDropShadowEffect
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask script for system monitoring
FLASK_SCRIPT = """from flask import Flask, jsonify
import psutil
import subprocess
import platform
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_ram_usage():
    mem = psutil.virtual_memory()
    return mem.percent

def get_gpu_usage():
    if platform.system() == 'Linux':
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            gpu_usage = result.stdout.decode('utf-8').strip()
            return gpu_usage if gpu_usage else "GPU not in use"
        except FileNotFoundError:
            return "No NVIDIA GPU or nvidia-smi not installed"
    return "Not a Linux system or no GPU"

@app.route('/system-usage', methods=['GET'])
def system_usage():
    return jsonify({
        'cpu_usage': get_cpu_usage(),
        'ram_usage': get_ram_usage(),
        'gpu_usage': get_gpu_usage()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
"""

def save_flask_script():
    """Save Flask script to a temporary file."""
    flask_file_path = os.path.join(os.getcwd(), "system_monitor.py")
    with open(flask_file_path, "w") as f:
        f.write(FLASK_SCRIPT)
    logger.info(f"Flask script saved to {flask_file_path}")
    return flask_file_path

def start_vlm_server():
    """Start the VLM server in a new terminal."""
    logger.debug("Starting VLM server...")
    vlm_command = (
        "cd ~/Drone/YOLO+LLM/llama.cpp/build && "
        "./bin/llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -ngl 99"
    )
    subprocess.Popen(["xterm", "-hold", "-e", f"bash -c '{vlm_command}'"])
    time.sleep(10)  # Wait for server to start
    logger.info("VLM server started.")

def start_flask_server(flask_file_path):
    """Start the Flask system monitoring server."""
    logger.debug("Starting Flask server...")
    flask_command = f"python3 {flask_file_path}"
    subprocess.Popen(["xterm", "-hold", "-e", f"bash -c '{flask_command}'"])
    time.sleep(2)  # Wait for server to start
    logger.info("Flask server started.")

class VLMClient:
    """Handle VLM API interactions."""
    def __init__(self, endpoint, instruction="Describe the scene"):
        self.endpoint = endpoint
        self.instruction = instruction
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def process_frame(self, frame):
        """Process a frame with the VLM."""
        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        payload = {
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": self.instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]}
            ]
        }
        try:
            response = self.session.post(self.endpoint, json=payload, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
        except requests.RequestException as e:
            logger.error(f"VLM request failed: {e}")
            return f"Error: {e}"

class VLMWorker(QThread):
    """Worker thread for VLM processing."""
    response_ready = pyqtSignal(int, str)

    def __init__(self, vlm_client, frame, frame_number):
        super().__init__()
        self.vlm_client = vlm_client
        self.frame = frame
        self.frame_number = frame_number

    def run(self):
        vlm_text = self.vlm_client.process_frame(self.frame)
        self.response_ready.emit(self.frame_number, vlm_text)

class SystemUsageWorker(QThread):
    """Worker thread for system usage updates."""
    usage_updated = pyqtSignal(dict)

    def run(self):
        try:
            response = requests.get("http://localhost:5000/system-usage", timeout=2)
            response.raise_for_status()
            self.usage_updated.emit(response.json())
        except requests.RequestException as e:
            logger.error(f"System usage fetch failed: {e}")
            self.usage_updated.emit({})

class MainWindow(QMainWindow):
    def __init__(self, source, vlm_endpoint="http://localhost:8080/v1/chat/completions", instruction="Describe the scene", inference_interval=500):
        super().__init__()
        self.setWindowTitle("Video Scene Analysis")
        self.setMinimumSize(900, 400)

        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open source: {source}")
            sys.exit(1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame = None

        # Setup CSV file
        if isinstance(source, str):
            self.csv_path = os.path.join(os.path.dirname(source), "vlm_responses.csv")
        else:
            self.csv_path = "vlm_responses.csv"
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame_number', 'vlm_response'])

        # VLM setup
        self.vlm_client = VLMClient(vlm_endpoint, instruction)
        self.inference_interval = inference_interval
        self.is_running = False
        self.active_workers = []  # List to track active worker threads

        # UI setup
        self.setup_ui()

        # Timers
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feed)
        self.camera_timer.setInterval(int(1000 / self.fps))

        self.vlm_timer = QTimer()
        self.vlm_timer.timeout.connect(self.update_vlm_response)
        self.vlm_timer.setInterval(inference_interval)

        self.sys_usage_timer = QTimer()
        self.sys_usage_timer.timeout.connect(self.update_system_usage)
        self.sys_usage_timer.setInterval(1000)

    def setup_ui(self):
        """Set up the PyQt GUI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Main sections
        sections_layout = QHBoxLayout()

        # Video Feed
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        self.video_label = QLabel("Loading video...")
        self.video_label.setMinimumSize(400, 300)
        video_layout.addWidget(QLabel("Video Feed"))
        video_layout.addWidget(self.video_label)
        sections_layout.addWidget(video_widget, stretch=1)

        # VLM Response
        vlm_widget = QWidget()
        vlm_layout = QVBoxLayout(vlm_widget)
        self.vlm_text = QTextEdit("Waiting for VLM response...")
        self.vlm_text.setReadOnly(True)
        self.vlm_text.setFixedSize(300, 200)
        vlm_layout.addWidget(QLabel("VLM Response"))
        vlm_layout.addWidget(self.vlm_text)
        sections_layout.addWidget(vlm_widget)

        # System Usage
        sys_widget = QWidget()
        sys_layout = QVBoxLayout(sys_widget)
        self.cpu_label = QLabel("CPU: --%")
        self.ram_label = QLabel("RAM: --%")
        self.gpu_label = QLabel("GPU: --")
        sys_layout.addWidget(QLabel("System Usage"))
        sys_layout.addWidget(self.cpu_label)
        sys_layout.addWidget(self.ram_label)
        sys_layout.addWidget(self.gpu_label)
        sections_layout.addWidget(sys_widget)

        main_layout.addLayout(sections_layout)

        # Controls
        controls_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_processing)
        controls_layout.addWidget(self.start_button)
        main_layout.addLayout(controls_layout)

    def toggle_processing(self):
        """Start or stop video processing."""
        self.is_running = not self.is_running
        if self.is_running:
            self.camera_timer.start()
            self.vlm_timer.start()
            self.sys_usage_timer.start()
            self.start_button.setText("Stop")
        else:
            self.camera_timer.stop()
            self.vlm_timer.stop()
            self.sys_usage_timer.stop()
            self.start_button.setText("Start")

    def update_camera_feed(self):
        """Update video feed in GUI."""
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)
        else:
            self.toggle_processing()  # Stop on error or end

    def update_vlm_response(self):
        """Process frame with VLM."""
        if self.frame is not None:
            frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            worker = VLMWorker(self.vlm_client, self.frame.copy(), frame_number)
            worker.response_ready.connect(self.handle_vlm_response)
            worker.finished.connect(lambda: self.active_workers.remove(worker))
            self.active_workers.append(worker)
            worker.start()

    def handle_vlm_response(self, frame_number, vlm_text):
        """Display VLM response and save to CSV."""
        self.vlm_text.setText(vlm_text)
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([frame_number, vlm_text])
        logger.debug(f"Frame {frame_number}: {vlm_text}")

    def update_system_usage(self):
        """Fetch and display system usage."""
        worker = SystemUsageWorker()
        worker.usage_updated.connect(self.display_system_usage)
        worker.finished.connect(lambda: self.active_workers.remove(worker))
        self.active_workers.append(worker)
        worker.start()

    def display_system_usage(self, data):
        """Update system usage labels."""
        self.cpu_label.setText(f"CPU: {data.get('cpu_usage', '--')}%")
        self.ram_label.setText(f"RAM: {data.get('ram_usage', '--')}%")
        self.gpu_label.setText(f"GPU: {data.get('gpu_usage', '--')}")

    def closeEvent(self, event):
        """Clean up on close."""
        self.toggle_processing()  # Stop timers and processing
        # Wait for all active workers to finish
        for worker in self.active_workers:
            worker.wait()
        self.cap.release()
        event.accept()

def main():
    """Run the application."""
    source = 0  # Use default camera
    try:
        flask_file_path = save_flask_script()
        start_vlm_server()
        start_flask_server(flask_file_path)
        app = QApplication(sys.argv)
        window = MainWindow(source)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()