import subprocess
import os
import sys
import threading
import time
import logging
import signal
from statistics import mean

# Set up logging
logging.basicConfig(filename='orchestration.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Python executable
python_executable = sys.executable
# List to track terminal processes for cleanup
terminal_processes = []

# Verify script existence
for script in ['camera_server.py', 'vlm.py', 'yolo3d.py', 'llm_subscriber.py', 'sensors.py']:
    if not os.path.exists(script):
        logging.error(f"Script {script} not found in current directory")
        print(f"Error: Script {script} not found in current directory")
        sys.exit(1)

# Event for graceful shutdown
stop_event = threading.Event()

def cleanup_terminals():
    """Terminate all tracked terminal processes."""
    logging.info("Cleaning up terminal processes")
    for proc in terminal_processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            logging.warning(f"Process {proc.pid} did not terminate gracefully, killed")
        except Exception as e:
            logging.error(f"Error terminating process {proc.pid}: {e}")
    logging.info("All terminal processes terminated")

def run_yolo_script():
    logging.info("Starting YOLO3D script")
    with open(os.devnull, 'w') as devnull:
        start_time = time.time()
        process = None
        while not stop_event.is_set():
            try:
                process = subprocess.Popen(
                    [python_executable, 'yolo3d.py'], 
                    env=os.environ, stdout=devnull, stderr=devnull, text=True
                )
                while not stop_event.is_set():
                    if process.poll() is not None:
                        returncode = process.returncode
                        logging.info(f"YOLO3D script terminated with return code {returncode}")
                        if time.time() - start_time < 10 and returncode != 0:
                            logging.warning("YOLO3D script crashed within 10 seconds. Restarting...")
                            break
                        return
                    time.sleep(0.2)
            except FileNotFoundError:
                logging.error(f"YOLO3D script executable not found: {python_executable}")
                stop_event.set()
                return
            except PermissionError:
                logging.error(f"Permission denied running YOLO3D script: {python_executable}")
                stop_event.set()
                return
            except Exception as e:
                logging.error(f"Error starting YOLO3D script: {e}")
                if time.time() - start_time < 10:
                    logging.warning("Retrying YOLO3D script start...")
                    time.sleep(1)
                else:
                    logging.error("YOLO3D script failed repeatedly. Exiting.")
                    stop_event.set()
                    return
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                logging.info("YOLO3D script process terminated")

def run_llm_script():
    logging.info("Starting LLM script")
    start_time = time.time()
    process = None
    while not stop_event.is_set():
        try:
            process = subprocess.Popen(
                [python_executable, 'llm_subscriber.py'], 
                env=os.environ, stderr=subprocess.PIPE, text=True
            )
            while not stop_event.is_set():
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    returncode = process.returncode
                    logging.info(f"LLM script terminated with return code {returncode}")
                    if stderr:
                        logging.error(f"LLM script error: {stderr}")
                    if time.time() - start_time < 10 and returncode != 0:
                        logging.warning("LLM script crashed within 10 seconds. Restarting...")
                        break
                    return
                time.sleep(0.2)
        except FileNotFoundError:
            logging.error(f"LLM script executable not found: {python_executable}")
            stop_event.set()
            return
        except PermissionError:
            logging.error(f"Permission denied running LLM script: {python_executable}")
            stop_event.set()
            return
        except Exception as e:
            logging.error(f"Error starting LLM script: {e}")
            if time.time() - start_time < 10:
                logging.warning("Retrying LLM script start...")
                time.sleep(1)
            else:
                logging.error("LLM script failed repeatedly. Exiting.")
                stop_event.set()
                return
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        _, stderr = process.communicate()
        if stderr:
            logging.error(f"LLM script error on termination: {stderr}")
        logging.info("LLM script process terminated")

def run_sensors_script():
    logging.info("Starting sensors script")
    start_time = time.time()
    process = None
    while not stop_event.is_set():
        try:
            process = subprocess.Popen(
                [python_executable, 'sensors.py'], 
                env=os.environ, stderr=subprocess.PIPE, text=True
            )
            while not stop_event.is_set():
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    returncode = process.returncode
                    logging.info(f"Sensors script terminated with return code {returncode}")
                    if stderr:
                        logging.error(f"Sensors script error: {stderr}")
                    if time.time() - start_time < 10 and returncode != 0:
                        logging.warning("Sensors script crashed within 10 seconds. Restarting...")
                        break
                    return
                time.sleep(0.2)
        except FileNotFoundError:
            logging.error(f"Sensors script executable not found: {python_executable}")
            stop_event.set()
            return
        except PermissionError:
            logging.error(f"Permission denied running sensors script: {python_executable}")
            stop_event.set()
            return
        except Exception as e:
            logging.error(f"Error starting sensors script: {e}")
            if time.time() - start_time < 10:
                logging.warning("Retrying sensors script start...")
                time.sleep(1)
            else:
                logging.error("Sensors script failed repeatedly. Exiting.")
                stop_event.set()
                return
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        _, stderr = process.communicate()
        if stderr:
            logging.error(f"Sensors script error on termination: {stderr}")
        logging.info("Sensors script process terminated")

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    logging.info("Received termination signal. Stopping scripts...")
    stop_event.set()
    cleanup_terminals()

# Function to read latency from a log file
def get_latency(file_path):
    try:
        with open(file_path, 'r') as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0.0  # Default to 0 if file is missing or invalid

# Store latencies for averaging
latency_data = {
    "VLM": [],
    "YOLOv11": [],
    "Depth Estimation": [],
    "JSON Packaging + LLM Processing": []
}

def collect_latencies():
    """Collect latencies from log files periodically."""
    while not stop_event.is_set():
        latency_data["VLM"].append(get_latency('vlm_latency.log'))
        latency_data["YOLOv11"].append(get_latency('yolo_latency.log'))
        latency_data["Depth Estimation"].append(get_latency('depth_latency.log'))
        latency_data["JSON Packaging + LLM Processing"].append(get_latency('llm_latency.log'))
        time.sleep(1)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info("Starting orchestration script")
    print("Starting drone context analysis system...")

    # Start ollama serve in a new terminal
    logging.info("Starting ollama serve in a new terminal")
    ollama_process = subprocess.Popen([
        "xterm", "-hold", "-e", "bash -c 'source ~/.bashrc && ollama serve'"
    ])
    terminal_processes.append(ollama_process)
    time.sleep(2)  # Wait for ollama serve to initialize

    # Start camera_server.py in a new terminal
    logging.info("Starting camera server in a new terminal")
    camera_server_command = f"{python_executable} camera_server.py --device 0 --resolution 240 180 --port 5556"
    camera_process = subprocess.Popen([
        "xterm", "-hold", "-e", f"bash -c 'source ~/.bashrc && {camera_server_command}'"
    ])
    terminal_processes.append(camera_process)
    time.sleep(2)  # Wait for camera server to initialize

    # Start vlm.py in a new terminal
    logging.info("Starting VLM script in a new terminal")
    vlm_command = f"{python_executable} vlm.py --vlm-endpoint http://localhost:8080/v1/chat/completions --instruction 'You are a drone, decide what you need to do now' --inference-interval 1000"
    vlm_process = subprocess.Popen([
        "xterm", "-hold", "-e", f"bash -c 'source ~/.bashrc && {vlm_command}'"
    ])
    terminal_processes.append(vlm_process)
    time.sleep(2)  # Wait for VLM script to initialize

    # Start sensors.py in a new terminal
    logging.info("Starting sensors script in a new terminal")
    sensors_command = f"{python_executable} sensors.py"
    sensors_process = subprocess.Popen([
        "xterm", "-hold", "-e", f"bash -c 'source ~/.bashrc && {sensors_command}'"
    ])
    terminal_processes.append(sensors_process)
    time.sleep(2)  # Wait for sensors script to initialize

    # Start both scripts simultaneously
    llm_thread = threading.Thread(target=run_llm_script)
    llm_thread.start()

    yolo_thread = threading.Thread(target=run_yolo_script)
    yolo_thread.start()

    sensors_thread = threading.Thread(target=run_sensors_script)
    sensors_thread.start()

    # Start latency collection thread
    latency_thread = threading.Thread(target=collect_latencies)
    latency_thread.start()

    # Wait for threads to finish
    yolo_thread.join()
    llm_thread.join()
    sensors_thread.join()
    latency_thread.join()
    logging.info("All scripts have finished execution")
    print("Drone context analysis system stopped.")

    # Cleanup all terminal processes
    cleanup_terminals()

    # Calculate and print final latency results
    fps = 30  # Assuming 30 FPS based on camera resolution and typical drone systems
    avg_latencies = {
        key: mean([x * 1000 for x in values if x > 0]) if any(x > 0 for x in values) else 0
        for key, values in latency_data.items()
    }
    print("\nPerception and Decision-Making")
    print(f"1) Sensor Processing and Response Time: Latencies of perception modules were measured at {fps} FPS:")
    print(f"• Vision Language Model (VLM): {avg_latencies['VLM']:.2f} ms per frame.")
    print(f"• YOLOv11: {avg_latencies['YOLOv11']:.2f} ms per frame.")
    print(f"• Depth Estimation: {avg_latencies['Depth Estimation']:.2f} ms per frame.")
    print(f"• JSON Packaging + LLM Processing: {avg_latencies['JSON Packaging + LLM Processing']:.2f} ms per decision.")