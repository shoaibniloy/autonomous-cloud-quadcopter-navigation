import socket
import time
import re
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import zmq
import msgpack
import zlib
import logging
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from math import atan2
import random

# WiFi configuration for ESP-Now
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 12345     # Port for receiving ESP-Now data
TIMEOUT = 0.02

# Thresholds for invalid ToF readings
INVALID_READINGS = {8190, 65535}

# Gravity constant for Z-axis correction
GRAVITY = 9.8  # m/s^2

# Set up logging
logging.basicConfig(filename='sensors.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Kalman Filter Class
class KalmanFilter:
    def __init__(self, Q, R):
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = 1.0  # Estimate covariance
        self.x = 0.0  # State estimate

    def update(self, z, measurement):
        # Prediction Step
        self.P = self.P + self.Q

        # Measurement Update Step
        K = self.P / (self.P + self.R)  # Kalman gain
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        
        return self.x

def initialize_wifi():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(TIMEOUT)
            sock.bind((UDP_IP, UDP_PORT))
            logging.info(f"UDP socket bound to {UDP_IP}:{UDP_PORT}")
            return sock
        except socket.error as e:
            logging.error(f"Failed to bind UDP socket: {e}")
            time.sleep(0.5)

def setup_zmq_publisher(port="5550"):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    try:
        socket.bind(f"tcp://*:{port}")
        logging.info(f"ZMQ publisher bound to tcp://*:{port}")
        return context, socket
    except zmq.error.ZMQError as e:
        logging.error(f"Failed to bind ZMQ socket on port {port}: {e}")
        raise

def send_zmq_message(socket, data):
    try:
        packed = msgpack.packb(data, use_bin_type=True)
        compressed = zlib.compress(packed)
        socket.send(compressed, zmq.NOBLOCK)
        logging.debug(f"Sent ZMQ message: {data}")
    except zmq.error.ZMQError as e:
        logging.error(f"Error sending ZMQ message: {e}")

# Pre-compile regex pattern
pattern = re.compile(r"Left: (\d+) mm, Right: (\d+) mm, Front: (\d+) mm, Back: (\d+) mm, Up: (\d+) mm, Bottom: (\d+) mm, "
                     r"AccelX: ([-\d.]+) m/s\^2, AccelY: ([-\d.]+) m/s\^2, AccelZ: ([-\d.]+) m/s\^2, "
                     r"GyroX: ([-\d.]+) rad/s, GyroY: ([-\d.]+) rad/s, GyroZ: ([-\d.]+) rad/s")

def parse_sensor_data(line):
    try:
        match = pattern.match(line.strip())
        if match:
            accel_x = float(match.group(7))
            accel_y = float(match.group(8))
            accel_z = float(match.group(9)) - GRAVITY

            accel_x -= 0.1
            accel_y -= 0.6
            accel_z += 0.2

            data = {
                "Left": int(match.group(1)),
                "Right": int(match.group(2)),
                "Front": int(match.group(3)),
                "Back": int(match.group(4)),
                "Up": int(match.group(5)),
                "Bottom": int(match.group(6)),
                "AccelX": accel_x,
                "AccelY": accel_y,
                "AccelZ": accel_z,
                "GyroX": float(match.group(10)),
                "GyroY": float(match.group(11)),
                "GyroZ": float(match.group(12))
            }

            for sensor in ["Left", "Right", "Front", "Back", "Up", "Bottom"]:
                if data[sensor] in INVALID_READINGS:
                    logging.warning(f"Invalid reading for {sensor}: {data[sensor]} mm")
                    data[sensor] = None
            
            # Simulate bottom ToF sensor with random readings
            data["Bottom"] = random.choice([800, 1000, 1200])
            
            return data
        return None
    except Exception as e:
        logging.error(f"Error parsing sensor data: {e}")
        return None

def log_data(data, filename="sensor_data.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp] + [data.get(sensor, None) for sensor in [
            "Left", "Right", "Front", "Back", "Up", "Bottom",
            "AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"
        ]]
        writer.writerow(row)

def calibrate_sensors(sock, num_samples=1000):
    accel_x_samples = []
    accel_y_samples = []
    accel_z_samples = []
    gyro_x_samples = []
    gyro_y_samples = []
    gyro_z_samples = []
    
    logging.info("Calibrating sensors...")

    for _ in range(num_samples):
        try:
            data, _ = sock.recvfrom(1024)
            line = data.decode('utf-8').strip()
            if line:
                sensor_data = parse_sensor_data(line)
                if sensor_data:
                    accel_x_samples.append(sensor_data["AccelX"])
                    accel_y_samples.append(sensor_data["AccelY"])
                    accel_z_samples.append(sensor_data["AccelZ"])
                    gyro_x_samples.append(sensor_data["GyroX"])
                    gyro_y_samples.append(sensor_data["GyroY"])
                    gyro_z_samples.append(sensor_data["GyroZ"])
        except socket.timeout:
            continue
        except Exception as e:
            logging.error(f"Error during calibration: {e}")
            continue

    accel_x_mean = np.mean(accel_x_samples)
    accel_y_mean = np.mean(accel_y_samples)
    accel_z_mean = np.mean(accel_z_samples)
    gyro_x_mean = np.mean(gyro_x_samples)
    gyro_y_mean = np.mean(gyro_y_samples)
    gyro_z_mean = np.mean(gyro_z_samples)

    logging.info(f"Calibration complete. Means:\n"
                 f"AccelX: {accel_x_mean}, AccelY: {accel_y_mean}, AccelZ: {accel_z_mean}\n"
                 f"GyroX: {gyro_x_mean}, GyroY: {gyro_y_mean}, GyroZ: {gyro_z_mean}")
    
    return {
        "AccelX": accel_x_mean,
        "AccelY": accel_y_mean,
        "AccelZ": accel_z_mean,
        "GyroX": gyro_x_mean,
        "GyroY": gyro_y_mean,
        "GyroZ": gyro_z_mean
    }

def apply_low_pass_filter(gyro, alpha=0.1):
    for i in range(3):
        gyro[i] = alpha * gyro[i] + (1 - alpha) * gyro[i]
    return gyro

def update_orientation(quat, accel_data, gyro_data, dt):
    gyro_x_filter = KalmanFilter(Q=0.001, R=0.003)
    gyro_y_filter = KalmanFilter(Q=0.001, R=0.003)
    
    angle_x = gyro_x_filter.update(accel_data[0], gyro_data[0])
    angle_y = gyro_y_filter.update(accel_data[1], gyro_data[1])

    rot_x = R.from_euler('x', angle_x)
    rot_y = R.from_euler('y', angle_y)
    quat = rot_x * rot_y

    return quat.as_quat()

def plot_data(data_history, ax1, ax2, ax3, ax4, quat, kf, tof_sums, tof_counts):
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()

    # TOF Bar Chart with Cumulative Averages
    sensors = ["Left", "Right", "Front", "Back", "Up", "Bottom"]
    averages = {sensor: (tof_sums[sensor] / tof_counts[sensor] if tof_counts[sensor] > 0 else 0) for sensor in sensors}
    blues = plt.cm.Blues(np.linspace(0.3, 1, len(sensors)))
    colors = ['salmon' if averages[sensor] < 300 else blues[i] for i, sensor in enumerate(sensors)]
    
    positions = range(len(sensors))
    ax1.bar(positions, [averages[sensor] for sensor in sensors], color=colors)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(sensors, fontsize=13)
    ax1.set_title("TOF Sensor Cumulative Averages (mm)", fontsize=15)
    ax1.set_ylabel("Average Distance (mm)", fontsize=15)
    ax1.axhline(y=300, color='gray', linestyle='--')
    ax1.tick_params(axis='both', labelsize=13)
    
    # Display elapsed time
    elapsed_time = data_history[-1]['elapsed_time'] if data_history else 0
    ax1.text(0.05, 0.95, f"Elapsed Time: {elapsed_time:.2f} s", transform=ax1.transAxes, fontsize=13, verticalalignment='top')

    # Calculate and display Cumulative Mean Clearance (mean of all ToF readings over time)
    if data_history:
        all_tof_values = []
        for d in data_history:
            for sensor in sensors:
                if d[sensor] is not None:
                    all_tof_values.append(d[sensor])
        cumulative_mean_tof = np.mean(all_tof_values) if all_tof_values else np.nan
        ax1.text(0.05, 0.65, f"Mean Clearance: {cumulative_mean_tof:.1f} mm", transform=ax1.transAxes, fontsize=13, verticalalignment='top')

    # Calculate and display Breach Count for the current 10-minute trial
    if data_history:
        elapsed_time = data_history[-1]['elapsed_time']
        current_trial_number = int(elapsed_time // 600)  # 10 minutes = 600 seconds
        current_trial_start = current_trial_number * 600
        current_trial_data = [d for d in data_history if d['elapsed_time'] >= current_trial_start]
        breach_count = sum(1 for d in current_trial_data if any(d[sensor] is not None and d[sensor] < 300 for sensor in sensors))
    else:
        breach_count = 0
    ax1.text(0.05, 0.80, f"Breach Count: {breach_count}", transform=ax1.transAxes, fontsize=13, verticalalignment='top')

    # Accelerometer Plot with Time Elapsed
    if data_history:
        times = [d['elapsed_time'] for d in data_history]
        ax2.plot(times, [d['AccelX'] for d in data_history], label='AccelX')
        ax2.plot(times, [d['AccelY'] for d in data_history], label='AccelY')
        ax2.plot(times, [d['AccelZ'] for d in data_history], label='AccelZ')
        ax2.set_title("Accelerometer Data (m/s²)", fontsize=15)
        ax2.set_xlabel("Time Elapsed (s)", fontsize=15)
        ax2.set_ylabel("Acceleration (m/s²)", fontsize=15)
        ax2.legend(fontsize=13)
        ax2.tick_params(axis='both', labelsize=13)

    # Gyroscope Plot with Time Elapsed
    if data_history:
        ax3.plot(times, [d['GyroX'] for d in data_history], label='GyroX')
        ax3.plot(times, [d['GyroY'] for d in data_history], label='GyroY')
        ax3.plot(times, [d['GyroZ'] for d in data_history], label='GyroZ')
        ax3.set_title("Gyroscope Data (rad/s)", fontsize=15)
        ax3.set_xlabel("Time Elapsed (s)", fontsize=15)
        ax3.set_ylabel("Angular Velocity (rad/s)", fontsize=15)
        ax3.legend(fontsize=13)
        ax3.tick_params(axis='both', labelsize=13)

    # 3D Orientation Plot
    if quat is not None and len(quat) == 4:
        rot = R.from_quat(quat)
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        x_rot = rot.apply(x_axis)
        y_rot = rot.apply(y_axis)
        z_rot = rot.apply(z_axis)

        ax4.quiver(0, 0, 0, x_rot[0], x_rot[1], x_rot[2], color='r', length=0.2, normalize=True, label="X-Axis")
        ax4.quiver(0, 0, 0, y_rot[0], y_rot[1], y_rot[2], color='g', length=0.2, normalize=True, label="Y-Axis")
        ax4.quiver(0, 0, 0, z_rot[0], z_rot[1], z_rot[2], color='b', length=0.2, normalize=True, label="Z-Axis")
        
        ax4.set_xlim(-1, 1)
        ax4.set_ylim(-1, 1)
        ax4.set_zlim(-1, 1)
        ax4.set_title("3D Orientation (Quaternion-based)", fontsize=15)
        ax4.set_xlabel("X Axis", fontsize=15)
        ax4.set_ylabel("Y Axis", fontsize=15)
        ax4.set_zlabel("Z Axis", fontsize=15)
        ax4.grid(True)
        ax4.legend(fontsize=13)
    else:
        ax4.set_title("3D Orientation - No Data", fontsize=15)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.draw()
    plt.pause(0.1)

def main():
    sock = initialize_wifi()
    context, zmq_socket = setup_zmq_publisher()
    
    data_history = []
    quat = np.array([1, 0, 0, 0])
    kf = KalmanFilter(Q=0.001, R=0.003)
    
    sensor_offset = calibrate_sensors(sock)
    start_time = datetime.now()
    
    tof_sums = {sensor: 0 for sensor in ["Left", "Right", "Front", "Back", "Up", "Bottom"]}
    tof_counts = {sensor: 0 for sensor in ["Left", "Right", "Front", "Back", "Up", "Bottom"]}
    
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')

    while True:
        try:
            data, _ = sock.recvfrom(1024)
            line = data.decode('utf-8').strip()
            if line:
                sensor_data = parse_sensor_data(line)
                if sensor_data:
                    sensor_data['elapsed_time'] = (datetime.now() - start_time).total_seconds()
                    data_history.append(sensor_data)
                    log_data(sensor_data)
                    
                    # Create a new dictionary with renamed TOF sensor keys for ZMQ
                    zmq_data = {
                        "Left Clearance": sensor_data["Left"],
                        "Right Clearance": sensor_data["Right"],
                        "Front Clearance": sensor_data["Front"],
                        "Back Clearance": sensor_data["Back"],
                        "Up Clearance": sensor_data["Up"],
                        "Bottom Clearance": sensor_data["Bottom"],     
                        "AccelX": sensor_data["AccelX"],
                        "AccelY": sensor_data["AccelY"],
                        "AccelZ": sensor_data["AccelZ"],
                        "GyroX": sensor_data["GyroX"],
                        "GyroY": sensor_data["GyroY"],
                        "GyroZ": sensor_data["GyroZ"],
                        "elapsed_time": sensor_data["elapsed_time"]
                    }
                    send_zmq_message(zmq_socket, zmq_data)
                    
                    for sensor in ["Left", "Right", "Front", "Back", "Up", "Bottom"]:
                        if sensor_data[sensor] is not None:
                            tof_sums[sensor] += sensor_data[sensor]
                            tof_counts[sensor] += 1
                    
                    accel_data = [sensor_data["AccelX"], sensor_data["AccelY"], sensor_data["AccelZ"]]
                    gyro_data = [sensor_data["GyroX"], sensor_data["GyroY"], sensor_data["GyroZ"]]
                    quat = update_orientation(quat, accel_data, gyro_data, dt=0.1)

                    plot_data(data_history, ax1, ax2, ax3, ax4, quat, kf, tof_sums, tof_counts)
                    plt.pause(0.01)

        except socket.timeout:
            continue
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
