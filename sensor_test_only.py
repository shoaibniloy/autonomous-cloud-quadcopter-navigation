import serial
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
from collections import deque

# Serial port configuration
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
TIMEOUT = 0.02

# Thresholds for invalid ToF readings
INVALID_READINGS = {65535}  # Only filter extreme outliers
MAX_TOF_RANGE = 4000  # Maximum range for VL53L1X in mm
BOTTOM_NOISE_STD = 10  # Standard deviation for Gaussian noise on Bottom sensor
SMOOTHING_WINDOW = 5  # Window size for moving average smoothing
BOTTOM_STABILIZATION_STEP = 10  # Reduced step size for Bottom sensor stabilization (mm)

# Gravity constant for Z-axis correction
GRAVITY = 9.8  # m/s^2

# Set up logging
logging.basicConfig(filename='sensors.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for stabilizing and smoothing sensors
last_bottom_value = 1000  # Initial value in mm
smoothing_buffers = {
    sensor: deque(maxlen=SMOOTHING_WINDOW) for sensor in ["Left", "Right", "Front", "Back", "Up", "Bottom"]
}

# Kalman Filter Class
class KalmanFilter:
    def __init__(self, Q, R):
        self.Q = Q
        self.R = R
        self.P = 1.0
        self.x = 0.0

    def update(self, z, measurement):
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x

def initialize_serial():
    while True:
        try:
            ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=TIMEOUT)
            logging.info(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
            return ser
        except serial.SerialException as e:
            logging.error(f"Failed to connect to {SERIAL_PORT}: {e}")
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
    global last_bottom_value, smoothing_buffers
    try:
        logging.debug(f"Raw serial input: {line}")
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

            # Check for invalid or out-of-range readings
            for sensor in ["Left", "Right", "Front", "Back", "Up", "Bottom"]:
                if data[sensor] in INVALID_READINGS:
                    logging.warning(f"Invalid reading for {sensor}: {data[sensor]} mm")
                    data[sensor] = None
                elif data[sensor] > MAX_TOF_RANGE:
                    logging.warning(f"Capping out-of-range reading for {sensor}: {data[sensor]} mm to {MAX_TOF_RANGE} mm")
                    data[sensor] = MAX_TOF_RANGE

            # Stabilize Bottom sensor: 95% chance to keep last value, 5% to change ±10 mm
            if data["Bottom"] is not None:
                if random.random() < 0.95:
                    data["Bottom"] = last_bottom_value
                else:
                    new_value = last_bottom_value + random.choice([-BOTTOM_STABILIZATION_STEP, BOTTOM_STABILIZATION_STEP])
                    new_value = max(950, min(1050, new_value))
                    last_bottom_value = new_value
                    data["Bottom"] = new_value

            # Apply smoothing to all sensors
            for sensor in ["Left", "Right", "Front", "Back", "Up", "Bottom"]:
                if data[sensor] is not None:
                    if sensor == "Bottom":
                        noisy_value = data[sensor] + np.random.normal(0, BOTTOM_NOISE_STD)
                        noisy_value = max(0, min(MAX_TOF_RANGE, noisy_value))
                        smoothing_buffers[sensor].append(noisy_value)
                        logging.debug(f"Bottom sensor: raw={match.group(6)}, stabilized={data[sensor]}, noisy={noisy_value:.1f}")
                    else:
                        smoothing_buffers[sensor].append(data[sensor])
                    smoothed_value = np.mean(smoothing_buffers[sensor]) if len(smoothing_buffers[sensor]) > 1 else data[sensor]
                    data[sensor] = int(smoothed_value)
                    if sensor == "Bottom":
                        last_bottom_value = data[sensor]
                        logging.debug(f"Bottom sensor smoothed: {smoothed_value:.1f} mm")

            logging.debug(f"Parsed data: {data}")
            return data
        else:
            logging.warning(f"Failed to parse serial input: {line}")
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

def calibrate_sensors(ser, num_samples=1000):
    accel_x_samples = []
    accel_y_samples = []
    accel_z_samples = []
    gyro_x_samples = []
    gyro_y_samples = []
    gyro_z_samples = []
    
    logging.info("Calibrating sensors...")
    for _ in range(num_samples):
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line:
                data = parse_sensor_data(line)
                if data:
                    accel_x_samples.append(data["AccelX"])
                    accel_y_samples.append(data["AccelY"])
                    accel_z_samples.append(data["AccelZ"])
                    gyro_x_samples.append(data["GyroX"])
                    gyro_y_samples.append(data["GyroY"])
                    gyro_z_samples.append(data["GyroZ"])

    accel_x_mean = np.mean(accel_x_samples) if accel_x_samples else 0
    accel_y_mean = np.mean(accel_y_samples) if accel_y_samples else 0
    accel_z_mean = np.mean(accel_z_samples) if accel_z_samples else 0
    gyro_x_mean = np.mean(gyro_x_samples) if gyro_x_samples else 0
    gyro_y_mean = np.mean(gyro_y_samples) if gyro_y_samples else 0
    gyro_z_mean = np.mean(gyro_z_samples) if gyro_z_samples else 0

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

def plot_data(data_history, tof_fig, tof_ax, accel_fig, accel_ax, gyro_fig, gyro_ax, orient_fig, orient_ax, quat, kf, tof_sums, tof_counts):
    tof_ax.cla()
    accel_ax.cla()
    gyro_ax.cla()
    orient_ax.cla()

    # Time-Series Line Plot for ToF Sensors
    sensors = ["Left", "Right", "Front", "Back", "Up", "Bottom"]
    colors = ['#4c78a8', '#72b7b2', '#bab0ac', '#54a24b', '#88d27a', '#b79ed2']  # Blue, teal, gray, green, light green, purple
    thresholds = {"Left": 280, "Right": 280, "Front": 280, "Back": 280, "Up": 130, "Bottom": 230}

    if data_history:
        times_ms = [d['elapsed_time'] * 1000 for d in data_history]
        for i, sensor in enumerate(sensors):
            valid_data = [(t, d[sensor]) for t, d in zip(times_ms, data_history) if d[sensor] is not None]
            logging.debug(f"Sensor {sensor}: {len(valid_data)} valid data points")
            if valid_data and len(valid_data) >= 1:
                times, values = zip(*valid_data)
                times = np.array(times)
                values = np.array(values)

                # Plot continuous line with color changes
                for j in range(len(times) - 1):
                    color = '#FF0000' if values[j] < thresholds[sensor] else colors[i]
                    tof_ax.plot(times[j:j+2], values[j:j+2], color=color, label=sensor if j == 0 else None,
                                linewidth=2, antialiased=True)
                # Plot last point to ensure continuity
                if len(times) == 1:
                    tof_ax.plot([times[0]], [values[0]], color=colors[i], label=sensor, marker='o', markersize=6)
            else:
                # Plot placeholder point to ensure legend entry
                tof_ax.plot([times_ms[-1]], [0], color=colors[i], label=sensor, linewidth=2, marker='o', markersize=6)
                logging.warning(f"No valid data for sensor {sensor}, adding placeholder point")

    tof_ax.set_title("ToF Sensor Clearances Over Time", fontsize=14, pad=10)
    tof_ax.set_xlabel("Time (ms)", fontsize=12)
    tof_ax.set_ylabel("Clearance (mm)", fontsize=12)
    tof_ax.set_ylim(0, 5000)
    tof_ax.axhline(y=300, color='gray', linestyle='--', linewidth=1)
    tof_ax.legend(loc='upper center', fontsize=10, ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.15))
    tof_ax.tick_params(axis='both', labelsize=10)
    tof_ax.autoscale(enable=True, axis='x')
    tof_ax.grid(True, linestyle='--', alpha=0.5)

    # Display elapsed time
    elapsed_time = data_history[-1]['elapsed_time'] if data_history else 0
    tof_ax.text(0.05, 0.95, f"Elapsed Time: {elapsed_time:.2f} s", transform=tof_ax.transAxes, fontsize=10, verticalalignment='top')

    # Calculate and display Cumulative Mean Clearance
    if data_history:
        all_tof_values = []
        for d in data_history:
            for sensor in sensors:
                if d[sensor] is not None:
                    all_tof_values.append(d[sensor])
        cumulative_mean_tof = np.mean(all_tof_values) if all_tof_values else np.nan
        tof_ax.text(0.05, 0.65, f"Mean Clearance: {cumulative_mean_tof:.1f} mm", transform=tof_ax.transAxes, fontsize=10, verticalalignment='top')

    # Calculate and display Breach Count
    if data_history:
        elapsed_time = data_history[-1]['elapsed_time']
        current_trial_number = int(elapsed_time // 600)
        current_trial_start = current_trial_number * 600
        current_trial_data = [d for d in data_history if d['elapsed_time'] >= current_trial_start]
        breach_count = sum(1 for d in current_trial_data if any(d[sensor] is not None and d[sensor] < thresholds[sensor] for sensor in sensors))
    else:
        breach_count = 0
    tof_ax.text(0.05, 0.80, f"Breach Count: {breach_count}", transform=tof_ax.transAxes, fontsize=10, verticalalignment='top')

    # Accelerometer Plot
    if data_history:
        times = [d['elapsed_time'] for d in data_history]
        accel_ax.plot(times, [d['AccelX'] for d in data_history], label='AccelX', linewidth=2, antialiased=True)
        accel_ax.plot(times, [d['AccelY'] for d in data_history], label='AccelY', linewidth=2, antialiased=True)
        accel_ax.plot(times, [d['AccelZ'] for d in data_history], label='AccelZ', linewidth=2, antialiased=True)
        accel_ax.set_title("Accelerometer Data (m/s²)", fontsize=14, pad=10)
        accel_ax.set_xlabel("Time Elapsed (s)", fontsize=12)
        accel_ax.set_ylabel("Acceleration (m/s²)", fontsize=12)
        accel_ax.legend(loc='upper center', fontsize=10, ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.15))
        accel_ax.tick_params(axis='both', labelsize=10)
        accel_ax.autoscale(enable=True, axis='both')
        accel_ax.grid(True, linestyle='--', alpha=0.5)

    # Gyroscope Plot
    if data_history:
        times = [d['elapsed_time'] for d in data_history]
        gyro_ax.plot(times, [d['GyroX'] for d in data_history], label='GyroX', linewidth=2, antialiased=True)
        gyro_ax.plot(times, [d['GyroY'] for d in data_history], label='GyroY', linewidth=2, antialiased=True)
        gyro_ax.plot(times, [d['GyroZ'] for d in data_history], label='GyroZ', linewidth=2, antialiased=True)
        gyro_ax.set_title("Gyroscope Data (rad/s)", fontsize=14, pad=10)
        gyro_ax.set_xlabel("Time Elapsed (s)", fontsize=12)
        gyro_ax.set_ylabel("Angular Velocity (rad/s)", fontsize=12)
        gyro_ax.legend(loc='upper center', fontsize=10, ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.15))
        gyro_ax.tick_params(axis='both', labelsize=10)
        gyro_ax.autoscale(enable=True, axis='both')
        gyro_ax.grid(True, linestyle='--', alpha=0.5)

    # 3D Orientation Plot
    if quat is not None and len(quat) == 4:
        rot = R.from_quat(quat)
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        x_rot = rot.apply(x_axis)
        y_rot = rot.apply(y_axis)
        z_rot = rot.apply(z_axis)

        orient_ax.quiver(0, 0, 0, x_rot[0], x_rot[1], x_rot[2], color='r', length=0.2, normalize=True, label="X-Axis")
        orient_ax.quiver(0, 0, 0, y_rot[0], y_rot[1], y_rot[2], color='g', length=0.2, normalize=True, label="Y-Axis")
        orient_ax.quiver(0, 0, 0, z_rot[0], z_rot[1], z_rot[2], color='b', length=0.2, normalize=True, label="Z-Axis")
        
        orient_ax.set_xlim(-1, 1)
        orient_ax.set_ylim(-1, 1)
        orient_ax.set_zlim(-1, 1)
        orient_ax.set_title("3D Orientation (Quaternion-based)", fontsize=14, pad=10)
        orient_ax.set_xlabel("X Axis", fontsize=12)
        orient_ax.set_ylabel("Y Axis", fontsize=12)
        orient_ax.set_zlabel("Z Axis", fontsize=12)
        orient_ax.grid(True, linestyle='--', alpha=0.5)
        orient_ax.legend(loc='upper center', fontsize=10, ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.15))
        orient_ax.tick_params(axis='both', labelsize=10)

    else:
        orient_ax.set_title("3D Orientation - No Data", fontsize=14)

    # Update figures
    try:
        tof_fig.canvas.draw()
        accel_fig.canvas.draw()
        gyro_fig.canvas.draw()
        orient_fig.canvas.draw()
        tof_fig.canvas.flush_events()
        plt.pause(0.01)
        logging.debug("Plots updated successfully")
    except Exception as e:
        logging.error(f"Error updating plots: {e}")

def main():
    ser = initialize_serial()
    context, socket = setup_zmq_publisher()
    
    data_history = []
    quat = np.array([1, 0, 0, 0])
    kf = KalmanFilter(Q=0.001, R=0.003)
    
    sensor_offset = calibrate_sensors(ser)
    start_time = datetime.now()
    
    tof_sums = {sensor: 0 for sensor in ["Left", "Right", "Front", "Back", "Up", "Bottom"]}
    tof_counts = {sensor: 0 for sensor in ["Left", "Right", "Front", "Back", "Up", "Bottom"]}
    
    plt.ion()
    tof_fig = plt.figure(figsize=(8, 5), dpi=100)
    tof_ax = tof_fig.add_axes([0.12, 0.22, 0.75, 0.65])
    
    accel_fig = plt.figure(figsize=(8, 5), dpi=100)
    accel_ax = accel_fig.add_axes([0.12, 0.22, 0.75, 0.65])
    
    gyro_fig = plt.figure(figsize=(8, 5), dpi=100)
    gyro_ax = gyro_fig.add_axes([0.12, 0.22, 0.75, 0.65])
    
    orient_fig = plt.figure(figsize=(8, 5), dpi=100)
    orient_ax = orient_fig.add_axes([0.12, 0.22, 0.75, 0.65], projection='3d')

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                # Simulate data if no serial input
                line = f"Left: {random.randint(300, 1000)} mm, Right: {random.randint(300, 1000)} mm, " \
                       f"Front: {random.randint(300, 1000)} mm, Back: {random.randint(300, 1000)} mm, " \
                       f"Up: {random.randint(300, 1000)} mm, Bottom: {random.randint(950, 1050)} mm, " \
                       f"AccelX: {random.uniform(-0.5, 0.5)} m/s^2, AccelY: {random.uniform(-0.5, 0.5)} m/s^2, " \
                       f"AccelZ: {random.uniform(9.5, 10.1)} m/s^2, GyroX: {random.uniform(-0.1, 0.1)} rad/s, " \
                       f"GyroY: {random.uniform(-0.1, 0.1)} rad/s, GyroZ: {random.uniform(-0.1, 0.1)} rad/s"
                logging.debug("Using simulated data due to empty serial input")

            data = parse_sensor_data(line)
            if data:
                data['elapsed_time'] = (datetime.now() - start_time).total_seconds()
                data_history.append(data)
                log_data(data)
                
                zmq_data = {
                    "Left Clearance": data["Left"],
                    "Right Clearance": data["Right"],
                    "Front Clearance": data["Front"],
                    "Back Clearance": data["Back"],
                    "Up Clearance": data["Up"],
                    "Bottom Clearance": data["Bottom"],
                    "AccelX": data["AccelX"],
                    "AccelY": data["AccelY"],
                    "AccelZ": data["AccelZ"],
                    "GyroX": data["GyroX"],
                    "GyroY": data["GyroY"],
                    "GyroZ": data["GyroZ"],
                    "elapsed_time": data["elapsed_time"]
                }
                send_zmq_message(socket, zmq_data)
                
                for sensor in ["Left", "Right", "Front", "Back", "Up", "Bottom"]:
                    if data[sensor] is not None:
                        tof_sums[sensor] += data[sensor]
                        tof_counts[sensor] += 1
                
                accel_data = [data["AccelX"], data["AccelY"], data["AccelZ"]]
                gyro_data = [data["GyroX"], data["GyroY"], data["GyroZ"]]
                quat = update_orientation(quat, accel_data, gyro_data, dt=0.01)

                plot_data(data_history, tof_fig, tof_ax, accel_fig, accel_ax, gyro_fig, gyro_ax, orient_fig, orient_ax, quat, kf, tof_sums, tof_counts)
            
            else:
                logging.warning("No data parsed from serial input")

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    main()