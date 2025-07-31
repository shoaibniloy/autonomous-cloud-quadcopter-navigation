import serial
import time
import re
from datetime import datetime
import math

# Serial port configuration
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
TIMEOUT = 0.02

# Gravity constant for unit conversion
GRAVITY = 9.8  # m/s^2

# Updated biases incorporating latest filtered residuals
BIAS_GYRO_X = -4.01  # deg/s (no change, as residual ~0.00)
BIAS_GYRO_Y = -1.15  # deg/s (no change, as residual ~0.00)
BIAS_GYRO_Z = 0.57   # deg/s (no change, as residual ~0.00)
BIAS_ACCEL_X = 0.04  # g (no change, as residual ~0.00)
BIAS_ACCEL_Y = -0.06 # g (previous -0.07 + residual 0.01)
BIAS_ACCEL_Z = -0.01 # g (no change, as residual ~0.00)

def initialize_serial():
    while True:
        try:
            ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=TIMEOUT)
            print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
            return ser
        except serial.SerialException as e:
            print(f"Failed to connect to {SERIAL_PORT}: {e}")
            time.sleep(0.5)

# Pre-compile regex pattern (trimmed to capture only IMU parts, but matches full for compatibility)
pattern = re.compile(r"Left: (\d+) mm, Right: (\d+) mm, Front: (\d+) mm, Back: (\d+) mm, Up: (\d+) mm, Bottom: (\d+) mm, "
                     r"AccelX: ([-\d.]+) m/s\^2, AccelY: ([-\d.]+) m/s\^2, AccelZ: ([-\d.]+) m/s\^2, "
                     r"GyroX: ([-\d.]+) rad/s, GyroY: ([-\d.]+) rad/s, GyroZ: ([-\d.]+) rad/s")

def parse_sensor_data(line):
    try:
        match = pattern.match(line.strip())
        if match:
            # Raw values in original units
            accel_x_raw = float(match.group(7)) / GRAVITY  # Convert to g
            accel_y_raw = float(match.group(8)) / GRAVITY  # Convert to g
            accel_z_raw = float(match.group(9)) / GRAVITY  # Convert to g
            gyro_x_raw = float(match.group(10)) * (180 / math.pi)  # Convert to deg/s
            gyro_y_raw = float(match.group(11)) * (180 / math.pi)  # Convert to deg/s
            gyro_z_raw = float(match.group(12)) * (180 / math.pi)  # Convert to deg/s
            
            # Apply bias correction
            accel_x = accel_x_raw - BIAS_ACCEL_X
            accel_y = accel_y_raw - BIAS_ACCEL_Y
            accel_z = accel_z_raw - BIAS_ACCEL_Z
            gyro_x = gyro_x_raw - BIAS_GYRO_X
            gyro_y = gyro_y_raw - BIAS_GYRO_Y
            gyro_z = gyro_z_raw - BIAS_GYRO_Z
            
            return {
                "AccelX": accel_x,
                "AccelY": accel_y,
                "AccelZ": accel_z,
                "GyroX": gyro_x,
                "GyroY": gyro_y,
                "GyroZ": gyro_z
            }
        return None
    except Exception as e:
        print(f"Error parsing sensor data: {e}")
        return None

def main():
    ser = initialize_serial()
    start_time = datetime.now()
    
    # Print header
    print("Time (s),Gyroscope X (deg/s),Gyroscope Y (deg/s),Gyroscope Z (deg/s),Accelerometer X (g),Accelerometer Y (g),Accelerometer Z (g)")
    
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                data = parse_sensor_data(line)
                if data:
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    print(f"{elapsed_time:.2f},{data['GyroX']:.2f},{data['GyroY']:.2f},{data['GyroZ']:.2f},{data['AccelX']:.2f},{data['AccelY']:.2f},{data['AccelZ']:.2f}")
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()