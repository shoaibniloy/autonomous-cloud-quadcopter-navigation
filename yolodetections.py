import time
import cv2
import logging
import torch
from ultralytics import YOLO
import numpy as np

# Camera parameters and calibration (unchanged)
camera_matrix = np.array([[579.04358002, 0., 314.84761866],
                          [0., 584.13155795, 184.69953967],
                          [0., 0., 1.]])
distortion_coefficients = np.array([[-6.76100761e-01, 1.49113089e+01, 7.15002649e-03, 4.09622163e-03, -9.12554991e+01]])

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

logging.basicConfig(filename='yolo.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    model = YOLO('furnitureepoch50.pt').to(device)
    logging.info("YOLO model loaded successfully on device: {}".format(device))
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    exit(1)

buffer_time = 5.0
detection_buffer = []
start_time = time.time()

CLASS_MAPPINGS = {
    0: 'appliances',
    1: 'bathtub',
    2: 'bed',
    3: 'book',
    4: 'ceiling light',
    5: 'ceiling-fan',
    6: 'chair',
    7: 'crib',
    8: 'electronics',
    9: 'faucet',
    10: 'floor decor',
    11: 'glass decor',
    12: 'lamps',
    13: 'linens',
    14: 'outdoor-misc',
    15: 'pillows',
    16: 'plants',
    17: 'pots',
    18: 'rugs',
    19: 'sconce',
    20: 'shelves',
    21: 'sink',
    22: 'small seating',
    23: 'sofa',
    24: 'storage',
    25: 'table decor',
    26: 'tables',
    27: 'toilet',
    28: 'wall decor'
}

def process_yolo_frame(frame):
    global detection_buffer, start_time

    detections = []
    results = model(frame, verbose=False)
    detections.extend(results[0].boxes.data.tolist())

    detection_buffer.extend(detections)

    # Clear buffer every buffer_time seconds (currently just clears)
    if time.time() - start_time >= buffer_time:
        detection_buffer.clear()
        start_time = time.time()

    frame_copy = frame.copy()
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        label = CLASS_MAPPINGS.get(int(class_id), 'unknown')
        cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame_copy, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame_copy

def main():
    cap = cv2.VideoCapture(0)  # Use webcam; change to file path if needed

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    print("Press 'q' to quit the window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLO detection and draw boxes
        processed_frame = process_yolo_frame(frame)

        # Show the processed frame
        cv2.imshow('YOLO Detections', processed_frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
